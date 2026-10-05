import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from data.config import NOMINAL
from data.trajectory_gen import fly_scenario
from filters.knet import numpy_forward
from filters.knet.gain_network import KalmanGainGRU
from filters.knet.knet_filter import KalmanNetFilter
from filters.knet.rkn_filter import RKNFilter
from filters.knet.rkn_network import RKNGainCovNetwork
from filters.motion_models import cv_model
from filters.state import STATE_DIM
from scripts.eval_knet import _INIT_OFFSET, _priors
from scripts.replay import (
    RKN_LAYER_MULTS,
    error_and_nees,
    knet_errors,
    knet_errors_numpy,
    numpy_weights,
    rkn_run,
    rkn_run_numpy,
)

REPO = Path(__file__).resolve().parents[1]
SEED_SCENARIO = REPO / "data" / "scenarios" / "break_turn_9g.json"
STAND_IN_SEED = 7
GRU_WEIGHT_SCALE = 0.1
KNET_WEIGHT_SCALE, KNET_GAIN_LAYER_SCALE = 0.1, 3e-3
RKN_WEIGHT_SCALE = 0.03
GRU_INPUT, GRU_HIDDEN = 5, 4
P0_VARIANCE = 100.0
SINGLE_STEP_ATOL = 1e-15
SEQUENCE_ATOL_STATE = 1e-10
SEQUENCE_ATOL_P = 5e-9
SEQUENCE_RTOL_NEES = 1e-10
GAIN_EFFECT_FLOOR_M = 1.0


def stand_in(net, scale):
    torch.manual_seed(STAND_IN_SEED)
    for parameter in net.parameters():
        torch.nn.init.uniform_(parameter, -scale, scale)
    return net.double().eval()


def knet_stand_in():
    """A gain layer at the hidden layers' 0.1 diverges over the scenario; 3e-3 stays bounded and moves x by 800 m."""
    net = stand_in(KalmanGainGRU(), KNET_WEIGHT_SCALE)
    for parameter in net.FC2[-1].parameters():
        torch.nn.init.uniform_(parameter, -KNET_GAIN_LAYER_SCALE, KNET_GAIN_LAYER_SCALE)
    return net


def rkn_stand_in():
    return stand_in(RKNGainCovNetwork(**RKN_LAYER_MULTS), RKN_WEIGHT_SCALE)


def exported(net, tmp_path):
    checkpoint = tmp_path / "stand_in.pt"
    torch.save(net.state_dict(), checkpoint)
    return numpy_weights(checkpoint)


@pytest.fixture(scope="module")
def flight():
    return fly_scenario(json.loads(SEED_SCENARIO.read_text()))


def test_numpy_gru_cell_steps_like_torch_gru_with_seeded_weights(tmp_path):
    """MS-10: the learned gain runs through torch.nn.GRU's gate equations, reset gate scaling W_hn h + b_hn.
    The new hidden state lies in (-1, 1), so atol 1e-15 is ~9 ulp of one float64 step; measured 0."""
    gru = stand_in(torch.nn.GRU(GRU_INPUT, GRU_HIDDEN), GRU_WEIGHT_SCALE)
    x, h = torch.rand(1, 1, GRU_INPUT, dtype=torch.float64), torch.rand(1, 1, GRU_HIDDEN, dtype=torch.float64)

    with torch.no_grad():
        _, h_torch = gru(x, h)
    h_numpy = numpy_forward.gru_cell(exported(gru, tmp_path), "", x[0, 0].numpy(), h[0, 0].numpy())

    np.testing.assert_allclose(h_numpy, h_torch[0, 0].numpy(), rtol=0.0, atol=SINGLE_STEP_ATOL)


def test_numpy_kalmannet_matches_torch_at_every_step_of_the_committed_scenario(flight, tmp_path):
    """MS-10: same CV f, h and learned gain. Checkpoints are float64 (both trainers call .double()); summation
    order drifts x by 3.6e-12 over the 296 steps at |x| <= 3.4e3, so atol 1e-10 is ~30x headroom."""
    states, z = flight
    net = knet_stand_in()
    priors = _priors(NOMINAL.dt, NOMINAL.R())
    torch_filter = KalmanNetFilter(net, NOMINAL.dt)
    torch_filter.init_sequence(torch.tensor(states[None, 0]) + _INIT_OFFSET, *priors)
    numpy_filter = numpy_forward.KalmanNetFilter(numpy_forward.KalmanGainGRU(exported(net, tmp_path)), NOMINAL.dt)
    numpy_filter.init_sequence(states[0] + _INIT_OFFSET.numpy(), *(prior.numpy() for prior in priors))
    F, _ = cv_model(NOMINAL.dt)
    open_loop = states[0] + _INIT_OFFSET.numpy()

    with torch.no_grad():
        for z_t in z:
            x_torch = torch_filter.step(torch.tensor(z_t)[None])[0].numpy()
            x_numpy = numpy_filter.step(z_t)
            open_loop = F @ open_loop
            assert np.all(np.isfinite(x_numpy))
            np.testing.assert_allclose(x_numpy, x_torch, rtol=0.0, atol=SEQUENCE_ATOL_STATE)

    assert np.linalg.norm(x_numpy[:3] - open_loop[:3]) > GAIN_EFFECT_FLOOR_M


def test_numpy_rkn_matches_torch_state_and_covariance_at_every_step_of_the_committed_scenario(flight, tmp_path):
    """MS-10: same CV f, h and H, learned K and B. float64 summation order over the 296 steps drifts x by 9.3e-12
    (atol 1e-10, ~10x) and P by 4.6e-10 at |P| <= 4e4 (atol 5e-9, ~10x)."""
    states, z = flight
    net = rkn_stand_in()
    x0, P0 = states[0] + _INIT_OFFSET.numpy(), np.eye(STATE_DIM) * P0_VARIANCE
    torch_filter = RKNFilter(net, NOMINAL.dt)
    torch_filter.init_sequence(torch.tensor(x0[None]), torch.tensor(P0[None]))
    numpy_filter = numpy_forward.RKNFilter(numpy_forward.RKNGainCovNetwork(exported(net, tmp_path)), NOMINAL.dt)
    numpy_filter.init_sequence(x0, P0)
    F, _ = cv_model(NOMINAL.dt)
    open_loop = x0

    with torch.no_grad():
        for z_t in z:
            x_torch, P_torch = torch_filter.step(torch.tensor(z_t)[None])
            x_numpy, P_numpy = numpy_filter.step(z_t)
            open_loop = F @ open_loop
            assert np.all(np.isfinite(x_numpy)) and np.all(np.isfinite(P_numpy))
            np.testing.assert_allclose(x_numpy, x_torch[0].numpy(), rtol=0.0, atol=SEQUENCE_ATOL_STATE)
            np.testing.assert_allclose(P_numpy, P_torch[0].numpy(), rtol=0.0, atol=SEQUENCE_ATOL_P)

    assert np.linalg.norm(x_numpy[:3] - open_loop[:3]) > GAIN_EFFECT_FLOOR_M


def test_numpy_forward_loads_without_torch_for_pyodide():
    """MS-10: the page runs the learned filters in Pyodide, which has numpy and no torch (#2 decision 5)."""
    probe = "import sys, filters.knet.numpy_forward; print('torch' in sys.modules)"

    result = subprocess.run([sys.executable, "-c", probe], cwd=REPO, capture_output=True, text=True)

    assert result.stdout.strip() == "False", result.stderr


def test_replay_numpy_backend_prints_the_torch_backend_learned_filter_flags(tmp_path):
    """MS-10: the learned filters replayed through the numpy forward trip each flag on the torch path's step."""
    knet, rkn, missing = tmp_path / "knet.pt", tmp_path / "rkn.pt", tmp_path / "missing"
    torch.save(knet_stand_in().state_dict(), knet)
    torch.save(rkn_stand_in().state_dict(), rkn)

    def learned_lines(backend):
        args = [SEED_SCENARIO, "--imm-tuning", missing, "--knet-checkpoint", knet, "--rkn-checkpoint", rkn]
        result = subprocess.run(
            [sys.executable, "scripts/replay.py", *map(str, args), "--learned-backend", backend],
            cwd=REPO, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        return [line for line in result.stdout.splitlines() if line.startswith(("KalmanNet:", "RKN:"))]

    torch_lines = learned_lines("torch")

    assert learned_lines("numpy") == torch_lines
    assert [line.split(":")[0] for line in torch_lines] == ["KalmanNet", "RKN"]
    assert all("raw_error=" in line for line in torch_lines)


def test_numpy_normalize_scales_a_sub_eps_feature_by_one_over_eps_like_torch():
    """MS-10: KalmanNet's input features go through F.normalize, which divides by eps below norm 1e-12."""
    tiny = np.array([2.0**-42, 0.0, 0.0])

    expected = torch.nn.functional.normalize(torch.tensor(tiny), p=2, dim=-1, eps=1e-12).numpy()

    np.testing.assert_array_equal(numpy_forward.normalize(tiny), expected)


def test_replay_numpy_rkn_errors_and_nees_match_the_torch_backend_from_the_same_start(flight, tmp_path):
    """MS-10: the overconfidence flag reads NEES, so both backends start from x0 + offset and P0 = 100 I.
    NEES spans 4e-2..1e6 and drifts by 6.6e-12 relative in float64, so rtol 1e-10 is ~15x headroom."""
    states, z = flight
    checkpoint = tmp_path / "rkn.pt"
    torch.save(rkn_stand_in().state_dict(), checkpoint)

    torch_err, torch_nees = rkn_run(checkpoint, states, z)
    numpy_err, numpy_nees = rkn_run_numpy(checkpoint, states, z)

    np.testing.assert_allclose(numpy_err, torch_err, rtol=0.0, atol=SEQUENCE_ATOL_STATE)
    assert np.all(np.isfinite(numpy_nees))
    np.testing.assert_allclose(numpy_nees, torch_nees, rtol=SEQUENCE_RTOL_NEES, atol=0.0)


def test_diverged_rkn_estimate_reads_a_signed_bounded_error_and_singular_p_a_huge_nees():
    """MS-10: a diverged numpy estimate trips raw error the way eval_rkn's torch run records it."""
    x = np.array([np.inf, -np.inf, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    e, nees = error_and_nees(x, np.zeros((STATE_DIM, STATE_DIM)), np.zeros(STATE_DIM))

    np.testing.assert_array_equal(e[:3], [1e5, -1e5, 1e5])
    assert nees == 1e10


def test_replay_numpy_kalmannet_errors_match_the_torch_backend_from_the_same_start(flight, tmp_path):
    """MS-10: both backends start KalmanNet from x0 + offset and the same Q, Sigma, S hidden priors.
    Same float64 drift budget as the filter-level comparison: atol 1e-10."""
    states, z = flight
    checkpoint = tmp_path / "knet.pt"
    torch.save(knet_stand_in().state_dict(), checkpoint)

    torch_err = knet_errors(checkpoint, states, z, NOMINAL.R())
    numpy_err = knet_errors_numpy(checkpoint, states, z, NOMINAL.R())

    assert np.all(np.isfinite(numpy_err))
    np.testing.assert_allclose(numpy_err, torch_err, rtol=0.0, atol=SEQUENCE_ATOL_STATE)


def test_numpy_wrap_measurement_diff_matches_torch_on_azimuth_and_elevation_at_pi():
    """MS-34/MS-2: the Pyodide mirror wraps azimuth to [-pi, pi) and leaves elevation alone, like torch."""
    from filters.torch_ops import wrap_measurement_diff as wrap_measurement_diff_torch

    diff = np.array([np.pi, np.pi, 5.0])
    expected = wrap_measurement_diff_torch(torch.tensor(diff)).numpy()
    assert np.array_equal(numpy_forward.wrap_measurement_diff(diff), expected)
