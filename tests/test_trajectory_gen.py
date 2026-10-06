from dataclasses import asdict

import numpy as np

from data.config import NOMINAL, Regime
from data.fixed_wing import DEFAULT_PRESET, PRESETS, STANDARD_GRAVITY_M_S2
from data.trajectory_gen import fly_scenario, generate_ground_truth, measure


def test_ground_truth_has_no_nans_and_switches_modes():
    rng = np.random.default_rng(0)
    small = Regime(name="test_small", T=300)
    states, labels = generate_ground_truth(rng, small)
    assert states.shape == (300, 9)
    assert np.all(np.isfinite(states))
    assert set(np.unique(labels)).issubset({0, 1, 2})
    assert len(np.unique(labels)) >= 2  # switches modes within 300 steps


def test_generation_is_reproducible_given_same_seed():
    regime = Regime(name="test_repro", T=100)
    s1, l1 = generate_ground_truth(np.random.default_rng(42), regime)
    s2, l2 = generate_ground_truth(np.random.default_rng(42), regime)
    assert np.array_equal(s1, s2)
    assert np.array_equal(l1, l2)


def test_measurements_stay_within_wrapped_azimuth_range():
    rng = np.random.default_rng(1)
    regime = Regime(name="test_meas", T=200)
    states, _ = generate_ground_truth(rng, regime)
    z = measure(np.random.default_rng(2), states, regime)
    assert np.all(z[:, 0] >= -np.pi) and np.all(z[:, 0] < np.pi)
    assert np.all(z[:, 2] > 0)  # range stays positive for these initial conditions


def test_heavy_tailed_regime_has_larger_measurement_spread():
    T = 2000
    gaussian = Regime(name="g", T=T, heavy_tailed=False)
    heavy = Regime(name="h", T=T, heavy_tailed=True, heavy_tailed_dof=3.0)

    rng = np.random.default_rng(7)
    states, _ = generate_ground_truth(rng, gaussian)

    z_gauss = measure(np.random.default_rng(8), states, gaussian)
    z_heavy = measure(np.random.default_rng(8), states, heavy)

    from filters.measurement_model import h
    resid_gauss = np.array([z_gauss[t, 2] - h(states[t])[2] for t in range(T)])
    resid_heavy = np.array([z_heavy[t, 2] - h(states[t])[2] for t in range(T)])
    assert np.std(resid_heavy) > np.std(resid_gauss)


START_SPEED_SEED_3_M_S = 153.89726829480458


def level_scenario(seed=3, noise_regime="nominal", steps=50):
    return {
        "seed": seed,
        "envelope": asdict(PRESETS[DEFAULT_PRESET]),
        "noise_regime": noise_regime,
        "controls": [[0.0, 1.0, 0.5]] * steps,
    }


def test_scenario_replays_to_the_identical_flight_and_measurements():
    """MS-3: one measurement per fixed 0.02 s step, none dropped."""
    first_states, first_z = fly_scenario(level_scenario())
    second_states, second_z = fly_scenario(level_scenario())

    assert first_states.shape == (50, 9) and first_z.shape == (50, 3)
    assert np.array_equal(first_states, second_states) and np.array_equal(first_z, second_z)


def test_scenario_starts_inside_the_regime_range_and_speed_bounds():
    """MS-19."""
    for seed in range(20):
        states, _ = fly_scenario(level_scenario(seed=seed, steps=1))
        assert NOMINAL.range0[0] <= np.linalg.norm(states[0, 0:3]) <= NOMINAL.range0[1]
        assert NOMINAL.speed0[0] <= np.linalg.norm(states[0, 3:6]) <= NOMINAL.speed0[1]


def test_scenario_starts_wings_level_so_neutral_controls_hold_heading():
    """MS-1. atol 1e-12: wings level, every force lies in the velocity's vertical plane; only roundoff turns it."""
    states, _ = fly_scenario(level_scenario())
    heading = np.arctan2(states[:, 4], states[:, 3])
    np.testing.assert_allclose(heading, heading[0], rtol=0, atol=1e-12)


def test_scenario_measurement_noise_follows_its_named_regime():
    """MS-21."""
    states, z_nominal = fly_scenario(level_scenario(noise_regime="nominal", steps=400))
    _, z_noisy = fly_scenario(level_scenario(noise_regime="high_noise", steps=400))
    range_m = np.linalg.norm(states[:, 0:3], axis=1)
    assert np.std(z_noisy[:, 2] - range_m) > 2 * np.std(z_nominal[:, 2] - range_m)


def test_scenario_flies_its_recorded_envelope_so_a_9_g_command_pulls_that_envelope_max_g():
    """MS-1, z up. rtol 1e-9: acceleration is a velocity difference over dt, ~1e-12 m/s^2 roundoff at 200 m/s."""
    scenario = level_scenario(seed=8, steps=1)
    scenario["envelope"]["max_load_factor"] = 3.0
    scenario["controls"] = [[0.0, 9.0, 1.0]]

    states, _ = fly_scenario(scenario)

    unit_velocity = states[0, 3:6] / np.linalg.norm(states[0, 3:6])
    specific_force = states[0, 6:9] + np.array([0.0, 0.0, STANDARD_GRAVITY_M_S2])
    lift = specific_force - (specific_force @ unit_velocity) * unit_velocity
    np.testing.assert_allclose(np.linalg.norm(lift), 3.0 * STANDARD_GRAVITY_M_S2, rtol=1e-9)


def test_small_aerial_object_starts_inside_its_own_speed_envelope():
    """MS-19: the start speed is drawn inside the chosen envelope, not the regime's speed0."""
    envelope = PRESETS["small aerial object"]
    speeds = []
    for seed in range(20):
        scenario = level_scenario(seed=seed, steps=1) | {"envelope": asdict(envelope)}
        states, _ = fly_scenario(scenario)
        speeds.append(np.linalg.norm(states[0, 3:6]))
        assert envelope.min_speed_m_s <= np.linalg.norm(states[0, 3:6]) <= envelope.max_speed_m_s
    assert max(speeds) - min(speeds) > 1.0


def test_benchmark_envelope_start_speed_is_the_seed_draw_it_always_was():
    """MS-19. rtol 1e-12: norm of the stored velocity components, roundoff only."""
    states, _ = fly_scenario(level_scenario(seed=3, steps=1))
    np.testing.assert_allclose(np.linalg.norm(states[0, 3:6]), START_SPEED_SEED_3_M_S, rtol=1e-12, atol=0)


def test_envelope_disjoint_from_the_regime_speed_range_starts_inside_the_envelope():
    """MS-19: no overlap with speed0 (20-250 m/s), so the whole envelope is the draw range."""
    envelope = asdict(PRESETS["jet"]) | {"min_speed_m_s": 300.0, "max_speed_m_s": 400.0}
    speeds = []
    for seed in range(20):
        states, _ = fly_scenario(level_scenario(seed=seed, steps=1) | {"envelope": envelope})
        speeds.append(np.linalg.norm(states[0, 3:6]))
        assert 300.0 <= speeds[-1] <= 400.0
    assert max(speeds) - min(speeds) > 1.0
