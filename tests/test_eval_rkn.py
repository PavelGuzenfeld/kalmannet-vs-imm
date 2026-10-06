import numpy as np
import torch

from filters.state import STATE_DIM
from scripts import eval_rkn

TINY_DIAGONAL = 2.0**-30


class _IllConditionedFilter:
    def __init__(self, net, dt):
        factor = np.eye(STATE_DIM)
        factor[1, 0] = 1.0
        factor[1, 1] = TINY_DIAGONAL
        self.S = torch.tensor(factor).unsqueeze(0)

    def init_sequence(self, x0, P0):
        pass

    def step(self, z):
        x_upd = torch.zeros(1, STATE_DIM, dtype=torch.float64)
        x_upd[0, 0] = 1.0
        return x_upd, self.S @ self.S.transpose(1, 2)


def test_nees_is_whitened_through_s_when_p_rounds_to_singular(monkeypatch):
    """MS-36: P = S S^T rounds to singular here; e = [1, 0] is outside range(P) and whitens through S to [1, -2^30], so NEES is 2^60."""
    monkeypatch.setattr(eval_rkn, "RKNFilter", _IllConditionedFilter)
    states = np.zeros((1, 1, STATE_DIM))
    measurements = np.zeros((1, 1, 1, 3))

    result = eval_rkn.evaluate_dataset(None, 0.02, states, measurements)

    assert result["nees"][0, 0, 0] == 2.0**60
