"""KalmanNet architecture #2 (Revach et al., arXiv:2107.10043) -- three GRUs
(Q, Sigma, S) plus FC pre/post layers, ported from the reference
KNet/KalmanNet_nn.py (see issue #14 sec. 1). Kept as close to the reference's
data flow as possible, including the non-standard "backward flow" where FC3
-> FC4 overwrites the Sigma-GRU's hidden state each step.

Tensors use the reference's seq-first GRU convention (seq_len=1, batch,
feature) rather than batch_first -- not a style choice: the backward-flow
output (shape [1, batch, hidden]) is fed straight back in as the Sigma-GRU's
next hidden state (shape [num_layers=1, batch, hidden]), and only lines up
because both leading dims are 1. Switching to batch_first would silently
transpose batch and hidden on that feedback path for batch_size > 1.
"""

import torch
import torch.nn as nn

from filters.knet.numpy_forward import DROPOUT_LAYERS
from filters.state import MEAS_DIM, STATE_DIM

CONCRETE_TEMPERATURE = 0.1
CONCRETE_EPS = 1e-7


def bernoulli_entropy(p):
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


class ConcreteDropout(nn.Module):
    """Relaxed Bernoulli dropout with a learned p: Gal, Hron and Kendall 2017, temperature and eps from their code."""

    def __init__(self, p_init):
        super().__init__()
        self.p_logit = nn.Parameter(torch.logit(torch.tensor([p_init], dtype=torch.float64)))

    @property
    def p(self):
        return torch.sigmoid(self.p_logit)

    def forward(self, x):
        p, u = self.p, torch.rand_like(x)
        log_odds = torch.log(p + CONCRETE_EPS) - torch.log(1.0 - p + CONCRETE_EPS)
        noise = torch.log(u + CONCRETE_EPS) - torch.log(1.0 - u + CONCRETE_EPS)
        dropped = torch.sigmoid((log_odds + noise) / CONCRETE_TEMPERATURE)
        return x * (1.0 - dropped) / (1.0 - p)


class KalmanGainGRU(nn.Module):
    def __init__(self, m=STATE_DIM, n=MEAS_DIM, in_mult=5, out_mult=40, dropout_p=0.0, learned_p_init=None):
        super().__init__()
        self.m, self.n = m, n
        if learned_p_init is None:
            self.dropout = nn.ModuleDict(dict.fromkeys(DROPOUT_LAYERS, nn.Dropout(dropout_p)))
        else:
            self.dropout = nn.ModuleDict({layer: ConcreteDropout(p) for layer, p in zip(DROPOUT_LAYERS, learned_p_init, strict=True)})

        self.d_hidden_Q = m * m
        self.GRU_Q = nn.GRU(m * in_mult, self.d_hidden_Q)

        self.d_hidden_Sigma = m * m
        self.GRU_Sigma = nn.GRU(self.d_hidden_Q + m * in_mult, self.d_hidden_Sigma)

        self.d_hidden_S = n * n
        self.GRU_S = nn.GRU(n * n + 2 * n * in_mult, self.d_hidden_S)

        self.FC1 = nn.Sequential(nn.Linear(self.d_hidden_Sigma, n * n), nn.ReLU())

        d_in_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        d_out_FC2 = n * m
        self.FC2 = nn.Sequential(
            nn.Linear(d_in_FC2, d_in_FC2 * out_mult),
            nn.ReLU(),
            nn.Linear(d_in_FC2 * out_mult, d_out_FC2),
        )
        self.FC3 = nn.Sequential(nn.Linear(self.d_hidden_S + d_out_FC2, m * m), nn.ReLU())
        self.FC4 = nn.Sequential(nn.Linear(self.d_hidden_Sigma + m * m, self.d_hidden_Sigma), nn.ReLU())
        self.FC5 = nn.Sequential(nn.Linear(m, m * in_mult), nn.ReLU())
        self.FC6 = nn.Sequential(nn.Linear(m, m * in_mult), nn.ReLU())
        self.FC7 = nn.Sequential(nn.Linear(2 * n, 2 * n * in_mult), nn.ReLU())

        # A freshly initialized gain matrix has no restoring-force guarantee
        # the way an analytic Kalman gain does -- closed-loop (I-KH)F can
        # easily have spectral radius > 1, so an untrained network diverges
        # from step one. Start the gain near zero (x_upd ~= x_prior, pure CV
        # prediction, which is itself stable) and let training grow it.
        final_linear = self.FC2[-1]
        nn.init.uniform_(final_linear.weight, -1e-3, 1e-3)
        nn.init.zeros_(final_linear.bias)

    def dropped_unit_weights(self):
        """Per dropout layer, the weight columns its dropped units multiply: eq 24's θ_λ, regularised as [47] does."""
        widths = {layer: getattr(self, layer)[0].out_features for layer in DROPOUT_LAYERS}
        consumers = {
            "FC5": (self.GRU_Q.weight_ih_l0, 0),
            "FC6": (self.GRU_Sigma.weight_ih_l0, self.d_hidden_Q),
            "FC1": (self.GRU_S.weight_ih_l0, 0),
            "FC7": (self.GRU_S.weight_ih_l0, widths["FC1"]),
            "FC2": (self.FC2[2].weight, 0),
            "FC3": (self.FC4[0].weight, self.d_hidden_Sigma),
            "FC4": (self.GRU_Sigma.weight_hh_l0, 0),
        }
        return {layer: weight[:, offset : offset + widths[layer]] for layer, (weight, offset) in consumers.items()}

    def kl_regulariser(self, c1, c2):
        """Eq 24 summed over the dropout layers: c1 ‖θ_λ‖² / (1 − p) − c2 H(p) n(λ), n(λ) the dropped units."""
        total = torch.zeros((), dtype=torch.float64)
        for layer, theta in self.dropped_unit_weights().items():
            p = self.dropout[layer].p.squeeze()
            total = total + c1 * theta.pow(2).sum() / (1.0 - p) - c2 * bernoulli_entropy(p) * theta.shape[1]
        return total

    def init_hidden(self, batch_size, prior_Q, prior_Sigma, prior_S, device):
        def rep(prior, hidden_dim):
            return prior.flatten().reshape(1, 1, hidden_dim).repeat(1, batch_size, 1).to(device)
        return (
            rep(prior_Q, self.d_hidden_Q),
            rep(prior_Sigma, self.d_hidden_Sigma),
            rep(prior_S, self.d_hidden_S),
        )

    def forward(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff, hidden):
        """All four *_diff args: [batch, dim]. hidden: (h_Q, h_Sigma, h_S),
        each [1, batch, hidden_dim]. Returns (gain [batch, m, n], new hidden)."""
        h_Q, h_Sigma, h_S = hidden
        obs_diff = obs_diff.unsqueeze(0)
        obs_innov_diff = obs_innov_diff.unsqueeze(0)
        fw_evol_diff = fw_evol_diff.unsqueeze(0)
        fw_update_diff = fw_update_diff.unsqueeze(0)

        out_FC5 = self.dropout["FC5"](self.FC5(fw_update_diff))
        out_Q, h_Q = self.GRU_Q(out_FC5, h_Q)

        out_FC6 = self.dropout["FC6"](self.FC6(fw_evol_diff))
        in_Sigma = torch.cat((out_Q, out_FC6), dim=-1)
        out_Sigma, h_Sigma = self.GRU_Sigma(in_Sigma, h_Sigma)

        out_FC1 = self.dropout["FC1"](self.FC1(out_Sigma))

        in_FC7 = torch.cat((obs_diff, obs_innov_diff), dim=-1)
        out_FC7 = self.dropout["FC7"](self.FC7(in_FC7))

        in_S = torch.cat((out_FC1, out_FC7), dim=-1)
        out_S, h_S = self.GRU_S(in_S, h_S)

        in_FC2 = torch.cat((out_Sigma, out_S), dim=-1)
        out_FC2 = self.FC2[2](self.dropout["FC2"](self.FC2[1](self.FC2[0](in_FC2))))

        in_FC3 = torch.cat((out_S, out_FC2), dim=-1)
        out_FC3 = self.dropout["FC3"](self.FC3(in_FC3))

        in_FC4 = torch.cat((out_Sigma, out_FC3), dim=-1)
        out_FC4 = self.dropout["FC4"](self.FC4(in_FC4))
        h_Sigma = out_FC4  # backward flow: overwrite Sigma-GRU's hidden state

        gain_flat = out_FC2.squeeze(0)  # [batch, n*m]
        gain = gain_flat.view(-1, self.m, self.n)
        return gain, (h_Q, h_Sigma, h_S)
