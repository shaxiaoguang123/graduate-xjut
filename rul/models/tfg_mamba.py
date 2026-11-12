"""TFG-Mamba (Wei et al., Advanced Engineering Informatics 72 (2026) 104443).

Structure (paper Figs. 1-3, Eqs. 1-20):

    x (B, L, 26) = [13 time-domain | 13 frequency-domain] features per snapshot
    time path : TCN (residual blocks of dilated causal conv + WeightNorm + ReLU + Dropout) -> Linear
    freq path : Conv1D -> squeeze-and-excitation (SEFA) -> Linear
    GRSM      : LayerNorm(a + GLU(W1 ELU(W2 a + W3 c + b2) + b1))
    backbone  : Linear projection -> Dropout -> N x Mamba block
    head      : fully connected layer -> output layer -> RUL at the last time step

The Mamba block is a pure-PyTorch implementation of the selective SSM with the
zero-order-hold discretisation of Eqs. (18)-(19). Details the paper leaves open
(SE reduction ratio, Conv1D kernel, Mamba state size, head activation) are
listed in papers/tfg-mamba/notes.md.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


@dataclass(frozen=True)
class TFGMambaConfig:
    n_time_features: int = 13
    n_freq_features: int = 13
    tcn_channels: tuple[int, ...] = (32, 64)
    kernel_size: int = 5
    hidden: int = 32
    n_layers: int = 6
    dropout: float = 0.2
    se_reduction: int = 16
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    scan: str = "logdepth"  # "logdepth" or "sequential"; numerically equivalent

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tcn_channels"] = list(self.tcn_channels)
        return d


# --------------------------------------------------------------------------- TCN


class TemporalBlock(nn.Module):
    """Residual block of Bai et al. (2018): two dilated causal convolutions."""

    def __init__(self, n_in: int, n_out: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(n_in, n_out, kernel_size, padding=padding, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(n_out, n_out, kernel_size, padding=padding, dilation=dilation))
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        nn.init.normal_(self.conv1.parametrizations.weight.original1, 0.0, 0.01)
        nn.init.normal_(self.conv2.parametrizations.weight.original1, 0.0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0.0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        length = x.size(-1)
        out = self.dropout(F.relu(self.conv1(x)[..., :length]))  # chomp -> causal
        out = self.dropout(F.relu(self.conv2(out)[..., :length]))
        residual = x if self.downsample is None else self.downsample(x)
        return F.relu(out + residual)


class TCN(nn.Module):
    def __init__(self, n_in: int, channels: tuple[int, ...], kernel_size: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        for i, n_out in enumerate(channels):
            blocks.append(TemporalBlock(n_in if i == 0 else channels[i - 1], n_out, kernel_size, 2**i, dropout))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# -------------------------------------------------------------------------- SEFA


class SEFA(nn.Module):
    """Eq. (11): SE(Conv1D(X_freq^T)) with Eqs. (4)-(5) squeeze, excitation, scale."""

    def __init__(self, n_in: int, channels: int, kernel_size: int, reduction: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(n_in, channels, kernel_size, padding=kernel_size - 1)
        squeezed = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, squeezed)
        self.fc2 = nn.Linear(squeezed, channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:  # (B, C_in, L)
        length = x.size(-1)
        u = self.conv(x)[..., :length]  # causal
        if mask is None:
            z = u.mean(dim=-1)
        else:  # global average over real (non-padded) positions only
            m = mask.to(u.dtype).unsqueeze(1)
            z = (u * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(z))))
        return u * s.unsqueeze(-1)


# -------------------------------------------------------------------------- GRSM


class GRSM(nn.Module):
    """Eqs. (13)-(15): LayerNorm(a + GLU(eta1)), eta1 = W1 eta2 + b1, eta2 = ELU(W2 a + W3 c + b2)."""

    def __init__(self, d: int, dropout: float) -> None:
        super().__init__()
        self.w2 = nn.Linear(d, d)  # W2 a + b2
        self.w3 = nn.Linear(d, d, bias=False)  # W3 c
        self.w1 = nn.Linear(d, 2 * d)  # W1 eta2 + b1 (2d so that GLU returns d)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        eta2 = F.elu(self.w2(a) + self.w3(c))
        eta1 = self.dropout(self.w1(eta2))
        return self.norm(a + F.glu(eta1, dim=-1))


# ------------------------------------------------------------------------- Mamba


def zoh_discretize(delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eqs. (18)-(19) for diagonal A.

    delta (b, l, d), A (d, n), B (b, l, n) -> A_bar, B_bar of shape (b, l, d, n), where
    A_bar = exp(delta*A) and B_bar = (delta*A)^-1 (exp(delta*A) - 1) delta*B = expm1(delta*A)/A * B.
    """
    dA = delta.unsqueeze(-1) * A
    A_bar = torch.exp(dA)
    B_bar = torch.expm1(dA) / A * B.unsqueeze(2)
    return A_bar, B_bar


def selective_scan_sequential(u, delta, A, B, C, D) -> torch.Tensor:
    """Reference recurrence x(k+1) = A_bar x(k) + B_bar u(k), y(k) = C x(k) + D u(k)."""
    A_bar, B_bar = zoh_discretize(delta, A, B)
    bu = B_bar * u.unsqueeze(-1)  # (b, l, d, n)
    h = torch.zeros_like(bu[:, 0])
    ys = []
    for t in range(u.size(1)):
        h = A_bar[:, t] * h + bu[:, t]
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
    return torch.stack(ys, dim=1) + u * D


def selective_scan_logdepth(u, delta, A, B, C, D) -> torch.Tensor:
    """Same recurrence as ``selective_scan_sequential``, evaluated by a Hillis-Steele scan.

    Each round composes (a_t, x_t) with (a_{t-k}, x_{t-k}): x_t += a_t x_{t-k}, a_t *= a_{t-k}.
    After ceil(log2 L) rounds x_t equals the hidden state h_t. All factors a lie in (0, 1],
    so this is numerically stable; it only changes the evaluation order.
    """
    A_bar, B_bar = zoh_discretize(delta, A, B)
    a = A_bar
    x = B_bar * u.unsqueeze(-1)  # (b, l, d, n)
    length = u.size(1)
    k = 1
    while k < length:
        x = x + a * F.pad(x[:, :-k], (0, 0, 0, 0, k, 0))
        a = a * F.pad(a[:, :-k], (0, 0, 0, 0, k, 0), value=1.0)
        k *= 2
    return torch.einsum("btdn,btn->btd", x, C) + u * D


class MambaMixer(nn.Module):
    """Mamba block (Gu & Dao, 2023; paper Fig. 2(b)): projection, conv + SiLU + SSM, SiLU gate, projection."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        scan: str = "logdepth",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        if scan not in ("logdepth", "sequential"):
            raise ValueError(f"unknown scan {scan!r}")
        self.scan = scan
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Standard Mamba initialisation of delta and A (S4D-real).
        std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, d_model)
        length = x.size(1)
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        xi = self.conv1d(xi.transpose(1, 2))[..., :length].transpose(1, 2)  # causal depthwise conv
        xi = F.silu(xi)
        dt, B, C = torch.split(self.x_proj(xi), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        scan = selective_scan_logdepth if self.scan == "logdepth" else selective_scan_sequential
        y = scan(xi, delta, A, B, C, self.D)
        return self.out_proj(y * F.silu(z))


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaLayer(nn.Module):
    """Pre-norm residual wrapper used by the reference Mamba backbone."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, scan: str) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaMixer(d_model, d_state, d_conv, expand, scan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


# ---------------------------------------------------------------------- TFG-Mamba


class TFGMamba(nn.Module):
    def __init__(self, config: TFGMambaConfig = TFGMambaConfig()) -> None:
        super().__init__()
        self.config = config
        ch_out = config.tcn_channels[-1]
        d = config.hidden
        self.tcn = TCN(config.n_time_features, config.tcn_channels, config.kernel_size, config.dropout)
        self.time_linear = nn.Linear(ch_out, d)
        self.sefa = SEFA(config.n_freq_features, ch_out, config.kernel_size, config.se_reduction)
        self.freq_linear = nn.Linear(ch_out, d)
        self.grsm = GRSM(d, config.dropout)
        self.projection = nn.Linear(d, d)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            MambaLayer(d, config.d_state, config.d_conv, config.expand, config.scan) for _ in range(config.n_layers)
        )
        self.norm_f = RMSNorm(d)
        self.fc = nn.Linear(d, d)
        self.out = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, L, n_time + n_freq) -> RUL at the last position, shape (B,)."""
        n_time = self.config.n_time_features
        x_time = x[..., :n_time].transpose(1, 2)  # Eq. (9): (B, D, L)
        x_freq = x[..., n_time:].transpose(1, 2)
        a = self.time_linear(self.tcn(x_time).transpose(1, 2))  # Eq. (10) + mapping layer
        c = self.freq_linear(self.sefa(x_freq, mask).transpose(1, 2))  # Eq. (11) + mapping layer
        h = self.grsm(a, c)
        h = self.dropout(self.projection(h))
        for layer in self.layers:
            h = layer(h)
        h = self.norm_f(h)[:, -1]
        h = F.relu(self.fc(h))
        return self.out(h).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
