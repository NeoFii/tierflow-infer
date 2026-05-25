"""CG-TabM regressor: Cross-Gated Tabular Model with BatchEnsemble."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_token_shape(input_dim: int, num_tokens: int = 30, token_dim: int = 128):
    if input_dim != num_tokens * token_dim:
        raise ValueError(f"Dim mismatch: {input_dim} != {num_tokens}*{token_dim}")


def _layer_norm_k(x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
    bsz, k, dim = x.shape
    return ln(x.reshape(bsz * k, dim)).reshape(bsz, k, dim)


class HardConcreteGates(nn.Module):
    def __init__(
        self,
        k: int,
        n_heads: int,
        temperature: float = 2.0 / 3.0,
        a: float = -0.1,
        b: float = 1.1,
    ):
        super().__init__()
        self.k = k
        self.n_heads = n_heads
        self.temperature = temperature
        self.a, self.b = a, b
        self.log_alpha = nn.Parameter(torch.zeros(k, n_heads))

    def forward(self, batch_size: int, device: torch.device, training: bool) -> torch.Tensor:
        if training:
            u = torch.rand((batch_size, self.k, self.n_heads), device=device).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid(
                (self.log_alpha.unsqueeze(0) + torch.log(u) - torch.log(1 - u)) / self.temperature
            )
        else:
            s = torch.sigmoid(self.log_alpha.unsqueeze(0)).expand(batch_size, -1, -1)
        s_bar = s * (self.b - self.a) + self.a
        return torch.clamp(s_bar, 0.0, 1.0).unsqueeze(-1)

    def expected_l0(self):
        return torch.sigmoid(self.log_alpha).sum()


class LowRankCrossLayer(nn.Module):
    def __init__(self, dim: int, rank: int = 64):
        super().__init__()
        self.rank = min(rank, dim)
        self.V = nn.Parameter(torch.empty(dim, self.rank))
        self.U = nn.Parameter(torch.empty(self.rank, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.U)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x + x0 * (x @ self.V @ self.U) + self.bias


class BatchEnsembleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, k: int, bias: bool = True):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.r = nn.Parameter(torch.ones(k, in_features))
        self.s = nn.Parameter(torch.ones(k, out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.r.uniform_(0.9, 1.1)
            self.s.uniform_(0.9, 1.1)

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.k, -1)
        y = torch.einsum("bkd,od->bko", x * self.r.unsqueeze(0), self.weight)
        y = y * self.s.unsqueeze(0)
        if self.bias is not None:
            y = y + self.bias.view(1, 1, -1)
        return y


class BatchEnsembleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, k: int, dropout: float = 0.15):
        super().__init__()
        self.fc1 = BatchEnsembleLinear(in_dim, hidden_dim, k=k)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = BatchEnsembleLinear(hidden_dim, hidden_dim, k=k)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = BatchEnsembleLinear(hidden_dim, out_dim, k=k)
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = _layer_norm_k(x, self.ln1)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = _layer_norm_k(x, self.ln2)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        return x


class CGTabMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        token_dim: int = 128,
        k: int = 8,
        num_tokens: int = 30,
        cross_layers: int = 3,
        cross_rank: int = 64,
        deep_hidden: int = 512,
        dropout: float = 0.15,
        l0_lambda: float = 1e-4,
        gate_temperature: float = 2.0 / 3.0,
    ):
        super().__init__()
        _require_token_shape(input_dim, num_tokens, token_dim)
        self.k = k
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.input_dim = input_dim
        self.gates = HardConcreteGates(k=k, n_heads=num_tokens, temperature=gate_temperature)
        self.l0_lambda = l0_lambda
        self.cross = nn.ModuleList([LowRankCrossLayer(input_dim, rank=cross_rank) for _ in range(cross_layers)])
        self.deep = BatchEnsembleMLP(input_dim, deep_hidden, deep_hidden, k=k, dropout=dropout)
        self.fuse_ln = nn.LayerNorm(input_dim + deep_hidden)
        self.head = BatchEnsembleLinear(input_dim + deep_hidden, 1, k=k)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor):
        bsz, dev = x.size(0), x.device
        x_seq = x.view(bsz, self.num_tokens, self.token_dim)
        gate = self.gates(bsz, dev, training=self.training)
        xk = (x_seq.unsqueeze(1) * gate).reshape(bsz, self.k, self.input_dim)
        x0 = xk.reshape(bsz * self.k, self.input_dim)
        xi = x0
        for layer in self.cross:
            xi = layer(x0, xi)
        x_cross = xi.reshape(bsz, self.k, self.input_dim)
        x_deep = self.deep(xk)
        h = torch.cat([x_cross, x_deep], dim=-1)
        h = _layer_norm_k(h, self.fuse_ln)
        out = self.head(h)
        if self.training and self.l0_lambda > 0:
            self.aux_loss = self.l0_lambda * self.gates.expected_l0()
        else:
            self.aux_loss = torch.tensor(0.0, device=dev)
        return out
