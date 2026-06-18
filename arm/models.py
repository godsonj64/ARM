"""Core models for Algebraic Resonance Memory experiments.

This module contains the proposed ARM retrieval layer, a direct attention-memory
baseline, and small encoders used by the benchmark scripts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ARMHyperParams:
    dim: int = 128
    num_memories: int = 4
    num_operators: int = 8
    tau: float = 0.45
    dropout: float = 0.1


class AlgebraicResonanceMemory(nn.Module):
    """Algebraic Resonance Memory.

    For query q, memory atom m_i, and learned operators A_k, ARM scores

        rho(q, m_i) = logsumexp_k(-||A_k q + b_k - m_i||^2_D / tau - c_k).

    The retrieved vector is the weighted sum of memory atoms under softmax(rho).
    """

    def __init__(self, dim: int, num_memories: int, num_operators: int = 8, tau: float = 0.45):
        super().__init__()
        if dim <= 0 or num_memories <= 0 or num_operators <= 0:
            raise ValueError("dim, num_memories and num_operators must be positive")
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.dim = dim
        self.num_memories = num_memories
        self.num_operators = num_operators
        self.tau = tau

        self.memory = nn.Parameter(torch.randn(num_memories, dim) / math.sqrt(dim))
        eye = torch.eye(dim).unsqueeze(0).repeat(num_operators, 1, 1)
        self.operators = nn.Parameter(eye + 0.02 * torch.randn(num_operators, dim, dim))
        self.operator_bias = nn.Parameter(torch.zeros(num_operators, dim))
        self.metric_raw = nn.Parameter(torch.zeros(dim))
        self.operator_cost_raw = nn.Parameter(torch.zeros(num_operators))
        self.query_projector = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def positive_metric(self) -> torch.Tensor:
        return F.softplus(self.metric_raw) + 1.0e-5

    def operator_cost(self) -> torch.Tensor:
        return F.softplus(self.operator_cost_raw)

    def transformed_queries(self, q: torch.Tensor) -> torch.Tensor:
        if q.ndim != 2 or q.shape[-1] != self.dim:
            raise ValueError(f"expected q with shape [batch, {self.dim}], got {tuple(q.shape)}")
        q = self.query_projector(q)
        transformed = torch.einsum("bd,ked->bke", q, self.operators)
        return transformed + self.operator_bias.unsqueeze(0)

    def resonance_logits(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        transformed = self.transformed_queries(q)
        diff = transformed.unsqueeze(2) - self.memory.unsqueeze(0).unsqueeze(0)
        dist = (diff.square() * self.positive_metric().view(1, 1, 1, -1)).sum(dim=-1)
        path_scores = -dist / self.tau - self.operator_cost().view(1, -1, 1)
        logits = torch.logsumexp(path_scores, dim=1)
        return logits, path_scores

    def forward(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits, path_scores = self.resonance_logits(q)
        weights = F.softmax(logits, dim=-1)
        retrieved = weights @ self.memory
        return {"logits": logits, "weights": weights, "retrieved": retrieved, "path_scores": path_scores}

    def cycle_consistency_loss(self, order: int = 4) -> torch.Tensor:
        if order <= 0:
            raise ValueError("order must be positive")
        A = self.operators[0]
        A_power = torch.matrix_power(A, order)
        I = torch.eye(self.dim, dtype=A.dtype, device=A.device)
        return F.mse_loss(A_power, I)

    def operator_regularization(self) -> torch.Tensor:
        I = torch.eye(self.dim, dtype=self.operators.dtype, device=self.operators.device)
        near_identity = (self.operators - I.unsqueeze(0)).square().mean()
        bias_penalty = self.operator_bias.square().mean()
        return near_identity + 0.1 * bias_penalty


class AttentionMemory(nn.Module):
    """Direct attention baseline over learned memory atoms.

    This baseline retrieves memory by direct query-memory dot product. It uses
    a learned projection but has no transformed-query operator paths.
    """

    def __init__(self, dim: int, num_memories: int, dropout: float = 0.1):
        super().__init__()
        if dim <= 0 or num_memories <= 0:
            raise ValueError("dim and num_memories must be positive")
        self.dim = dim
        self.num_memories = num_memories
        self.memory = nn.Parameter(torch.randn(num_memories, dim) / math.sqrt(dim))
        self.query_projector = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.scale = math.sqrt(dim)

    def forward(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        if q.ndim != 2 or q.shape[-1] != self.dim:
            raise ValueError(f"expected q with shape [batch, {self.dim}], got {tuple(q.shape)}")
        q = self.query_projector(q)
        logits = q @ self.memory.t() / self.scale
        weights = F.softmax(logits, dim=-1)
        retrieved = weights @ self.memory
        return {"logits": logits, "weights": weights, "retrieved": retrieved}


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or max(out_dim, in_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        out, _ = self.gru(x)
        mask = attention_mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


class MemoryClassifier(nn.Module):
    """Encoder plus memory layer classifier."""

    def __init__(self, encoder: nn.Module, memory_layer: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.memory_layer = memory_layer

    def forward(self, *args: torch.Tensor) -> Dict[str, torch.Tensor]:
        q = self.encoder(*args)
        return self.memory_layer(q)
