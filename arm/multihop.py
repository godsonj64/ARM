"""Multi-hop Algebraic Resonance Memory layers.

This module keeps experimental multi-hop ARM separate from the original
single-hop implementation in ``arm.models``.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHopAlgebraicResonanceMemory(nn.Module):
    """ARM retrieval with repeated operator application before memory scoring.

    Single-hop ARM scores memories after one transformed query step:

        q -> A_k q + b_k -> memory

    Multi-hop ARM keeps a beam of transformed query states and applies the same
    operator family repeatedly:

        q -> A_i q -> A_j A_i q -> ...

    It aggregates memory scores across depths 1..H and across retained path
    states with ``logsumexp``. ``beam_width`` controls the number of path states
    carried to the next hop.
    """

    def __init__(
        self,
        dim: int,
        num_memories: int,
        num_operators: int = 8,
        max_hops: int = 4,
        beam_width: int = 16,
        tau: float = 0.45,
        score_intermediate: bool = True,
        dropout: float = 0.05,
    ):
        super().__init__()
        if dim <= 0 or num_memories <= 0 or num_operators <= 0 or max_hops <= 0 or beam_width <= 0:
            raise ValueError("dim, num_memories, num_operators, max_hops and beam_width must be positive")
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.dim = dim
        self.num_memories = num_memories
        self.num_operators = num_operators
        self.max_hops = max_hops
        self.beam_width = beam_width
        self.tau = tau
        self.score_intermediate = score_intermediate

        self.memory = nn.Parameter(torch.randn(num_memories, dim) / math.sqrt(dim))
        eye = torch.eye(dim).unsqueeze(0).repeat(num_operators, 1, 1)
        self.operators = nn.Parameter(eye + 0.02 * torch.randn(num_operators, dim, dim))
        self.operator_bias = nn.Parameter(torch.zeros(num_operators, dim))
        self.metric_raw = nn.Parameter(torch.zeros(dim))
        self.operator_cost_raw = nn.Parameter(torch.zeros(num_operators))
        self.hop_cost_raw = nn.Parameter(torch.zeros(max_hops))
        self.query_projector = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def positive_metric(self) -> torch.Tensor:
        return F.softplus(self.metric_raw) + 1.0e-5

    def operator_cost(self) -> torch.Tensor:
        return F.softplus(self.operator_cost_raw)

    def hop_cost(self) -> torch.Tensor:
        return F.softplus(self.hop_cost_raw)

    def _expand_paths(self, states: torch.Tensor, path_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, paths, _ = states.shape
        flat = states.reshape(batch * paths, self.dim)
        expanded = torch.einsum("bd,ked->bke", flat, self.operators)
        expanded = expanded + self.operator_bias.unsqueeze(0)
        expanded = expanded.reshape(batch, paths * self.num_operators, self.dim)
        next_scores = path_scores.unsqueeze(-1) - self.operator_cost().view(1, 1, -1)
        next_scores = next_scores.reshape(batch, paths * self.num_operators)
        keep = min(self.beam_width, next_scores.shape[1])
        top_scores, top_idx = torch.topk(next_scores, keep, dim=1)
        top_states = expanded.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, self.dim))
        return top_states, top_scores

    def _memory_scores(self, states: torch.Tensor, path_scores: torch.Tensor, hop_idx: int) -> torch.Tensor:
        diff = states.unsqueeze(2) - self.memory.view(1, 1, self.num_memories, self.dim)
        dist = (diff.square() * self.positive_metric().view(1, 1, 1, -1)).sum(dim=-1)
        scores = -dist / self.tau + path_scores.unsqueeze(-1) - self.hop_cost()[hop_idx]
        return scores

    def resonance_logits(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.ndim != 2 or q.shape[-1] != self.dim:
            raise ValueError(f"expected q with shape [batch, {self.dim}], got {tuple(q.shape)}")
        states = self.query_projector(q).unsqueeze(1)
        path_scores = q.new_zeros(q.shape[0], 1)
        scored_hops = []
        path_counts = []
        for hop in range(self.max_hops):
            states, path_scores = self._expand_paths(states, path_scores)
            if self.score_intermediate or hop == self.max_hops - 1:
                scored_hops.append(self._memory_scores(states, path_scores, hop))
                path_counts.append(states.shape[1])
        flat_scores = torch.cat([scores.reshape(q.shape[0], -1, self.num_memories) for scores in scored_hops], dim=1)
        logits = torch.logsumexp(flat_scores, dim=1)
        return logits, flat_scores

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
