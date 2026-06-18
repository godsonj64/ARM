"""Synthetic cyclic fan-state dataset for ARM and attention benchmarks."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CyclicFanMemoryDataset(Dataset):
    """Multiple cue types retrieve the same hidden cyclic state.

    Hidden state lives in Z_n. Each sample has two different cue vectors that
    refer to the same state, allowing tests of retrieval equivalence.
    """

    def __init__(
        self,
        num_samples: int,
        num_states: int = 4,
        cue_types: int = 5,
        dim: int = 64,
        noise_std: float = 0.08,
        seed: int = 123,
        shared_basis: Dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.num_samples = num_samples
        self.num_states = num_states
        self.cue_types = cue_types
        self.dim = dim
        self.noise_std = noise_std
        g = torch.Generator().manual_seed(seed)

        if shared_basis is None:
            state_base = F.normalize(torch.randn(num_states, dim, generator=g), dim=-1)
            cue_maps = torch.randn(cue_types, dim, dim, generator=g) / math.sqrt(dim)
            cue_maps = 0.70 * torch.eye(dim).unsqueeze(0) + 0.30 * cue_maps
            pull_embed = F.normalize(torch.randn(num_states, dim, generator=g), dim=-1)
            shared_basis = {"state_base": state_base, "cue_maps": cue_maps, "pull_embed": pull_embed}
        self.shared_basis = shared_basis
        self.state_base = shared_basis["state_base"]
        self.cue_maps = shared_basis["cue_maps"]
        self.pull_embed = shared_basis["pull_embed"]

        self.states = torch.randint(0, num_states, (num_samples,), generator=g)
        self.cue_a = torch.randint(0, cue_types, (num_samples,), generator=g)
        self.cue_b = torch.randint(0, cue_types, (num_samples,), generator=g)
        self.pulls_seen = torch.randint(0, 12, (num_samples,), generator=g)
        self.q_a = self._make_queries(self.states, self.cue_a, self.pulls_seen, g)
        self.q_b = self._make_queries(self.states, self.cue_b, self.pulls_seen + 1, g)

    def _make_queries(self, states: torch.Tensor, cue_ids: torch.Tensor, pulls_seen: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        base = self.state_base[states]
        phase = pulls_seen % self.num_states
        x = base + 0.35 * self.pull_embed[phase]
        q = torch.bmm(self.cue_maps[cue_ids], x.unsqueeze(-1)).squeeze(-1)
        q = torch.tanh(q)
        q = q + self.noise_std * torch.randn(q.shape, generator=g)
        return q.float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "q_a": self.q_a[idx],
            "q_b": self.q_b[idx],
            "state": self.states[idx].long(),
            "cue_a": self.cue_a[idx].long(),
            "cue_b": self.cue_b[idx].long(),
        }
