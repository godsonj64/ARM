from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GenerativeARMConfig:
    vocab_size: int = 259
    max_len: int = 256
    emb_dim: int = 128
    hidden_dim: int = 192
    arm_dim: int = 192
    num_layers: int = 1
    num_memory_atoms: int = 256
    num_operators: int = 8
    tau: float = 0.6
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GenerativeARMConfig":
        return cls(**values)


class CausalARMBlock(nn.Module):
    def __init__(self, dim: int, num_memory_atoms: int, num_operators: int, tau: float, dropout: float):
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.dim = dim
        self.num_memory_atoms = num_memory_atoms
        self.num_operators = num_operators
        self.tau = tau

        self.query = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.memory = nn.Parameter(torch.randn(num_memory_atoms, dim) / math.sqrt(dim))
        eye = torch.eye(dim).unsqueeze(0).repeat(num_operators, 1, 1)
        self.operators = nn.Parameter(eye + 0.02 * torch.randn(num_operators, dim, dim))
        self.operator_bias = nn.Parameter(torch.zeros(num_operators, dim))
        self.operator_cost_raw = nn.Parameter(torch.zeros(num_operators))
        self.metric_raw = nn.Parameter(torch.zeros(dim))
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def positive_metric(self) -> torch.Tensor:
        return F.softplus(self.metric_raw) + 1.0e-6

    def operator_cost(self) -> torch.Tensor:
        return F.softplus(self.operator_cost_raw)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        q = self.query(x)
        transformed = torch.einsum("btd,ked->btke", q, self.operators) + self.operator_bias.view(1, 1, self.num_operators, self.dim)
        diff = transformed.unsqueeze(3) - self.memory.view(1, 1, 1, self.num_memory_atoms, self.dim)
        dist = (diff.square() * self.positive_metric().view(1, 1, 1, 1, self.dim)).sum(dim=-1)
        path_scores = -dist / self.tau - self.operator_cost().view(1, 1, self.num_operators, 1)
        atom_logits = torch.logsumexp(path_scores, dim=2)
        weights = F.softmax(atom_logits, dim=-1)
        retrieved = weights @ self.memory
        x = x + self.out(retrieved)
        x = x + self.ff(x)
        return {"hidden": self.norm(x), "weights": weights, "path_scores": path_scores}

    def regularization_loss(self) -> torch.Tensor:
        identity = torch.eye(self.dim, dtype=self.operators.dtype, device=self.operators.device)
        return (self.operators - identity.unsqueeze(0)).square().mean() + 0.1 * self.operator_bias.square().mean()


class GenerativeARMLanguageModel(nn.Module):
    def __init__(self, config: GenerativeARMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_len, config.emb_dim)
        self.input_dropout = nn.Dropout(config.dropout)
        self.rnn = nn.GRU(
            input_size=config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.to_arm = nn.Linear(config.hidden_dim, config.arm_dim)
        self.arm = CausalARMBlock(
            dim=config.arm_dim,
            num_memory_atoms=config.num_memory_atoms,
            num_operators=config.num_operators,
            tau=config.tau,
            dropout=config.dropout,
        )
        self.lm_head = nn.Sequential(nn.LayerNorm(config.arm_dim), nn.Linear(config.arm_dim, config.vocab_size))

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_len {self.config.max_len}")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.input_dropout(x)
        states, _ = self.rnn(x)
        arm_out = self.arm(self.to_arm(states))
        logits = self.lm_head(arm_out["hidden"])
        out = {"logits": logits, "weights": arm_out["weights"], "path_scores": arm_out["path_scores"]}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=0)
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 120,
        temperature: float = 0.9,
        top_k: int = 40,
        eos_id: int | None = 2,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_len :]
            logits = self(context)["logits"][:, -1, :]
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
                    filtered = torch.full_like(logits, -torch.inf)
                    logits = filtered.scatter(-1, indices, values)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
        return generated
