from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    num_classes: int
    max_len: int = 256
    emb_dim: int = 128
    hidden_dim: int = 128
    latent_dim: int = 128
    num_memory_atoms: int | None = None
    num_operators: int = 8
    tau: float = 0.45
    dropout: float = 0.15
    transformer_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff_dim: int = 512


class GRUTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, out_dim: int, dropout: float):
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
        return self.proj(masked_mean_pool(out, attention_mask))


class AlgebraicResonanceMemory(nn.Module):
    def __init__(
        self,
        dim: int,
        num_memory_atoms: int,
        num_operators: int = 8,
        tau: float = 0.45,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_memory_atoms = num_memory_atoms
        self.num_operators = num_operators
        self.tau = tau
        self.query_projector = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.memory = nn.Parameter(torch.randn(num_memory_atoms, dim) / math.sqrt(dim))
        eye = torch.eye(dim).unsqueeze(0).repeat(num_operators, 1, 1)
        self.operators = nn.Parameter(eye + 0.02 * torch.randn(num_operators, dim, dim))
        self.operator_bias = nn.Parameter(torch.zeros(num_operators, dim))
        self.metric_raw = nn.Parameter(torch.zeros(dim))
        self.operator_cost_raw = nn.Parameter(torch.zeros(num_operators))

    def positive_metric(self) -> torch.Tensor:
        return F.softplus(self.metric_raw) + 1.0e-6

    def operator_cost(self) -> torch.Tensor:
        return F.softplus(self.operator_cost_raw)

    def forward(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        q = self.query_projector(q)
        transformed = torch.einsum("bd,ked->bke", q, self.operators) + self.operator_bias.unsqueeze(0)
        diff = transformed.unsqueeze(2) - self.memory.unsqueeze(0).unsqueeze(0)
        dist = (diff.square() * self.positive_metric().view(1, 1, 1, self.dim)).sum(dim=-1)
        path_scores = -dist / self.tau - self.operator_cost().view(1, self.num_operators, 1)
        logits = torch.logsumexp(path_scores, dim=1)
        weights = F.softmax(logits, dim=-1)
        return {"logits": logits, "weights": weights, "retrieved": weights @ self.memory, "path_scores": path_scores}

    def operator_regularization(self) -> torch.Tensor:
        identity = torch.eye(self.dim, dtype=self.operators.dtype, device=self.operators.device)
        near_identity = (self.operators - identity.unsqueeze(0)).square().mean()
        return near_identity + 0.1 * self.operator_bias.square().mean()

    def cycle_consistency_loss(self, order: int = 4) -> torch.Tensor:
        identity = torch.eye(self.dim, dtype=self.operators.dtype, device=self.operators.device)
        return F.mse_loss(torch.matrix_power(self.operators[0], order), identity)


class DotProductMemory(nn.Module):
    def __init__(self, dim: int, num_memory_atoms: int, dropout: float):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_memory_atoms, dim) / math.sqrt(dim))
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
        q = self.query_projector(q)
        logits = q @ self.memory.t() / self.scale
        weights = F.softmax(logits, dim=-1)
        return {"logits": logits, "weights": weights, "retrieved": weights @ self.memory}


class PrototypeMemory(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, dim) / math.sqrt(dim))
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.query_projector = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        q = self.query_projector(q)
        dist = (q.unsqueeze(1) - self.prototypes.unsqueeze(0)).square().sum(dim=-1)
        logits = -dist * self.log_scale.exp().clamp_max(100.0)
        weights = F.softmax(logits, dim=-1)
        return {"logits": logits, "weights": weights, "retrieved": weights @ self.prototypes}


class RBFMemory(nn.Module):
    def __init__(self, dim: int, num_memory_atoms: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_memory_atoms, dim) / math.sqrt(dim))
        self.log_gamma = nn.Parameter(torch.zeros(num_memory_atoms))
        self.query_projector = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        q = self.query_projector(q)
        dist = (q.unsqueeze(1) - self.centers.unsqueeze(0)).square().sum(dim=-1)
        logits = -F.softplus(self.log_gamma).unsqueeze(0) * dist
        weights = F.softmax(logits, dim=-1)
        return {"logits": logits, "weights": weights, "retrieved": weights @ self.centers}


class HopfieldMemory(nn.Module):
    def __init__(self, dim: int, num_memory_atoms: int, beta: float = 1.0):
        super().__init__()
        self.patterns = nn.Parameter(torch.randn(num_memory_atoms, dim) / math.sqrt(dim))
        self.beta_raw = nn.Parameter(torch.tensor(beta))
        self.out = nn.Linear(dim, num_memory_atoms)
        self.query_projector = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        q = self.query_projector(q)
        scores = self.beta_raw.exp().clamp_max(100.0) * (q @ self.patterns.t()) / math.sqrt(q.shape[-1])
        weights = F.softmax(scores, dim=-1)
        retrieved = weights @ self.patterns
        logits = self.out(retrieved)
        return {"logits": logits, "weights": weights, "retrieved": retrieved}


class MemoryClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, memory: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.memory = memory

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        q = self.encoder(input_ids, attention_mask)
        return self.memory(q)


class TransformerClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.transformer_dim % config.transformer_heads != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads")
        self.embedding = nn.Embedding(config.vocab_size, config.transformer_dim, padding_idx=0)
        self.position = nn.Embedding(config.max_len, config.transformer_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.transformer_layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(nn.LayerNorm(config.transformer_dim), nn.Linear(config.transformer_dim, config.num_classes))
        self.scale = math.sqrt(config.transformer_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        x = self.embedding(input_ids) * self.scale + self.position(positions)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        pooled = masked_mean_pool(x, attention_mask)
        return {"logits": self.classifier(pooled)}


def build_model(name: str, config: ModelConfig) -> nn.Module:
    if name == "transformer":
        return TransformerClassifier(config)

    atoms = config.num_memory_atoms or config.num_classes
    encoder = GRUTextEncoder(config.vocab_size, config.emb_dim, config.hidden_dim, config.latent_dim, config.dropout)
    if name == "arm":
        memory = AlgebraicResonanceMemory(config.latent_dim, atoms, config.num_operators, config.tau, config.dropout)
    elif name == "dot_memory":
        memory = DotProductMemory(config.latent_dim, atoms, config.dropout)
    elif name == "prototype":
        memory = PrototypeMemory(config.latent_dim, config.num_classes)
    elif name == "rbf":
        memory = RBFMemory(config.latent_dim, atoms)
    elif name == "hopfield":
        memory = HopfieldMemory(config.latent_dim, atoms)
    else:
        raise ValueError(f"unknown model: {name}")
    return MemoryClassifier(encoder, memory)

