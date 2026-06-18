from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared utilities
# ============================================================

def masked_mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool sequence embeddings using an attention mask.

    Args:
        x: Tensor of shape [batch, seq_len, dim].
        attention_mask: Tensor of shape [batch, seq_len], where 1 = valid token.

    Returns:
        Tensor of shape [batch, dim].
    """
    mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)
    summed = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard fixed sinusoidal positional encoding.
    """

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be positive")

        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)

        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, seq_len, dim].

        Returns:
            Tensor of shape [batch, seq_len, dim].
        """
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :].to(dtype=x.dtype, device=x.device)


# ============================================================
# Pure ARM model
# ============================================================

@dataclass
class ARMConfig:
    vocab_size: int
    num_classes: int
    dim: int = 128
    num_memory_atoms: Optional[int] = None
    num_operators: int = 8
    max_len: int = 256
    dropout: float = 0.1
    tau: float = 0.45
    use_positional_encoding: bool = True


class AlgebraicResonanceMemory(nn.Module):
    """
    Pure ARM retrieval layer.

    For query q, memory atom m_i, and learned operators A_k, ARM computes

        rho(q, m_i) = logsumexp_k(
            - || A_k q + b_k - m_i ||^2_D / tau - c_k
        )

    where D is a learned positive diagonal metric.

    The output logits are the resonance scores over memory atoms.
    If num_memory_atoms == num_classes, these logits can be used directly
    for classification.
    """

    def __init__(
        self,
        dim: int,
        num_memory_atoms: int,
        num_operators: int = 8,
        tau: float = 0.45,
        dropout: float = 0.0,
    ):
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_memory_atoms <= 0:
            raise ValueError("num_memory_atoms must be positive")
        if num_operators <= 0:
            raise ValueError("num_operators must be positive")
        if tau <= 0:
            raise ValueError("tau must be positive")

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

    def transformed_queries(self, q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: Tensor of shape [batch, dim].

        Returns:
            Tensor of shape [batch, num_operators, dim].
        """
        if q.ndim != 2 or q.shape[-1] != self.dim:
            raise ValueError(f"expected q shape [batch, {self.dim}], got {tuple(q.shape)}")

        q = self.query_projector(q)

        # q: [B, D]
        # operators: [K, D_out, D_in]
        # transformed: [B, K, D_out]
        transformed = torch.einsum("bd,ked->bke", q, self.operators)

        return transformed + self.operator_bias.unsqueeze(0)

    def resonance_logits(self, q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: Tensor of shape [batch, dim].

        Returns:
            Tensor of shape [batch, num_memory_atoms].
        """
        transformed = self.transformed_queries(q)

        # transformed: [B, K, D]
        # memory: [M, D]
        # diff: [B, K, M, D]
        diff = transformed.unsqueeze(2) - self.memory.unsqueeze(0).unsqueeze(0)

        metric = self.positive_metric().view(1, 1, 1, self.dim)
        dist = (diff.square() * metric).sum(dim=-1)

        path_scores = -dist / self.tau
        path_scores = path_scores - self.operator_cost().view(1, self.num_operators, 1)

        # Aggregate all transformed paths reaching the same memory atom.
        logits = torch.logsumexp(path_scores, dim=1)

        return logits

    def operator_regularization(self) -> torch.Tensor:
        """
        Keeps operators near stable transformations early in training.
        """
        identity = torch.eye(self.dim, dtype=self.operators.dtype, device=self.operators.device)
        near_identity = (self.operators - identity.unsqueeze(0)).square().mean()
        bias_penalty = self.operator_bias.square().mean()
        return near_identity + 0.1 * bias_penalty

    def cycle_consistency_loss(self, order: int = 4) -> torch.Tensor:
        """
        Optional regularizer.

        Encourages the first operator to behave approximately like an order-n cycle:

            A^order ~= I
        """
        if order <= 0:
            raise ValueError("order must be positive")

        A = self.operators[0]
        A_power = torch.matrix_power(A, order)
        identity = torch.eye(self.dim, dtype=A.dtype, device=A.device)

        return F.mse_loss(A_power, identity)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.resonance_logits(q)


class PureARMClassifier(nn.Module):
    """
    Pure text ARM classifier.

    Architecture:

        tokens -> embedding -> optional positional encoding -> masked mean pooling
        -> ARM resonance memory -> class logits

    The number of memory atoms is normally equal to num_classes.
    """

    def __init__(self, config: ARMConfig):
        super().__init__()

        if config.num_memory_atoms is None:
            config.num_memory_atoms = config.num_classes

        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.dim,
            padding_idx=0,
        )

        self.positional_encoding = (
            SinusoidalPositionalEncoding(config.dim, config.max_len)
            if config.use_positional_encoding
            else nn.Identity()
        )

        self.dropout = nn.Dropout(config.dropout)

        self.pre_arm = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim, config.dim),
            nn.LayerNorm(config.dim),
        )

        self.arm = AlgebraicResonanceMemory(
            dim=config.dim,
            num_memory_atoms=config.num_memory_atoms,
            num_operators=config.num_operators,
            tau=config.tau,
            dropout=config.dropout,
        )

        if config.num_memory_atoms == config.num_classes:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(config.num_memory_atoms, config.num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape [batch, seq_len].
            attention_mask: Tensor of shape [batch, seq_len].

        Returns:
            Class logits of shape [batch, num_classes].
        """
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        q = masked_mean_pool(x, attention_mask)
        q = self.pre_arm(q)

        resonance_logits = self.arm(q)
        logits = self.classifier(resonance_logits)

        return logits

    def regularization_loss(
        self,
        operator_weight: float = 1.0e-3,
        cycle_weight: float = 0.0,
        cycle_order: int = 4,
    ) -> torch.Tensor:
        loss = operator_weight * self.arm.operator_regularization()

        if cycle_weight > 0:
            loss = loss + cycle_weight * self.arm.cycle_consistency_loss(cycle_order)

        return loss


# ============================================================
# Standard Transformer model
# ============================================================

@dataclass
class TransformerConfig:
    vocab_size: int
    num_classes: int
    dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 512
    max_len: int = 256
    dropout: float = 0.1
    use_cls_token: bool = True


class StandardTransformerClassifier(nn.Module):
    """
    Standard Transformer encoder classifier.

    Architecture:

        tokens -> embedding + positional encoding
        -> TransformerEncoder
        -> CLS pooling or masked mean pooling
        -> linear classifier
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        if config.dim % config.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.dim,
            padding_idx=0,
        )

        self.positional_encoding = SinusoidalPositionalEncoding(
            dim=config.dim,
            max_len=config.max_len + int(config.use_cls_token),
        )

        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        else:
            self.cls_token = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.dim),
        )

        self.dropout = nn.Dropout(config.dropout)

        self.classifier = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape [batch, seq_len].
            attention_mask: Tensor of shape [batch, seq_len], where 1 = valid token.

        Returns:
            Class logits of shape [batch, num_classes].
        """
        x = self.embedding(input_ids)

        if self.cls_token is not None:
            batch_size = input_ids.shape[0]
            cls = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls, x], dim=1)

            cls_mask = torch.ones(
                batch_size,
                1,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        x = self.positional_encoding(x)
        x = self.dropout(x)

        # PyTorch Transformer uses True for positions that should be ignored.
        key_padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        if self.cls_token is not None:
            pooled = x[:, 0]
        else:
            pooled = masked_mean_pool(x, attention_mask)

        logits = self.classifier(pooled)

        return logits


# ============================================================
# Minimal training usage
# ============================================================

def classification_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids, attention_mask)
    return F.cross_entropy(logits, labels)


def example_usage() -> None:
    batch_size = 8
    seq_len = 64
    vocab_size = 5000
    num_classes = 18

    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, num_classes, (batch_size,))

    arm_model = PureARMClassifier(
        ARMConfig(
            vocab_size=vocab_size,
            num_classes=num_classes,
            dim=128,
            num_operators=8,
            tau=0.45,
            dropout=0.1,
        )
    )

    transformer_model = StandardTransformerClassifier(
        TransformerConfig(
            vocab_size=vocab_size,
            num_classes=num_classes,
            dim=128,
            num_heads=4,
            num_layers=4,
            ff_dim=512,
            dropout=0.1,
        )
    )

    arm_logits = arm_model(input_ids, attention_mask)
    transformer_logits = transformer_model(input_ids, attention_mask)

    arm_loss = F.cross_entropy(arm_logits, labels)
    arm_loss = arm_loss + arm_model.regularization_loss(
        operator_weight=1.0e-3,
        cycle_weight=3.0e-3,
        cycle_order=4,
    )

    transformer_loss = F.cross_entropy(transformer_logits, labels)

    print("ARM logits:", arm_logits.shape)
    print("Transformer logits:", transformer_logits.shape)
    print("ARM loss:", float(arm_loss))
    print("Transformer loss:", float(transformer_loss))


if __name__ == "__main__":
    example_usage()
