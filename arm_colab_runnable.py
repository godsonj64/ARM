# ============================================================
# Algebraic Resonance Memory (ARM)
# Fully runnable PyTorch / Google Colab script
# Author concept: Godson Johnson
# ============================================================

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

# Prevent CPU thread oversubscription in notebooks and hosted runtimes.
# GPU execution is unaffected.
torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
torch.set_num_interop_threads(1)

import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


@dataclass
class ARMConfig:
    num_states: int = 4
    cue_types: int = 5
    dim: int = 64
    num_operators: int = 8
    tau: float = 0.35
    noise_std: float = 0.08
    train_samples: int = 12000
    val_samples: int = 2500
    batch_size: int = 256
    epochs: int = 35
    lr: float = 2.0e-3
    weight_decay: float = 1.0e-4
    retrieval_weight: float = 1.0
    equivalence_weight: float = 0.35
    cycle_weight: float = 0.015
    operator_cost_weight: float = 0.002
    cycle_order: int = 4
    grad_clip: float = 1.0


cfg = ARMConfig()


class CyclicFanMemoryDataset(torch.utils.data.Dataset):
    """
    Synthetic cyclic hidden-state benchmark.

    The hidden state lives in Z_n. Each sample gives two different sensory or
    procedural cues that point to the same hidden memory state. The model must
    learn multi-query retrieval: different cues should converge to the same
    memory atom.
    """

    def __init__(
        self,
        num_samples: int,
        num_states: int,
        cue_types: int,
        dim: int,
        noise_std: float,
        seed: int,
        shared_basis: Dict[str, torch.Tensor] = None,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.num_samples = num_samples
        self.num_states = num_states
        self.cue_types = cue_types
        self.dim = dim
        self.noise_std = noise_std

        if shared_basis is None:
            state_base = F.normalize(torch.randn(num_states, dim, generator=g), dim=-1)
            cue_maps = torch.randn(cue_types, dim, dim, generator=g) / math.sqrt(dim)
            cue_maps = 0.70 * torch.eye(dim).unsqueeze(0) + 0.30 * cue_maps
            pull_embed = F.normalize(torch.randn(num_states, dim, generator=g), dim=-1)
            self.shared_basis = {
                "state_base": state_base,
                "cue_maps": cue_maps,
                "pull_embed": pull_embed,
            }
        else:
            self.shared_basis = shared_basis

        self.state_base = self.shared_basis["state_base"]
        self.cue_maps = self.shared_basis["cue_maps"]
        self.pull_embed = self.shared_basis["pull_embed"]

        self.states = torch.randint(0, num_states, (num_samples,), generator=g)
        self.cue_a = torch.randint(0, cue_types, (num_samples,), generator=g)
        self.cue_b = torch.randint(0, cue_types, (num_samples,), generator=g)
        self.pulls_seen = torch.randint(0, 12, (num_samples,), generator=g)

        self.q_a = self._make_queries(self.states, self.cue_a, self.pulls_seen, g)
        self.q_b = self._make_queries(self.states, self.cue_b, self.pulls_seen + 1, g)

    def _make_queries(self, states, cue_ids, pulls_seen, g):
        base = self.state_base[states]
        cue_matrix = self.cue_maps[cue_ids]
        phase = pulls_seen % self.num_states
        phase_signal = self.pull_embed[phase]
        x = base + 0.35 * phase_signal
        q = torch.bmm(cue_matrix, x.unsqueeze(-1)).squeeze(-1)
        q = torch.tanh(q)
        q = q + self.noise_std * torch.randn(q.shape, generator=g)
        return q.float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return {
            "q_a": self.q_a[idx],
            "q_b": self.q_b[idx],
            "state": self.states[idx],
            "cue_a": self.cue_a[idx],
            "cue_b": self.cue_b[idx],
        }


class AlgebraicResonanceMemory(nn.Module):
    """
    Algebraic Resonance Memory.

    Given query q, memory atom m_i, and operator family A_k, ARM scores:

        rho(q, m_i) = logsumexp_k(-||A_k q + b_k - m_i||^2_D / tau - c_k)

    The retrieval distribution is softmax over rho.
    """

    def __init__(self, dim: int, num_memories: int, num_operators: int, tau: float):
        super().__init__()
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

        self.query_norm = nn.LayerNorm(dim)
        self.query_proj = nn.Sequential(
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
        q = self.query_proj(self.query_norm(q))
        transformed = torch.einsum("bd,ked->bke", q, self.operators)
        return transformed + self.operator_bias.unsqueeze(0)

    def resonance_logits(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        transformed = self.transformed_queries(q)
        metric = self.positive_metric()
        costs = self.operator_cost()

        diff = transformed.unsqueeze(2) - self.memory.unsqueeze(0).unsqueeze(0)
        dist = (diff.pow(2) * metric.view(1, 1, 1, -1)).sum(dim=-1)
        path_scores = -dist / self.tau - costs.view(1, -1, 1)
        logits = torch.logsumexp(path_scores, dim=1)
        return logits, path_scores

    def forward(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits, path_scores = self.resonance_logits(q)
        weights = F.softmax(logits, dim=-1)
        retrieved = weights @ self.memory
        return {
            "retrieved": retrieved,
            "weights": weights,
            "logits": logits,
            "path_scores": path_scores,
        }

    def cycle_consistency_loss(self, order: int) -> torch.Tensor:
        A = self.operators[0]
        A_power = torch.matrix_power(A, order)
        I = torch.eye(self.dim, device=A.device, dtype=A.dtype)
        return F.mse_loss(A_power, I)

    def operator_regularization(self) -> torch.Tensor:
        I = torch.eye(self.dim, device=self.operators.device, dtype=self.operators.dtype)
        near_identity = (self.operators - I.unsqueeze(0)).pow(2).mean()
        bias_penalty = self.operator_bias.pow(2).mean()
        cost_penalty = self.operator_cost().mean()
        return near_identity + 0.1 * bias_penalty + 0.01 * cost_penalty


class ARMClassifier(nn.Module):
    def __init__(self, cfg: ARMConfig):
        super().__init__()
        self.arm = AlgebraicResonanceMemory(
            dim=cfg.dim,
            num_memories=cfg.num_states,
            num_operators=cfg.num_operators,
            tau=cfg.tau,
        )

    def forward(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.arm(q)


def make_loaders(cfg: ARMConfig):
    train_ds = CyclicFanMemoryDataset(
        cfg.train_samples,
        cfg.num_states,
        cfg.cue_types,
        cfg.dim,
        cfg.noise_std,
        seed=123,
    )
    val_ds = CyclicFanMemoryDataset(
        cfg.val_samples,
        cfg.num_states,
        cfg.cue_types,
        cfg.dim,
        cfg.noise_std,
        seed=456,
        shared_basis=train_ds.shared_basis,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_ds, val_ds, train_loader, val_loader


def compute_loss(model: ARMClassifier, batch: Dict[str, torch.Tensor], cfg: ARMConfig):
    q_a = batch["q_a"].to(device)
    q_b = batch["q_b"].to(device)
    state = batch["state"].to(device)

    out_a = model(q_a)
    out_b = model(q_b)

    loss_ret = 0.5 * (
        F.cross_entropy(out_a["logits"], state)
        + F.cross_entropy(out_b["logits"], state)
    )

    loss_eq = F.mse_loss(out_a["retrieved"], out_b["retrieved"]) + 0.25 * F.mse_loss(
        out_a["weights"], out_b["weights"]
    )
    loss_cycle = model.arm.cycle_consistency_loss(cfg.cycle_order)
    loss_op = model.arm.operator_regularization()

    loss = (
        cfg.retrieval_weight * loss_ret
        + cfg.equivalence_weight * loss_eq
        + cfg.cycle_weight * loss_cycle
        + cfg.operator_cost_weight * loss_op
    )

    with torch.no_grad():
        pred_a = out_a["logits"].argmax(dim=-1)
        pred_b = out_b["logits"].argmax(dim=-1)
        metrics = {
            "loss": float(loss.item()),
            "loss_ret": float(loss_ret.item()),
            "loss_eq": float(loss_eq.item()),
            "loss_cycle": float(loss_cycle.item()),
            "loss_op": float(loss_op.item()),
            "acc_a": float((pred_a == state).float().mean().item()),
            "acc_b": float((pred_b == state).float().mean().item()),
            "agreement": float((pred_a == pred_b).float().mean().item()),
        }
    return loss, metrics


@torch.no_grad()
def evaluate(model: ARMClassifier, loader, cfg: ARMConfig):
    model.eval()
    totals = {
        "loss": 0.0,
        "loss_ret": 0.0,
        "loss_eq": 0.0,
        "loss_cycle": 0.0,
        "loss_op": 0.0,
        "acc_a": 0.0,
        "acc_b": 0.0,
        "agreement": 0.0,
    }
    count = 0
    for batch in loader:
        _, metrics = compute_loss(model, batch, cfg)
        bs = batch["state"].shape[0]
        for key in totals:
            totals[key] += metrics[key] * bs
        count += bs
    return {key: value / max(1, count) for key, value in totals.items()}


def run_tests():
    print("\nRunning ARM tests...")
    test_cfg = ARMConfig(train_samples=512, val_samples=128, batch_size=64, epochs=1, dim=32, num_operators=4)
    test_ds = CyclicFanMemoryDataset(
        num_samples=128,
        num_states=test_cfg.num_states,
        cue_types=test_cfg.cue_types,
        dim=test_cfg.dim,
        noise_std=test_cfg.noise_std,
        seed=999,
    )
    test_model = ARMClassifier(test_cfg).to(device)
    batch = next(iter(torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=True)))
    out = test_model(batch["q_a"].to(device))
    assert out["retrieved"].shape == (16, test_cfg.dim)
    assert out["weights"].shape == (16, test_cfg.num_states)
    assert out["logits"].shape == (16, test_cfg.num_states)
    assert out["path_scores"].shape == (16, test_cfg.num_operators, test_cfg.num_states)
    loss, _ = compute_loss(test_model, batch, test_cfg)
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in test_model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"
    print("All tests passed.")


def train():
    train_ds, val_ds, train_loader, val_loader = make_loaders(cfg)
    model = ARMClassifier(cfg).to(device)
    print(model)
    run_tests()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_agreement": []}
    best_val_acc = 0.0

    print("\nStarting training...")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        seen = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_loss(model, batch, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            bs = batch["state"].shape[0]
            running_loss += metrics["loss"] * bs
            running_acc += 0.5 * (metrics["acc_a"] + metrics["acc_b"]) * bs
            seen += bs

        scheduler.step()
        train_loss = running_loss / seen
        train_acc = running_acc / seen
        val_metrics = evaluate(model, val_loader, cfg)
        val_acc = 0.5 * (val_metrics["acc_a"] + val_metrics["acc_b"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_agreement"].append(val_metrics["agreement"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "best_val_acc": best_val_acc,
                    "history": history,
                },
                "arm_colab_checkpoint.pt",
            )

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_acc:.4f} | "
            f"agreement={val_metrics['agreement']:.4f} | cycle={val_metrics['loss_cycle']:.5f}"
        )

    print("\nTraining complete.")
    print("Best validation accuracy:", round(best_val_acc, 4))
    print("Checkpoint saved to: arm_colab_checkpoint.pt")

    inspect_examples(model, val_ds)
    plot_history(history)
    return model, history


@torch.no_grad()
def inspect_examples(model: ARMClassifier, dataset, n: int = 12):
    model.eval()
    print("\nExample retrievals:")
    for i in range(n):
        item = dataset[i]
        q = item["q_a"].unsqueeze(0).to(device)
        state = int(item["state"].item())
        cue = int(item["cue_a"].item())
        probs = model(q)["weights"].squeeze(0).detach().cpu()
        pred = int(probs.argmax().item())
        probs_text = ", ".join([f"{p:.3f}" for p in probs.tolist()])
        print(f"sample={i:02d} | cue={cue} | true_state={state} | pred={pred} | memory_probs=[{probs_text}]")


def plot_history(history):
    if plt is None:
        return
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ARM Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="train accuracy")
    plt.plot(epochs, history["val_acc"], label="validation accuracy")
    plt.plot(epochs, history["val_agreement"], label="query agreement")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("ARM Retrieval Accuracy and Multi-query Agreement")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train()
