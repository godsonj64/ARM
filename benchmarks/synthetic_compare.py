"""Compare ARM against direct attention on the cyclic fan-state benchmark.

Run:
    python benchmarks/synthetic_compare.py --epochs 20 --device cpu

Outputs:
    benchmark_results/synthetic_compare.json
    benchmark_results/synthetic_learning_curves.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arm import AlgebraicResonanceMemory, AttentionMemory, CyclicFanMemoryDataset, MLPEncoder, MemoryClassifier

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_threads() -> None:
    try:
        torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def make_model(kind: str, input_dim: int, latent_dim: int, num_states: int, num_operators: int, dropout: float) -> MemoryClassifier:
    encoder = MLPEncoder(input_dim, latent_dim, dropout=dropout)
    if kind == "arm":
        memory = AlgebraicResonanceMemory(latent_dim, num_states, num_operators=num_operators)
    elif kind == "attention":
        memory = AttentionMemory(latent_dim, num_states, dropout=dropout)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return MemoryClassifier(encoder, memory)


def batch_loss(model: MemoryClassifier, batch: Dict[str, torch.Tensor], kind: str, device: torch.device, cycle_weight: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    q_a = batch["q_a"].to(device)
    q_b = batch["q_b"].to(device)
    y = batch["state"].to(device)
    out_a = model(q_a)
    out_b = model(q_b)
    retrieval = 0.5 * (F.cross_entropy(out_a["logits"], y) + F.cross_entropy(out_b["logits"], y))
    equivalence = F.mse_loss(out_a["weights"], out_b["weights"]) + F.mse_loss(out_a["retrieved"], out_b["retrieved"])
    loss = retrieval + 0.25 * equivalence
    cycle = torch.tensor(0.0, device=device)
    if kind == "arm":
        cycle = model.memory_layer.cycle_consistency_loss(order=4)
        loss = loss + cycle_weight * cycle + 0.001 * model.memory_layer.operator_regularization()
    with torch.no_grad():
        pred_a = out_a["logits"].argmax(-1)
        pred_b = out_b["logits"].argmax(-1)
        acc = 0.5 * ((pred_a == y).float().mean().item() + (pred_b == y).float().mean().item())
        agreement = (pred_a == pred_b).float().mean().item()
    return loss, {"loss": loss.item(), "acc": acc, "agreement": agreement, "cycle": cycle.item()}


@torch.no_grad()
def evaluate(model: MemoryClassifier, loader: DataLoader, kind: str, device: torch.device, cycle_weight: float) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "acc": 0.0, "agreement": 0.0, "cycle": 0.0}
    seen = 0
    for batch in loader:
        _, metrics = batch_loss(model, batch, kind, device, cycle_weight)
        bs = batch["state"].shape[0]
        for k in totals:
            totals[k] += metrics[k] * bs
        seen += bs
    return {k: v / max(1, seen) for k, v in totals.items()}


def train_one(kind: str, train_loader: DataLoader, val_loader: DataLoader, args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    model = make_model(kind, args.input_dim, args.latent_dim, args.num_states, args.num_operators, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_agreement": []}
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "acc": 0.0, "agreement": 0.0}
        seen = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, metrics = batch_loss(model, batch, kind, device, args.cycle_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            bs = batch["state"].shape[0]
            for k in totals:
                totals[k] += metrics[k] * bs
            seen += bs
        sched.step()
        val = evaluate(model, val_loader, kind, device, args.cycle_weight)
        train_loss = totals["loss"] / seen
        train_acc = totals["acc"] / seen
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        history["val_agreement"].append(val["agreement"])
        best = max(best, val["acc"])
        print(f"{kind:9s} epoch {epoch:03d}/{args.epochs} | train_acc={train_acc:.4f} | val_acc={val['acc']:.4f} | agreement={val['agreement']:.4f}")
    return {"best_val_acc": best, "final": evaluate(model, val_loader, kind, device, args.cycle_weight), "history": history}


def plot_results(results: Dict[str, object], out_dir: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    for name, result in results.items():
        plt.plot(result["history"]["val_acc"], label=f"{name} val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Synthetic cyclic benchmark: ARM vs direct attention")
    plt.grid(True)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "synthetic_learning_curves.png", dpi=160, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--val-samples", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-states", type=int, default=4)
    parser.add_argument("--cue-types", type=int, default=5)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-operators", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--cycle-weight", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    set_threads()
    seed_all(args.seed)
    device = torch.device(args.device)
    print("Device:", device)

    train_ds = CyclicFanMemoryDataset(args.train_samples, args.num_states, args.cue_types, args.input_dim, args.noise_std, seed=123)
    val_ds = CyclicFanMemoryDataset(args.val_samples, args.num_states, args.cue_types, args.input_dim, args.noise_std, seed=456, shared_basis=train_ds.shared_basis)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    results = {}
    for kind in ["attention", "arm"]:
        seed_all(args.seed)
        results[kind] = train_one(kind, train_loader, val_loader, args, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "synthetic_compare.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    plot_results(results, out_dir)
    print("\nFinal benchmark summary:")
    print(json.dumps({k: {"best_val_acc": v["best_val_acc"], "final": v["final"]} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
