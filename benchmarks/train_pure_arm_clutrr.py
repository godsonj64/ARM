"""Train the root-level PureARMClassifier from models.py on CLUTRR.

Example:
    python benchmarks/train_pure_arm_clutrr.py --epochs 8

Outputs:
    benchmark_results/pure_arm_clutrr.json
    benchmark_results/pure_arm_clutrr.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
for path in (ROOT, BENCHMARK_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from clutrr_compare import (  # noqa: E402
    TextRowsDataset,
    Tokenizer,
    load_clutrr,
    prepare_rows,
    seed_all,
    set_threads,
)
from models import ARMConfig, PureARMClassifier  # noqa: E402


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batch_loss(
    model: PureARMClassifier,
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    logits = model(input_ids, attention_mask)
    loss = F.cross_entropy(logits, labels)
    loss = loss + model.regularization_loss(
        operator_weight=args.op_weight,
        cycle_weight=args.cycle_weight,
        cycle_order=args.cycle_order,
    )

    with torch.no_grad():
        acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, acc


@torch.no_grad()
def evaluate(
    model: PureARMClassifier,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    seen = 0

    for batch in loader:
        loss, acc = batch_loss(model, batch, args, device)
        batch_size = batch["label"].shape[0]
        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        seen += batch_size

    return {"loss": total_loss / max(1, seen), "acc": total_acc / max(1, seen)}


def train(
    model: PureARMClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        seen = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, acc = batch_loss(model, batch, args, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            batch_size = batch["label"].shape[0]
            total_loss += loss.item() * batch_size
            total_acc += acc * batch_size
            seen += batch_size

        scheduler.step()
        val = evaluate(model, val_loader, args, device)
        train_loss = total_loss / max(1, seen)
        train_acc = total_acc / max(1, seen)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])

        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        print(
            f"pure_arm epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val['loss']:.4f} | val_acc={val['acc']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "parameters": count_parameters(model),
        "best_val_acc": best_val_acc,
        "final": evaluate(model, val_loader, args, device),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="CLUTRR/v1")
    parser.add_argument("--preferred-config", type=str, default="gen_train234_test2to10")
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--max-eval", type=int, default=3000)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-operators", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.45)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cycle-weight", type=float, default=0.003)
    parser.add_argument("--cycle-order", type=int, default=4)
    parser.add_argument("--op-weight", type=float, default=0.001)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="benchmark_results")
    parser.add_argument("--checkpoint-name", type=str, default="pure_arm_clutrr.pt")
    parser.add_argument("--metrics-name", type=str, default="pure_arm_clutrr.json")
    args = parser.parse_args()

    set_threads()
    seed_all(args.seed)
    device = torch.device(args.device)
    print("Device:", device)

    raw, used_config = load_clutrr(args.dataset_name, args.preferred_config)
    train_rows, val_rows = prepare_rows(raw, args.max_train, args.max_eval)
    labels = sorted({label for _, label in train_rows + val_rows})
    label_to_id = {label: index for index, label in enumerate(labels)}
    print("Labels:", labels)

    tokenizer = Tokenizer(args.max_vocab)
    tokenizer.fit([text for text, _ in train_rows])

    train_ds = TextRowsDataset(train_rows, tokenizer, label_to_id, args.max_len)
    val_ds = TextRowsDataset(val_rows, tokenizer, label_to_id, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    config = ARMConfig(
        vocab_size=len(tokenizer.itos),
        num_classes=len(labels),
        dim=args.dim,
        num_memory_atoms=len(labels),
        num_operators=args.num_operators,
        max_len=args.max_len,
        dropout=args.dropout,
        tau=args.tau,
        use_positional_encoding=True,
    )
    model = PureARMClassifier(config).to(device)
    print(f"PureARM trainable parameters: {count_parameters(model):,}")

    result = train(model, train_loader, val_loader, args, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / args.checkpoint_name
    metrics_path = out_dir / args.metrics_name

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "tokenizer_stoi": tokenizer.stoi,
            "tokenizer_itos": tokenizer.itos,
            "label_to_id": label_to_id,
            "labels": labels,
            "dataset_config": used_config,
            "args": vars(args),
            "result": result,
        },
        checkpoint_path,
    )

    summary = {
        "dataset_config": used_config,
        "labels": labels,
        "vocab_size": len(tokenizer.itos),
        "model": "PureARMClassifier",
        "result": result,
        "checkpoint": str(checkpoint_path),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nFinal PureARM CLUTRR summary:")
    print(json.dumps({"best_val_acc": result["best_val_acc"], "final": result["final"]}, indent=2))
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
