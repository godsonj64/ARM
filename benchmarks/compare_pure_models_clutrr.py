"""Compare root-level PureARMClassifier and StandardTransformerClassifier on CLUTRR.

Example:
    python benchmarks/compare_pure_models_clutrr.py --epochs 8

Outputs:
    benchmark_results/pure_models_clutrr/pure_models_clutrr.json
    benchmark_results/pure_models_clutrr/pure_arm_clutrr.pt
    benchmark_results/pure_models_clutrr/pure_transformer_clutrr.pt
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
from models import (  # noqa: E402
    ARMConfig,
    PureARMClassifier,
    StandardTransformerClassifier,
    TransformerConfig,
)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_model(
    kind: str,
    vocab_size: int,
    num_classes: int,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, object]:
    if kind == "pure_arm":
        config = ARMConfig(
            vocab_size=vocab_size,
            num_classes=num_classes,
            dim=args.dim,
            num_memory_atoms=num_classes,
            num_operators=args.num_operators,
            max_len=args.max_len,
            dropout=args.dropout,
            tau=args.tau,
            use_positional_encoding=True,
        )
        return PureARMClassifier(config), config

    if kind == "pure_transformer":
        transformer_dim = args.resolved_transformer_dim
        config = TransformerConfig(
            vocab_size=vocab_size,
            num_classes=num_classes,
            dim=transformer_dim,
            num_heads=args.resolved_transformer_heads,
            num_layers=args.resolved_transformer_layers,
            ff_dim=args.resolved_transformer_ff_dim,
            max_len=args.max_len,
            dropout=args.dropout,
            use_cls_token=args.use_cls_token,
        )
        return StandardTransformerClassifier(config), config

    raise ValueError(f"unknown model kind: {kind}")


def choose_matched_transformer(
    target_params: int,
    vocab_size: int,
    num_classes: int,
    args: argparse.Namespace,
) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int, int]] = []
    dim_values = range(args.match_min_dim, args.match_max_dim + 1, args.match_dim_step)
    ff_values = range(args.match_min_ff_dim, args.match_max_ff_dim + 1, args.match_ff_dim_step)

    for dim in dim_values:
        valid_heads = [head for head in args.match_heads if head > 0 and dim % head == 0]
        if not valid_heads:
            continue
        for heads in valid_heads:
            for layers in range(args.match_min_layers, args.match_max_layers + 1):
                for ff_dim in ff_values:
                    config = TransformerConfig(
                        vocab_size=vocab_size,
                        num_classes=num_classes,
                        dim=dim,
                        num_heads=heads,
                        num_layers=layers,
                        ff_dim=ff_dim,
                        max_len=args.max_len,
                        dropout=args.dropout,
                        use_cls_token=args.use_cls_token,
                    )
                    params = count_parameters(StandardTransformerClassifier(config))
                    candidates.append((abs(params - target_params), params, dim, heads, layers, ff_dim))

    if not candidates:
        raise ValueError("no valid Transformer candidates found for parameter matching")

    _, params, dim, heads, layers, ff_dim = min(candidates, key=lambda item: item[0])
    return params, dim, heads, layers, ff_dim


def batch_loss(
    model: torch.nn.Module,
    kind: str,
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    logits = model(input_ids, attention_mask)
    loss = F.cross_entropy(logits, labels)

    if kind == "pure_arm":
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
    model: torch.nn.Module,
    kind: str,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    seen = 0

    for batch in loader:
        loss, acc = batch_loss(model, kind, batch, args, device)
        batch_size = batch["label"].shape[0]
        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        seen += batch_size

    return {"loss": total_loss / max(1, seen), "acc": total_acc / max(1, seen)}


def train_one(
    kind: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, object, Dict[str, object]]:
    seed_all(args.seed)
    model, config = make_model(kind, vocab_size, num_classes, args)
    model = model.to(device)
    print(f"{kind} trainable parameters: {count_parameters(model):,}", flush=True)

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
            loss, acc = batch_loss(model, kind, batch, args, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            batch_size = batch["label"].shape[0]
            total_loss += loss.item() * batch_size
            total_acc += acc * batch_size
            seen += batch_size

        scheduler.step()
        val = evaluate(model, kind, val_loader, args, device)
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
            f"{kind:16s} epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val['loss']:.4f} | val_acc={val['acc']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    result = {
        "parameters": count_parameters(model),
        "best_val_acc": best_val_acc,
        "final": evaluate(model, kind, val_loader, args, device),
        "history": history,
    }
    return model, config, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="CLUTRR/v1")
    parser.add_argument("--preferred-config", type=str, default="gen_train234_test2to10")
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--max-eval", type=int, default=3000)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-operators", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.45)
    parser.add_argument("--transformer-dim", type=int, default=None)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-ff-dim", type=int, default=512)
    parser.add_argument("--use-cls-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--match-parameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--match-min-dim", type=int, default=64)
    parser.add_argument("--match-max-dim", type=int, default=160)
    parser.add_argument("--match-dim-step", type=int, default=8)
    parser.add_argument("--match-min-layers", type=int, default=1)
    parser.add_argument("--match-max-layers", type=int, default=6)
    parser.add_argument("--match-min-ff-dim", type=int, default=32)
    parser.add_argument("--match-max-ff-dim", type=int, default=768)
    parser.add_argument("--match-ff-dim-step", type=int, default=16)
    parser.add_argument("--match-heads", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cycle-weight", type=float, default=0.003)
    parser.add_argument("--cycle-order", type=int, default=4)
    parser.add_argument("--op-weight", type=float, default=0.001)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="benchmark_results/pure_models_clutrr")
    args = parser.parse_args()
    args.resolved_transformer_dim = args.transformer_dim or args.dim
    args.resolved_transformer_heads = args.transformer_heads
    args.resolved_transformer_layers = args.transformer_layers
    args.resolved_transformer_ff_dim = args.transformer_ff_dim

    set_threads()
    seed_all(args.seed)
    device = torch.device(args.device)
    print("Device:", device, flush=True)

    raw, used_config = load_clutrr(args.dataset_name, args.preferred_config)
    train_rows, val_rows = prepare_rows(raw, args.max_train, args.max_eval)
    labels = sorted({label for _, label in train_rows + val_rows})
    label_to_id = {label: index for index, label in enumerate(labels)}
    print("Labels:", labels)

    tokenizer = Tokenizer(args.max_vocab)
    tokenizer.fit([text for text, _ in train_rows])

    arm_for_count, _ = make_model("pure_arm", len(tokenizer.itos), len(labels), args)
    arm_params = count_parameters(arm_for_count)
    if args.match_parameters:
        matched_params, dim, heads, layers, ff_dim = choose_matched_transformer(
            arm_params,
            len(tokenizer.itos),
            len(labels),
            args,
        )
        args.resolved_transformer_dim = dim
        args.resolved_transformer_heads = heads
        args.resolved_transformer_layers = layers
        args.resolved_transformer_ff_dim = ff_dim
        print(
            "Parameter match: "
            f"ARM={arm_params:,}, Transformer={matched_params:,} "
            f"(dim={dim}, heads={heads}, layers={layers}, ff_dim={ff_dim}, "
            f"delta={matched_params - arm_params:+,})",
            flush=True,
        )

    train_ds = TextRowsDataset(train_rows, tokenizer, label_to_id, args.max_len)
    val_ds = TextRowsDataset(val_rows, tokenizer, label_to_id, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "dataset_config": used_config,
        "labels": labels,
        "vocab_size": len(tokenizer.itos),
        "args": vars(args),
        "parameter_match": {
            "target_model": "pure_arm",
            "target_parameters": arm_params,
            "transformer_dim": args.resolved_transformer_dim,
            "transformer_heads": args.resolved_transformer_heads,
            "transformer_layers": args.resolved_transformer_layers,
            "transformer_ff_dim": args.resolved_transformer_ff_dim,
        },
    }

    for kind in ("pure_arm", "pure_transformer"):
        model, config, result = train_one(
            kind,
            train_loader,
            val_loader,
            len(tokenizer.itos),
            len(labels),
            args,
            device,
        )
        checkpoint_path = out_dir / f"{kind}_clutrr.pt"
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
        summary[kind] = {**result, "checkpoint": str(checkpoint_path)}

    metrics_path = out_dir / "pure_models_clutrr.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nFinal Pure Model CLUTRR comparison:")
    print(
        json.dumps(
            {
                kind: {
                    "parameters": summary[kind]["parameters"],
                    "best_val_acc": summary[kind]["best_val_acc"],
                    "final": summary[kind]["final"],
                }
                for kind in ("pure_arm", "pure_transformer")
            },
            indent=2,
        )
    )
    print(f"Saved metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
