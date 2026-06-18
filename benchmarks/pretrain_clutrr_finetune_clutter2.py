"""Pretrain repo ARM and Transformer on CLUTRR, then tune on CLUTTER 2.1 and 2.0.

This uses the older repo ARM architecture from arm/models.py:

    GRUTextEncoder -> AlgebraicResonanceMemory -> MemoryClassifier

Both models are trained in three stages:
1. Pretrain on original CLUTRR.
2. Save pretrain checkpoint.
3. Fine-tune the same model on local CLUTTER 2.1.
4. Save CLUTTER 2.1 checkpoint.
5. Further fine-tune on local CLUTTER 2.0.
6. Save final checkpoint and metrics.

Example:
    python benchmarks/pretrain_clutrr_finetune_clutter2.py \
        --clutter21-data-path "/Users/godsonjohnson/CLUTTER2.0/cluttr 2.1" \
        --clutter20-data-path /Users/godsonjohnson/CLUTTER2.0 \
        --pretrain-epochs 6 \
        --clutter21-epochs 10 \
        --clutter20-epochs 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
for path in (ROOT, BENCHMARK_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from arm.models import AlgebraicResonanceMemory, GRUTextEncoder, MemoryClassifier  # noqa: E402
from clutrr_compare import load_clutrr, pick_column, textify  # noqa: E402
from clutrr_multi_transformer_compare import TRANSFORMER_SPECS, TransformerClassifier  # noqa: E402
from compare_arm_transformer_local_dataset import (  # noqa: E402
    TextDataset,
    Tokenizer,
    load_local_dataset,
    seed_all,
    set_threads,
)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def rows_from_clutrr_split(split, limit: int) -> list[Dict[str, Any]]:
    columns = list(split[0].keys())
    story_col = pick_column(columns, ["story", "clean_story", "text", "context", "sentence"])
    query_col = pick_column(columns, ["query", "question"], required=False)
    label_col = pick_column(columns, ["target_text", "target", "answer", "relation", "target_label", "label"])
    task_col = pick_column(columns, ["task_name"], required=False)

    examples = []
    for index in range(min(len(split), limit)):
        row = split[index]
        story = textify(row[story_col])
        query = textify(row[query_col]) if query_col else ""
        examples.append(
            {
                "text": (story + " [QUERY] " + query).strip(),
                "label": textify(row[label_col]),
                "group": textify(row[task_col]) if task_col else "clutrr",
                "source": "clutrr",
            }
        )
    return examples


def load_clutrr_examples(args: argparse.Namespace) -> tuple[str, list[Dict[str, Any]], list[Dict[str, Any]]]:
    raw, used_config = load_clutrr(args.clutrr_dataset_name, args.clutrr_config)
    split_names = list(raw.keys())
    train_name = "train" if "train" in raw else split_names[0]
    val_name = next((name for name in ["validation", "val", "test"] if name in raw and name != train_name), None)
    if val_name is None:
        val_name = next(name for name in split_names if name != train_name)

    train = rows_from_clutrr_split(raw[train_name], args.max_pretrain)
    val = rows_from_clutrr_split(raw[val_name], args.max_pretrain_eval)
    print(f"CLUTRR pretrain config={used_config} train={len(train)} validation={len(val)}", flush=True)
    return used_config, train, val


def make_model(name: str, vocab_size: int, num_labels: int, args: argparse.Namespace) -> nn.Module:
    if name == "arm":
        encoder = GRUTextEncoder(vocab_size, args.emb_dim, args.hidden_dim, args.latent_dim, dropout=args.dropout)
        memory = AlgebraicResonanceMemory(args.latent_dim, num_labels, args.num_operators, tau=args.tau)
        return MemoryClassifier(encoder, memory)

    if name == "transformer":
        spec = TRANSFORMER_SPECS[args.transformer_spec]
        return TransformerClassifier(vocab_size, num_labels, args.max_len, spec, args.dropout)

    raise ValueError(f"unknown model: {name}")


def batch_loss(
    model: nn.Module,
    name: str,
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    out = model(input_ids, attention_mask)
    loss = F.cross_entropy(out["logits"], labels)
    if name == "arm":
        loss = loss + args.cycle_weight * model.memory_layer.cycle_consistency_loss(args.cycle_order)
        loss = loss + args.op_weight * model.memory_layer.operator_regularization()

    with torch.no_grad():
        acc = (out["logits"].argmax(dim=-1) == labels).float().mean().item()
    return loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    name: str,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    seen = 0

    for batch in loader:
        loss, acc = batch_loss(model, name, batch, args, device)
        batch_size = batch["label"].shape[0]
        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        seen += batch_size

    return {"loss": total_loss / max(1, seen), "acc": total_acc / max(1, seen)}


def train_phase(
    model: nn.Module,
    name: str,
    phase: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        seen = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, acc = batch_loss(model, name, batch, args, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            batch_size = batch["label"].shape[0]
            total_loss += loss.item() * batch_size
            total_acc += acc * batch_size
            seen += batch_size

        scheduler.step()
        val = evaluate(model, name, val_loader, args, device)
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
            f"{name:11s} {phase:9s} epoch {epoch:03d}/{epochs} | "
            f"train_acc={train_acc:.4f} | val_acc={val['acc']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_acc": best_val_acc,
        "final_val": evaluate(model, name, val_loader, args, device),
        "history": history,
    }


@torch.no_grad()
def sample_predictions(
    models: Dict[str, nn.Module],
    examples: Sequence[Dict[str, Any]],
    tokenizer: Tokenizer,
    label_to_id: Dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
) -> list[Dict[str, Any]]:
    id_to_label = {index: label for label, index in label_to_id.items()}
    samples = []
    for index, example in enumerate(examples[: args.num_samples]):
        input_ids, attention_mask = tokenizer.encode(example["text"], args.max_len)
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        predictions = {}
        for name, model in models.items():
            model.eval()
            out = model(input_ids, attention_mask)
            probs = F.softmax(out["logits"], dim=-1)[0]
            pred_id = int(probs.argmax(dim=-1).item())
            predictions[name] = {
                "predicted": id_to_label[pred_id],
                "confidence": round(float(probs[pred_id].item()), 4),
                "correct": id_to_label[pred_id] == example["label"],
            }
        samples.append(
            {
                "sample": index,
                "group": example.get("group", "unknown"),
                "true": example["label"],
                "text": example["text"][:400],
                "predictions": predictions,
            }
        )
    return samples


def print_samples(samples: Sequence[Dict[str, Any]]) -> None:
    print("\nFinal CLUTTER 2.0 sample predictions:", flush=True)
    for sample in samples:
        print(
            f"\nsample={sample['sample']} group={sample['group']} true={sample['true']}\n"
            f"text={sample['text']}",
            flush=True,
        )
        for name, pred in sample["predictions"].items():
            marker = "OK" if pred["correct"] else "MISS"
            print(
                f"  {name:11s} pred={pred['predicted']:<18s} conf={pred['confidence']:.4f} {marker}",
                flush=True,
            )


def make_loader(
    examples: Sequence[Dict[str, Any]],
    tokenizer: Tokenizer,
    label_to_id: Dict[str, int],
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = TextDataset(examples, tokenizer, label_to_id, args.max_len)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    name: str,
    stage: str,
    tokenizer: Tokenizer,
    label_to_id: Dict[str, int],
    args: argparse.Namespace,
    metrics: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": name,
            "stage": stage,
            "model_state": model.state_dict(),
            "tokenizer_stoi": tokenizer.stoi,
            "tokenizer_itos": tokenizer.itos,
            "label_to_id": label_to_id,
            "labels": [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])],
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )
    print(f"Saved {stage} checkpoint: {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clutter21-data-path", type=str, default="/Users/godsonjohnson/CLUTTER2.0/cluttr 2.1")
    parser.add_argument("--clutter20-data-path", type=str, default="/Users/godsonjohnson/CLUTTER2.0")
    parser.add_argument("--clutrr-dataset-name", type=str, default="CLUTRR/v1")
    parser.add_argument("--clutrr-config", type=str, default="gen_train234_test2to10")
    parser.add_argument("--max-pretrain", type=int, default=12000)
    parser.add_argument("--max-pretrain-eval", type=int, default=3000)
    parser.add_argument("--max-clutter21-train", type=int, default=12000)
    parser.add_argument("--max-clutter21-eval", type=int, default=3000)
    parser.add_argument("--max-clutter21-test", type=int, default=3000)
    parser.add_argument("--max-clutter20-train", type=int, default=12000)
    parser.add_argument("--max-clutter20-eval", type=int, default=3000)
    parser.add_argument("--max-clutter20-test", type=int, default=3000)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--strict-columns", action="store_true")
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--pretrain-epochs", type=int, default=6)
    parser.add_argument("--clutter21-epochs", type=int, default=10)
    parser.add_argument("--clutter20-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-operators", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.45)
    parser.add_argument("--transformer-spec", type=str, default="small", choices=sorted(TRANSFORMER_SPECS))
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--pretrain-lr", type=float, default=2e-3)
    parser.add_argument("--clutter21-lr", type=float, default=7.5e-4)
    parser.add_argument("--clutter20-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cycle-weight", type=float, default=0.003)
    parser.add_argument("--cycle-order", type=int, default=4)
    parser.add_argument("--op-weight", type=float, default=0.001)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="benchmark_results/pretrain_clutrr_clutter21_clutter20")
    args = parser.parse_args()

    set_threads()
    seed_all(args.seed)
    device = torch.device(args.device)
    print("Device:", device, flush=True)
    print("Stage 1 source: original CLUTRR", flush=True)
    print(f"Stage 2 source: {args.clutter21_data_path}", flush=True)
    print(f"Stage 3 source: {args.clutter20_data_path}", flush=True)
    print("ARM source: arm/models.py", flush=True)
    print(f"Transformer spec: {args.transformer_spec}={TRANSFORMER_SPECS[args.transformer_spec]}", flush=True)

    clutrr_config, pretrain_train, pretrain_val = load_clutrr_examples(args)

    clutter21_args = argparse.Namespace(**vars(args))
    clutter21_args.data_path = args.clutter21_data_path
    clutter21_args.max_train = args.max_clutter21_train
    clutter21_args.max_eval = args.max_clutter21_eval
    clutter21_args.max_test = args.max_clutter21_test
    clutter21_train, clutter21_val, clutter21_test = load_local_dataset(clutter21_args)

    clutter20_args = argparse.Namespace(**vars(args))
    clutter20_args.data_path = args.clutter20_data_path
    clutter20_args.max_train = args.max_clutter20_train
    clutter20_args.max_eval = args.max_clutter20_eval
    clutter20_args.max_test = args.max_clutter20_test
    clutter20_train, clutter20_val, clutter20_test = load_local_dataset(clutter20_args)

    labels = sorted(
        {
            example["label"]
            for example in (
                pretrain_train
                + pretrain_val
                + clutter21_train
                + clutter21_val
                + clutter21_test
                + clutter20_train
                + clutter20_val
                + clutter20_test
            )
        }
    )
    label_to_id = {label: index for index, label in enumerate(labels)}
    print("Unified labels:", labels, flush=True)

    tokenizer = Tokenizer(args.max_vocab)
    tokenizer.fit([example["text"] for example in pretrain_train + clutter21_train + clutter20_train])

    pretrain_train_loader = make_loader(pretrain_train, tokenizer, label_to_id, args, shuffle=True)
    pretrain_val_loader = make_loader(pretrain_val, tokenizer, label_to_id, args, shuffle=False)
    clutter21_train_loader = make_loader(clutter21_train, tokenizer, label_to_id, args, shuffle=True)
    clutter21_val_loader = make_loader(clutter21_val, tokenizer, label_to_id, args, shuffle=False)
    clutter21_test_loader = make_loader(clutter21_test, tokenizer, label_to_id, args, shuffle=False)
    clutter20_train_loader = make_loader(clutter20_train, tokenizer, label_to_id, args, shuffle=True)
    clutter20_val_loader = make_loader(clutter20_val, tokenizer, label_to_id, args, shuffle=False)
    clutter20_test_loader = make_loader(clutter20_test, tokenizer, label_to_id, args, shuffle=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {
        "clutrr_config": clutrr_config,
        "clutter21_data_path": str(Path(args.clutter21_data_path).expanduser().resolve()),
        "clutter20_data_path": str(Path(args.clutter20_data_path).expanduser().resolve()),
        "labels": labels,
        "vocab_size": len(tokenizer.itos),
        "splits": {
            "pretrain_train": len(pretrain_train),
            "pretrain_validation": len(pretrain_val),
            "clutter21_train": len(clutter21_train),
            "clutter21_validation": len(clutter21_val),
            "clutter21_test": len(clutter21_test),
            "clutter20_train": len(clutter20_train),
            "clutter20_validation": len(clutter20_val),
            "clutter20_test": len(clutter20_test),
        },
        "args": vars(args),
        "models": {},
    }
    trained_models = {}

    for name in ("arm", "transformer"):
        seed_all(args.seed)
        model = make_model(name, len(tokenizer.itos), len(labels), args).to(device)
        print(f"\n{name} trainable parameters: {count_parameters(model):,}", flush=True)

        pretrain_metrics = train_phase(
            model,
            name,
            "pretrain",
            pretrain_train_loader,
            pretrain_val_loader,
            args.pretrain_epochs,
            args.pretrain_lr,
            args,
            device,
        )
        pretrain_metrics["clutter21_test_before_tune"] = evaluate(model, name, clutter21_test_loader, args, device)
        pretrain_metrics["clutter20_test_before_tune"] = evaluate(model, name, clutter20_test_loader, args, device)
        save_checkpoint(
            out_dir / f"{name}_pretrained_clutrr.pt",
            model,
            name,
            "pretrained_clutrr",
            tokenizer,
            label_to_id,
            args,
            pretrain_metrics,
        )

        clutter21_metrics = train_phase(
            model,
            name,
            "clutter21",
            clutter21_train_loader,
            clutter21_val_loader,
            args.clutter21_epochs,
            args.clutter21_lr,
            args,
            device,
        )
        clutter21_metrics["clutter21_test"] = evaluate(model, name, clutter21_test_loader, args, device)
        clutter21_metrics["clutter20_test_before_stage3"] = evaluate(model, name, clutter20_test_loader, args, device)
        save_checkpoint(
            out_dir / f"{name}_finetuned_clutter21.pt",
            model,
            name,
            "finetuned_clutter21",
            tokenizer,
            label_to_id,
            args,
            clutter21_metrics,
        )

        clutter20_metrics = train_phase(
            model,
            name,
            "clutter20",
            clutter20_train_loader,
            clutter20_val_loader,
            args.clutter20_epochs,
            args.clutter20_lr,
            args,
            device,
        )
        clutter20_metrics["clutter20_test"] = evaluate(model, name, clutter20_test_loader, args, device)
        save_checkpoint(
            out_dir / f"{name}_finetuned_clutter20.pt",
            model,
            name,
            "finetuned_clutter20",
            tokenizer,
            label_to_id,
            args,
            clutter20_metrics,
        )

        results["models"][name] = {
            "parameters": count_parameters(model),
            "pretrain": pretrain_metrics,
            "clutter21": clutter21_metrics,
            "clutter20": clutter20_metrics,
        }
        trained_models[name] = model

    samples = sample_predictions(trained_models, clutter20_test, tokenizer, label_to_id, args, device)
    results["samples"] = samples
    print_samples(samples)

    metrics_path = out_dir / "pretrain_clutrr_clutter21_clutter20.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    compact = {
        name: {
            "parameters": metrics["parameters"],
            "pretrain_best_val_acc": metrics["pretrain"]["best_val_acc"],
            "clutter21_test_before_tune": metrics["pretrain"]["clutter21_test_before_tune"],
            "clutter20_test_before_tune": metrics["pretrain"]["clutter20_test_before_tune"],
            "clutter21_best_val_acc": metrics["clutter21"]["best_val_acc"],
            "clutter21_test_after_stage2": metrics["clutter21"]["clutter21_test"],
            "clutter20_test_before_stage3": metrics["clutter21"]["clutter20_test_before_stage3"],
            "clutter20_best_val_acc": metrics["clutter20"]["best_val_acc"],
            "clutter20_test_after_stage3": metrics["clutter20"]["clutter20_test"],
        }
        for name, metrics in results["models"].items()
    }
    print("\nFinal transfer summary:", flush=True)
    print(json.dumps(compact, indent=2), flush=True)
    print(f"Wrote metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
