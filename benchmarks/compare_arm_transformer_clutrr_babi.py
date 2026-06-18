"""Compare repo ARM architecture against a Transformer on CLUTRR and bAbI.

This script uses the older repo ARM stack from arm/models.py:

    GRUTextEncoder -> AlgebraicResonanceMemory -> MemoryClassifier

It prints held-out sample predictions and, for ARM, memory-inspection details:
top memory atoms, resonance weights, and top operator path scores.

Example:
    python benchmarks/compare_arm_transformer_clutrr_babi.py --epochs 6

Faster smoke run:
    python benchmarks/compare_arm_transformer_clutrr_babi.py --epochs 1 --max-train 512 --max-eval 128 --max-test 128
"""

from __future__ import annotations

import argparse
import json
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
from clutrr_multi_transformer_compare import TRANSFORMER_SPECS, TransformerClassifier  # noqa: E402
from inductive_reasoning_compare import (  # noqa: E402
    TextExampleDataset,
    Tokenizer,
    load_babi_examples,
    load_clutrr_examples,
    parse_lengths,
    seed_all,
    set_threads,
)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_arm_model(vocab_size: int, num_labels: int, args: argparse.Namespace) -> MemoryClassifier:
    encoder = GRUTextEncoder(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.latent_dim,
        dropout=args.dropout,
    )
    memory = AlgebraicResonanceMemory(
        dim=args.latent_dim,
        num_memories=num_labels,
        num_operators=args.num_operators,
        tau=args.tau,
    )
    return MemoryClassifier(encoder, memory)


def make_model(name: str, vocab_size: int, num_labels: int, args: argparse.Namespace) -> nn.Module:
    if name == "arm":
        return make_arm_model(vocab_size, num_labels, args)
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
    logits = out["logits"]
    loss = F.cross_entropy(logits, labels)

    if name == "arm":
        loss = loss + args.cycle_weight * model.memory_layer.cycle_consistency_loss(args.cycle_order)
        loss = loss + args.op_weight * model.memory_layer.operator_regularization()

    with torch.no_grad():
        acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, acc


@torch.no_grad()
def evaluate_loader(
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


def train_one(
    name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
    num_labels: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, Dict[str, Any]]:
    seed_all(args.seed)
    model = make_model(name, vocab_size, num_labels, args).to(device)
    params = count_parameters(model)
    print(f"{name} trainable parameters: {params:,}", flush=True)

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
            loss, acc = batch_loss(model, name, batch, args, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            batch_size = batch["label"].shape[0]
            total_loss += loss.item() * batch_size
            total_acc += acc * batch_size
            seen += batch_size

        scheduler.step()
        val = evaluate_loader(model, name, val_loader, args, device)
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
            f"{name:11s} epoch {epoch:03d}/{args.epochs} | "
            f"train_acc={train_acc:.4f} | val_acc={val['acc']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "parameters": params,
        "best_val_acc": best_val_acc,
        "final_val": evaluate_loader(model, name, val_loader, args, device),
        "history": history,
    }


@torch.no_grad()
def inspect_arm_output(
    out: Dict[str, torch.Tensor],
    id_to_label: Dict[int, str],
    top_k: int,
) -> Dict[str, Any]:
    logits = out["logits"][0]
    weights = out["weights"][0]
    pred_id = int(logits.argmax(dim=-1).item())

    memory_scores = []
    for score, index in zip(*torch.topk(weights, k=min(top_k, weights.numel()))):
        label_id = int(index.item())
        memory_scores.append(
            {
                "label": id_to_label[label_id],
                "memory_id": label_id,
                "weight": round(float(score.item()), 4),
                "logit": round(float(logits[label_id].item()), 4),
            }
        )

    path_scores = out.get("path_scores")
    operator_scores = []
    if path_scores is not None:
        pred_path_scores = path_scores[0, :, pred_id]
        for score, index in zip(*torch.topk(pred_path_scores, k=min(top_k, pred_path_scores.numel()))):
            operator_scores.append(
                {
                    "operator_id": int(index.item()),
                    "path_score": round(float(score.item()), 4),
                }
            )

    return {
        "predicted": id_to_label[pred_id],
        "confidence": round(float(F.softmax(logits, dim=-1).max().item()), 4),
        "top_memory_atoms": memory_scores,
        "top_operator_paths_for_prediction": operator_scores,
    }


@torch.no_grad()
def evaluate_examples(
    models: Dict[str, nn.Module],
    examples: Sequence[Dict[str, Any]],
    tokenizer: Tokenizer,
    label_to_id: Dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    group_key: str,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    id_to_label = {index: label for label, index in label_to_id.items()}
    totals = {
        name: {"correct": 0, "total": 0, "by_group": {}}
        for name in models
    }
    samples = []

    for example_index, example in enumerate(examples):
        input_ids, attention_mask = tokenizer.encode(example["text"], args.max_len)
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        group = str(example.get(group_key, "unknown"))

        sample_predictions = {}
        for name, model in models.items():
            model.eval()
            out = model(input_ids, attention_mask)
            logits = out["logits"][0]
            pred_id = int(logits.argmax(dim=-1).item())
            predicted = id_to_label[pred_id]
            correct = predicted == example["label"]

            totals[name]["correct"] += int(correct)
            totals[name]["total"] += 1
            group_counts = totals[name]["by_group"].setdefault(group, {"correct": 0, "total": 0})
            group_counts["correct"] += int(correct)
            group_counts["total"] += 1

            if name == "arm":
                arm_inspection = inspect_arm_output(out, id_to_label, args.inspect_top_k)
                sample_predictions[name] = {**arm_inspection, "correct": correct}
            else:
                probs = F.softmax(logits, dim=-1)
                sample_predictions[name] = {
                    "predicted": predicted,
                    "confidence": round(float(probs.max().item()), 4),
                    "correct": correct,
                }

        if len(samples) < args.num_samples:
            samples.append(
                {
                    "sample": example_index,
                    "text": example["text"],
                    "true": example["label"],
                    "group": group,
                    "predictions": sample_predictions,
                }
            )

    summary = {}
    for name, values in totals.items():
        by_group = {
            group: {
                "acc": counts["correct"] / max(1, counts["total"]),
                "total": counts["total"],
            }
            for group, counts in sorted(values["by_group"].items(), key=lambda item: item[0])
        }
        summary[name] = {
            "acc": values["correct"] / max(1, values["total"]),
            "total": values["total"],
            "by_group": by_group,
        }

    return summary, samples


def print_samples(benchmark_name: str, samples: Sequence[Dict[str, Any]]) -> None:
    print(f"\n[{benchmark_name}] sample predictions", flush=True)
    for sample in samples:
        print(
            f"\nsample={sample['sample']} group={sample['group']} true={sample['true']}\n"
            f"text={sample['text'][:320]}",
            flush=True,
        )
        for model_name, prediction in sample["predictions"].items():
            marker = "OK" if prediction["correct"] else "MISS"
            print(
                f"  {model_name:11s} pred={prediction['predicted']:<18s} "
                f"conf={prediction['confidence']:.4f} {marker}",
                flush=True,
            )
            if model_name == "arm":
                print(f"    top_memory_atoms={prediction['top_memory_atoms']}", flush=True)
                print(
                    f"    top_operator_paths={prediction['top_operator_paths_for_prediction']}",
                    flush=True,
                )


def run_benchmark(
    benchmark_name: str,
    train_examples: Sequence[Dict[str, Any]],
    val_examples: Sequence[Dict[str, Any]],
    test_examples: Sequence[Dict[str, Any]],
    group_key: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    labels = sorted({example["label"] for example in train_examples + val_examples + test_examples})
    label_to_id = {label: index for index, label in enumerate(labels)}
    tokenizer = Tokenizer(args.max_vocab)
    tokenizer.fit([example["text"] for example in train_examples])

    train_ds = TextExampleDataset(train_examples, tokenizer, label_to_id, args.max_len)
    val_ds = TextExampleDataset(val_examples, tokenizer, label_to_id, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    models = {}
    results: Dict[str, Any] = {
        "labels": labels,
        "vocab_size": len(tokenizer.itos),
        "splits": {
            "train": len(train_examples),
            "validation": len(val_examples),
            "test": len(test_examples),
        },
        "models": {},
    }

    for name in ("arm", "transformer"):
        print(f"\n[{benchmark_name}] training {name}", flush=True)
        model, metrics = train_one(name, train_loader, val_loader, len(tokenizer.itos), len(labels), args, device)
        models[name] = model
        results["models"][name] = metrics

    test_summary, samples = evaluate_examples(
        models,
        test_examples,
        tokenizer,
        label_to_id,
        args,
        device,
        group_key,
    )
    for name, summary in test_summary.items():
        results["models"][name]["test"] = summary
        print(
            f"[{benchmark_name}] {name} test_acc={summary['acc']:.4f} "
            f"by_group={json.dumps(summary['by_group'])}",
            flush=True,
        )

    results["samples"] = samples
    print_samples(benchmark_name, samples)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=str, default="clutrr,babi")
    parser.add_argument("--dataset-name", type=str, default="CLUTRR/v1")
    parser.add_argument("--preferred-config", type=str, default="gen_train234_test2to10")
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--max-eval", type=int, default=3000)
    parser.add_argument("--max-test", type=int, default=3000)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-operators", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.45)
    parser.add_argument("--transformer-spec", type=str, default="small", choices=sorted(TRANSFORMER_SPECS))
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cycle-weight", type=float, default=0.003)
    parser.add_argument("--cycle-order", type=int, default=4)
    parser.add_argument("--op-weight", type=float, default=0.001)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--inspect-top-k", type=int, default=3)
    parser.add_argument("--babi-data-dir", type=str, default="data_babi")
    parser.add_argument("--babi-tasks", type=parse_lengths, default=parse_lengths("16"))
    parser.add_argument("--babi-val-fraction", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="benchmark_results/arm_transformer_reasoning")
    args = parser.parse_args()

    set_threads()
    seed_all(args.seed)
    device = torch.device(args.device)
    selected = {name.strip() for name in args.benchmarks.split(",") if name.strip()}
    print("Device:", device, flush=True)
    print("Benchmarks:", ", ".join(sorted(selected)), flush=True)
    print("ARM source: arm/models.py", flush=True)
    print(f"Transformer spec: {args.transformer_spec}={TRANSFORMER_SPECS[args.transformer_spec]}", flush=True)

    all_results: Dict[str, Any] = {
        "args": vars(args),
        "models": {
            "arm": "arm.models.GRUTextEncoder + arm.models.AlgebraicResonanceMemory + arm.models.MemoryClassifier",
            "transformer": f"benchmarks.clutrr_multi_transformer_compare.TransformerClassifier[{args.transformer_spec}]",
        },
        "benchmarks": {},
    }

    if "clutrr" in selected:
        used_config, train_examples, val_examples, test_examples = load_clutrr_examples(args)
        result = run_benchmark("clutrr", train_examples, val_examples, test_examples, "chain_length", args, device)
        result["dataset_config"] = used_config
        all_results["benchmarks"]["clutrr"] = result

    if "babi" in selected:
        train_examples, val_examples, test_examples = load_babi_examples(args)
        result = run_benchmark("babi", train_examples, val_examples, test_examples, "chain_length", args, device)
        result["tasks"] = args.babi_tasks
        all_results["benchmarks"]["babi"] = result

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "arm_transformer_clutrr_babi.json"
    with out_file.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)

    print("\nFinal summary:", flush=True)
    compact = {
        benchmark: {
            model_name: {
                "parameters": metrics["parameters"],
                "best_val_acc": metrics["best_val_acc"],
                "test_acc": metrics["test"]["acc"],
            }
            for model_name, metrics in result["models"].items()
        }
        for benchmark, result in all_results["benchmarks"].items()
    }
    print(json.dumps(compact, indent=2), flush=True)
    print(f"Wrote: {out_file}", flush=True)


if __name__ == "__main__":
    main()
