from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from standalone_arm_experiments.data import TextDataset, Tokenizer, load_clutrr_csvs, load_local, load_synthetic, rows_from_records
from standalone_arm_experiments.models import ModelConfig, build_model


VALID_MODELS = ["arm", "dot_memory", "prototype", "rbf", "hopfield", "transformer"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parse_models(value: str) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [model for model in models if model not in VALID_MODELS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown model(s): {unknown}; choose from {VALID_MODELS}")
    return models


def load_dataset(args: argparse.Namespace) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    if args.dataset == "synthetic":
        return load_synthetic(args.max_train, args.max_eval, args.max_test, args.seed)
    if args.dataset == "local":
        if not args.data_path:
            raise ValueError("--data-path is required for --dataset local")
        return load_local(args.data_path, args.max_train, args.max_eval, args.max_test, args.seed)
    if args.dataset == "clutrr":
        splits = load_clutrr_csvs(args.clutrr_config)
        return (
            rows_from_records(splits["train"], args.max_train),
            rows_from_records(splits["validation"], args.max_eval),
            rows_from_records(splits["test"], args.max_test),
        )
    raise ValueError(args.dataset)


def batch_loss(model: torch.nn.Module, name: str, batch: Dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)
    out = model(input_ids, attention_mask)
    loss = F.cross_entropy(out["logits"], labels)
    if name == "arm":
        loss = loss + args.op_weight * model.memory.operator_regularization()
        if args.cycle_weight > 0:
            loss = loss + args.cycle_weight * model.memory.cycle_consistency_loss(args.cycle_order)
    with torch.no_grad():
        acc = (out["logits"].argmax(dim=-1) == labels).float().mean().item()
    return loss, acc


@torch.no_grad()
def evaluate(model: torch.nn.Module, name: str, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
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


def train_one(name: str, config: ModelConfig, loaders: dict[str, DataLoader], args: argparse.Namespace, device: torch.device):
    seed_all(args.seed)
    model = build_model(name, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None
    print(f"\n{name} parameters: {count_parameters(model):,}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        seen = 0
        for batch in loaders["train"]:
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
        val = evaluate(model, name, loaders["val"], args, device)
        history["train_loss"].append(total_loss / max(1, seen))
        history["train_acc"].append(total_acc / max(1, seen))
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(f"{name:12s} epoch {epoch:03d}/{args.epochs} train_acc={history['train_acc'][-1]:.4f} val_acc={val['acc']:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "parameters": count_parameters(model),
        "best_val_acc": best_val_acc,
        "final_val": evaluate(model, name, loaders["val"], args, device),
        "test": evaluate(model, name, loaders["test"], args, device),
        "history": history,
    }


@torch.no_grad()
def sample_predictions(models: dict[str, torch.nn.Module], examples: Sequence[Dict[str, Any]], tokenizer: Tokenizer, label_to_id: Dict[str, int], args: argparse.Namespace, device: torch.device):
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
            pred_id = int(probs.argmax().item())
            predictions[name] = {
                "predicted": id_to_label[pred_id],
                "confidence": round(float(probs[pred_id].item()), 4),
                "correct": id_to_label[pred_id] == example["label"],
            }
            if name == "arm" and "weights" in out:
                weights = out["weights"][0]
                top = torch.topk(weights, k=min(args.inspect_top_k, weights.numel()))
                predictions[name]["top_memory_atoms"] = [
                    {"label": id_to_label[int(idx.item())], "weight": round(float(score.item()), 4)}
                    for score, idx in zip(top.values, top.indices)
                ]
        samples.append({"sample": index, "true": example["label"], "group": example.get("group", "unknown"), "text": example["text"][:300], "predictions": predictions})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "clutrr", "local"], default="synthetic")
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--clutrr-config", type=str, default="gen_train234_test2to10")
    parser.add_argument("--models", type=parse_models, default=parse_models("arm,dot_memory,prototype,rbf,hopfield,transformer"))
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--max-eval", type=int, default=3000)
    parser.add_argument("--max-test", type=int, default=3000)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-operators", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.45)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--transformer-dim", type=int, default=128)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ff-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--op-weight", type=float, default=1e-3)
    parser.add_argument("--cycle-weight", type=float, default=0.0)
    parser.add_argument("--cycle-order", type=int, default=4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--inspect-top-k", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default=str(HERE / "results"))
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device)
    train_examples, val_examples, test_examples = load_dataset(args)
    labels = sorted({example["label"] for example in train_examples + val_examples + test_examples})
    label_to_id = {label: index for index, label in enumerate(labels)}
    tokenizer = Tokenizer(args.max_vocab)
    tokenizer.fit([example["text"] for example in train_examples])
    loaders = {
        "train": DataLoader(TextDataset(train_examples, tokenizer, label_to_id, args.max_len), batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(TextDataset(val_examples, tokenizer, label_to_id, args.max_len), batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(TextDataset(test_examples, tokenizer, label_to_id, args.max_len), batch_size=args.batch_size, shuffle=False),
    }
    config = ModelConfig(
        vocab_size=len(tokenizer.itos),
        num_classes=len(labels),
        max_len=args.max_len,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_memory_atoms=len(labels),
        num_operators=args.num_operators,
        tau=args.tau,
        dropout=args.dropout,
        transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_ff_dim=args.transformer_ff_dim,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {
        "dataset": args.dataset,
        "labels": labels,
        "vocab_size": len(tokenizer.itos),
        "splits": {"train": len(train_examples), "validation": len(val_examples), "test": len(test_examples)},
        "args": vars(args),
        "models": {},
    }
    trained_models = {}
    for name in args.models:
        model, metrics = train_one(name, config, loaders, args, device)
        trained_models[name] = model
        results["models"][name] = metrics
        torch.save({"model": name, "state_dict": model.state_dict(), "config": config, "labels": labels, "tokenizer_stoi": tokenizer.stoi}, out_dir / f"{name}.pt")
    results["samples"] = sample_predictions(trained_models, test_examples, tokenizer, label_to_id, args, device)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print("\nSummary:")
    print(json.dumps({name: {"parameters": item["parameters"], "best_val_acc": item["best_val_acc"], "test_acc": item["test"]["acc"]} for name, item in results["models"].items()}, indent=2))
    print(f"Wrote: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
