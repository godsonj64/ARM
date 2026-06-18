"""Compare repo ARM and Transformer on a local text classification/QA dataset.

This script uses the older repo ARM stack from arm/models.py:

    GRUTextEncoder -> AlgebraicResonanceMemory -> MemoryClassifier

It auto-loads common local dataset formats from a file or directory:
CSV, JSON, JSONL, and simple tab-separated TXT. Split names are inferred from
file names containing train, validation/val/dev, or test.

Example:
    python benchmarks/compare_arm_transformer_local_dataset.py \
        --data-path /Users/godsonjohnson/path/to/CLUTTER2.0 \
        --epochs 6
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
for path in (ROOT, BENCHMARK_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from arm.models import AlgebraicResonanceMemory, GRUTextEncoder, MemoryClassifier  # noqa: E402
from clutrr_multi_transformer_compare import TRANSFORMER_SPECS, TransformerClassifier  # noqa: E402


TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[.,!?;:()\-]")
SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".txt", ".tsv"}


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_threads() -> None:
    try:
        import os

        torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def textify(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(textify(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key} {textify(item)}" for key, item in value.items())
    return str(value)


def pick_column(columns: Sequence[str], candidates: Sequence[str], required: bool = True) -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for column in columns:
        for candidate in candidates:
            if candidate.lower() in column.lower():
                return column
    if required:
        raise KeyError(f"Could not find any of {candidates} in columns {columns}")
    return ""


class Tokenizer:
    def __init__(self, max_vocab: int):
        self.max_vocab = max_vocab
        self.stoi = {"[PAD]": 0, "[UNK]": 1}
        self.itos = ["[PAD]", "[UNK]"]

    def tokenize(self, text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    def fit(self, texts: Sequence[str]) -> None:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(self.tokenize(text))
        for token, _ in counts.most_common(self.max_vocab - len(self.itos)):
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
        print("Vocabulary size:", len(self.itos), flush=True)

    def encode(self, text: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self.stoi.get(token, 1) for token in self.tokenize(text)[:max_len]]
        mask = [1.0] * len(ids)
        ids += [0] * (max_len - len(ids))
        mask += [0.0] * (max_len - len(mask))
        return torch.tensor(ids).long(), torch.tensor(mask).float()


class TextDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Dict[str, Any]],
        tokenizer: Tokenizer,
        label_to_id: Dict[str, int],
        max_len: int,
    ):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        input_ids, attention_mask = self.tokenizer.encode(example["text"], self.max_len)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(self.label_to_id[example["label"]]).long(),
        }


def infer_split(path: Path) -> str:
    name = path.name.lower()
    if any(token in name for token in ["validation", "valid", "val", "dev"]):
        return "validation"
    if "test" in name:
        return "test"
    if "train" in name:
        return "train"
    return "unknown"


def dataset_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    direct_split_files = [
        path / name
        for name in ["train.csv", "validation.csv", "valid.csv", "val.csv", "dev.csv", "test.csv"]
        if (path / name).is_file()
    ]
    if direct_split_files:
        return direct_split_files
    files = [
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def normalize_record(record: Dict[str, Any], source: Path, row_index: int) -> Dict[str, Any]:
    columns = list(record.keys())
    text_col = pick_column(columns, ["text", "story", "clean_story", "context", "sentence", "input", "passage"])
    query_col = pick_column(columns, ["query", "question", "prompt"], required=False)
    label_col = pick_column(columns, ["target_text", "target", "answer", "relation", "target_label", "label", "class"])

    text = textify(record[text_col])
    query = textify(record[query_col]) if query_col else ""
    label = textify(record[label_col])
    combined = (text + " [QUERY] " + query).strip() if query else text.strip()

    return {
        "text": combined,
        "label": label,
        "source_file": str(source),
        "row_index": row_index,
        "group": textify(record.get("task_name", record.get("task", record.get("group", source.stem)))),
    }


def rows_from_json_object(obj: Any) -> list[Dict[str, Any]]:
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict):
        for key in ["data", "examples", "rows", "records", "items"]:
            value = obj.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [obj]
    return []


def load_file(path: Path) -> list[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            return rows_from_json_object(json.load(file))
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        rows.append(item)
        return rows
    if suffix in {".txt", ".tsv"}:
        rows = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    rows.append({"text": parts[0], "question": parts[1], "label": parts[2]})
                elif len(parts) == 2:
                    rows.append({"text": parts[0], "label": parts[1]})
        return rows
    return []


def load_local_dataset(args: argparse.Namespace) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    root = Path(args.data_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    split_rows: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    files = dataset_files(root)
    if not files:
        raise RuntimeError(f"No supported dataset files found under {root}")

    for file in files:
        raw_rows = load_file(file)
        split = infer_split(file)
        for index, row in enumerate(raw_rows):
            if not isinstance(row, dict):
                continue
            try:
                example = normalize_record(row, file, index)
            except KeyError as exc:
                if args.strict_columns:
                    raise
                print(f"Skipping {file}:{index} because {exc}", flush=True)
                continue
            split_rows[split].append(example)

    train = split_rows.get("train", [])
    validation = split_rows.get("validation", [])
    test = split_rows.get("test", [])
    unknown = split_rows.get("unknown", [])

    if not train and unknown:
        random.Random(args.seed).shuffle(unknown)
        train_cut = int(0.8 * len(unknown))
        val_cut = int(0.9 * len(unknown))
        train = unknown[:train_cut]
        validation = unknown[train_cut:val_cut]
        test = unknown[val_cut:]
    elif train and not validation:
        random.Random(args.seed).shuffle(train)
        val_size = max(1, int(args.val_fraction * len(train)))
        validation = train[-val_size:]
        train = train[:-val_size]
    if train and not test:
        test = validation

    train = train[: args.max_train]
    validation = validation[: args.max_eval]
    test = test[: args.max_test]

    if not train or not validation or not test:
        raise RuntimeError(
            f"Could not create train/validation/test splits from {root}. "
            f"Found sizes: train={len(train)}, validation={len(validation)}, test={len(test)}"
        )

    print(f"Loaded local dataset: {root}", flush=True)
    print(f"Files used: {len(files)}", flush=True)
    print(f"Splits: train={len(train)}, validation={len(validation)}, test={len(test)}", flush=True)
    return train, validation, test


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_model(name: str, vocab_size: int, num_labels: int, args: argparse.Namespace) -> nn.Module:
    if name == "arm":
        encoder = GRUTextEncoder(vocab_size, args.emb_dim, args.hidden_dim, args.latent_dim, dropout=args.dropout)
        memory = AlgebraicResonanceMemory(args.latent_dim, num_labels, args.num_operators, tau=args.tau)
        return MemoryClassifier(encoder, memory)
    if name == "transformer":
        spec = TRANSFORMER_SPECS[args.transformer_spec]
        return TransformerClassifier(vocab_size, num_labels, args.max_len, spec, args.dropout)
    raise ValueError(name)


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
def inspect_arm(out: Dict[str, torch.Tensor], id_to_label: Dict[int, str], top_k: int) -> Dict[str, Any]:
    logits = out["logits"][0]
    weights = out["weights"][0]
    pred_id = int(logits.argmax(dim=-1).item())
    top_memory = []
    for score, index in zip(*torch.topk(weights, k=min(top_k, weights.numel()))):
        label_id = int(index.item())
        top_memory.append(
            {
                "label": id_to_label[label_id],
                "memory_id": label_id,
                "weight": round(float(score.item()), 4),
                "logit": round(float(logits[label_id].item()), 4),
            }
        )
    top_paths = []
    path_scores = out.get("path_scores")
    if path_scores is not None:
        scores = path_scores[0, :, pred_id]
        for score, index in zip(*torch.topk(scores, k=min(top_k, scores.numel()))):
            top_paths.append({"operator_id": int(index.item()), "path_score": round(float(score.item()), 4)})
    return {
        "predicted": id_to_label[pred_id],
        "confidence": round(float(F.softmax(logits, dim=-1).max().item()), 4),
        "top_memory_atoms": top_memory,
        "top_operator_paths": top_paths,
    }


@torch.no_grad()
def evaluate_examples(
    models: Dict[str, nn.Module],
    examples: Sequence[Dict[str, Any]],
    tokenizer: Tokenizer,
    label_to_id: Dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    id_to_label = {index: label for label, index in label_to_id.items()}
    totals = {name: {"correct": 0, "total": 0, "by_group": defaultdict(lambda: {"correct": 0, "total": 0})} for name in models}
    samples = []

    for index, example in enumerate(examples):
        input_ids, attention_mask = tokenizer.encode(example["text"], args.max_len)
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        group = str(example.get("group", "unknown"))
        predictions = {}

        for name, model in models.items():
            model.eval()
            out = model(input_ids, attention_mask)
            logits = out["logits"][0]
            pred_id = int(logits.argmax(dim=-1).item())
            predicted = id_to_label[pred_id]
            correct = predicted == example["label"]
            totals[name]["correct"] += int(correct)
            totals[name]["total"] += 1
            totals[name]["by_group"][group]["correct"] += int(correct)
            totals[name]["by_group"][group]["total"] += 1
            if name == "arm":
                predictions[name] = {**inspect_arm(out, id_to_label, args.inspect_top_k), "correct": correct}
            else:
                probs = F.softmax(logits, dim=-1)
                predictions[name] = {
                    "predicted": predicted,
                    "confidence": round(float(probs.max().item()), 4),
                    "correct": correct,
                }

        if len(samples) < args.num_samples:
            samples.append(
                {
                    "sample": index,
                    "text": example["text"],
                    "true": example["label"],
                    "group": group,
                    "predictions": predictions,
                }
            )

    summary = {}
    for name, values in totals.items():
        summary[name] = {
            "acc": values["correct"] / max(1, values["total"]),
            "total": values["total"],
            "by_group": {
                group: {"acc": counts["correct"] / max(1, counts["total"]), "total": counts["total"]}
                for group, counts in sorted(values["by_group"].items(), key=lambda item: item[0])
            },
        }
    return summary, samples


def print_samples(samples: Sequence[Dict[str, Any]]) -> None:
    print("\nSample predictions", flush=True)
    for sample in samples:
        print(
            f"\nsample={sample['sample']} group={sample['group']} true={sample['true']}\n"
            f"text={sample['text'][:320]}",
            flush=True,
        )
        for name, prediction in sample["predictions"].items():
            marker = "OK" if prediction["correct"] else "MISS"
            print(
                f"  {name:11s} pred={prediction['predicted']:<18s} "
                f"conf={prediction['confidence']:.4f} {marker}",
                flush=True,
            )
            if name == "arm":
                print(f"    top_memory_atoms={prediction['top_memory_atoms']}", flush=True)
                print(f"    top_operator_paths={prediction['top_operator_paths']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--max-eval", type=int, default=3000)
    parser.add_argument("--max-test", type=int, default=3000)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--strict-columns", action="store_true")
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=6)
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="benchmark_results/local_arm_transformer")
    args = parser.parse_args()

    set_threads()
    seed_all(args.seed)
    device = torch.device(args.device)
    print("Device:", device, flush=True)
    print("ARM source: arm/models.py", flush=True)
    print(f"Transformer spec: {args.transformer_spec}={TRANSFORMER_SPECS[args.transformer_spec]}", flush=True)

    train_examples, val_examples, test_examples = load_local_dataset(args)
    labels = sorted({example["label"] for example in train_examples + val_examples + test_examples})
    label_to_id = {label: index for index, label in enumerate(labels)}
    print("Labels:", labels, flush=True)

    tokenizer = Tokenizer(args.max_vocab)
    tokenizer.fit([example["text"] for example in train_examples])
    train_ds = TextDataset(train_examples, tokenizer, label_to_id, args.max_len)
    val_ds = TextDataset(val_examples, tokenizer, label_to_id, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    models = {}
    results: Dict[str, Any] = {
        "dataset_path": str(Path(args.data_path).expanduser().resolve()),
        "labels": labels,
        "vocab_size": len(tokenizer.itos),
        "splits": {"train": len(train_examples), "validation": len(val_examples), "test": len(test_examples)},
        "args": vars(args),
        "models": {},
    }

    for name in ("arm", "transformer"):
        model, metrics = train_one(name, train_loader, val_loader, len(tokenizer.itos), len(labels), args, device)
        models[name] = model
        results["models"][name] = metrics

    test_summary, samples = evaluate_examples(models, test_examples, tokenizer, label_to_id, args, device)
    for name, summary in test_summary.items():
        results["models"][name]["test"] = summary
        print(f"{name} test_acc={summary['acc']:.4f} by_group={json.dumps(summary['by_group'])}", flush=True)

    results["samples"] = samples
    print_samples(samples)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "local_arm_transformer_compare.json"
    with out_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    compact = {
        name: {
            "parameters": metrics["parameters"],
            "best_val_acc": metrics["best_val_acc"],
            "test_acc": metrics["test"]["acc"],
        }
        for name, metrics in results["models"].items()
    }
    print("\nFinal summary:", flush=True)
    print(json.dumps(compact, indent=2), flush=True)
    print(f"Wrote: {out_file}", flush=True)


if __name__ == "__main__":
    main()
