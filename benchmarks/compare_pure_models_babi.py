"""Compare pure ARM and pure Transformer on Facebook bAbI QA tasks.

The default task is bAbI task 16, "basic induction". You can also run
multi-hop QA tasks, e.g. --babi-tasks 2,3,15,16.

Examples:
    python benchmarks/compare_pure_models_babi.py --epochs 8
    python benchmarks/compare_pure_models_babi.py --babi-tasks 2,3,15,16 --epochs 8

Outputs:
    benchmark_results/pure_models_babi/pure_models_babi.json
    benchmark_results/pure_models_babi/pure_arm_babi.pt
    benchmark_results/pure_models_babi/pure_transformer_babi.pt
"""

from __future__ import annotations

import argparse
import json
import random
import re
import ssl
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence
from urllib.request import Request, urlopen

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import (  # noqa: E402
    ARMConfig,
    PureARMClassifier,
    StandardTransformerClassifier,
    TransformerConfig,
)


TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[.,!?;:()\-]")
BABI_URL = "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz"


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


class BabiDataset(Dataset):
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


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def read_url_bytes(url: str) -> bytes:
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = ssl.create_default_context()
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 ARM-benchmark"})
    with urlopen(request, timeout=120, context=context) as response:
        return response.read()


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if not str(target).startswith(str(destination)):
            raise RuntimeError(f"Unsafe archive path blocked: {member.name}")
    tar.extractall(destination)


def ensure_babi_data(data_dir: Path) -> Path:
    extracted_root = data_dir / "tasks_1-20_v1-2"
    if extracted_root.exists():
        return extracted_root

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "en-valid-10k.tar.gz"
    if not archive.exists():
        print("Downloading Facebook bAbI archive...", flush=True)
        archive.write_bytes(read_url_bytes(BABI_URL))

    print("Extracting Facebook bAbI archive...", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        safe_extract(tar, data_dir)

    if extracted_root.exists():
        return extracted_root

    candidates = list(data_dir.rglob("tasks_1-20_v1-2"))
    if not candidates:
        raise RuntimeError(f"Could not find extracted bAbI root under {data_dir}")
    return candidates[0]


def find_task_files(root: Path, language_dir: str, task_id: int) -> tuple[Path, Path]:
    base = root / language_dir
    if not base.exists():
        raise RuntimeError(f"bAbI language directory not found: {base}")

    prefix = f"qa{task_id}_"
    train_files = sorted(base.glob(f"{prefix}*train.txt"))
    test_files = sorted(base.glob(f"{prefix}*test.txt"))
    if not train_files or not test_files:
        available = sorted(path.name for path in base.glob("qa*_train.txt"))[:10]
        raise RuntimeError(f"Could not find task {task_id}; available examples: {available}")
    return train_files[0], test_files[0]


def parse_babi_file(path: Path, task_id: int, limit: int) -> list[Dict[str, Any]]:
    examples: list[Dict[str, Any]] = []
    story: list[str] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue

            number_and_text = line.split(" ", 1)
            if len(number_and_text) != 2:
                continue

            sentence_number = int(number_and_text[0])
            text = number_and_text[1].strip().lower()
            if sentence_number == 1:
                story = []

            if "\t" in text:
                question, answer, supporting_facts = text.split("\t")
                support_count = len([fact for fact in supporting_facts.split(" ") if fact])
                examples.append(
                    {
                        "text": " ".join(story) + " [QUERY] " + question,
                        "label": answer,
                        "task_id": task_id,
                        "support_count": support_count,
                        "source_file": path.name,
                    }
                )
                if len(examples) >= limit:
                    break
            else:
                story.append(text)

    return examples


def split_train_val(
    examples: Sequence[Dict[str, Any]],
    val_fraction: float,
    seed: int,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int((1.0 - val_fraction) * len(shuffled)))
    return shuffled[:cut], shuffled[cut:]


def load_babi_examples(args: argparse.Namespace) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    root = ensure_babi_data(Path(args.babi_data_dir))
    train_all: list[Dict[str, Any]] = []
    test_all: list[Dict[str, Any]] = []

    for task_id in args.babi_tasks:
        train_file, test_file = find_task_files(root, args.babi_language_dir, task_id)
        train_all.extend(parse_babi_file(train_file, task_id, args.max_train_per_task))
        test_all.extend(parse_babi_file(test_file, task_id, args.max_test_per_task))

    train, val = split_train_val(train_all, args.val_fraction, args.seed)
    print(
        f"bAbI tasks={args.babi_tasks} splits: "
        f"train={len(train)}, validation={len(val)}, test={len(test_all)}",
        flush=True,
    )
    return train, val, test_all


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
        config = TransformerConfig(
            vocab_size=vocab_size,
            num_classes=num_classes,
            dim=args.resolved_transformer_dim,
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
) -> tuple[int, int, int, int, int]:
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for dim in range(args.match_min_dim, args.match_max_dim + 1, args.match_dim_step):
        valid_heads = [head for head in args.match_heads if head > 0 and dim % head == 0]
        for heads in valid_heads:
            for layers in range(args.match_min_layers, args.match_max_layers + 1):
                for ff_dim in range(args.match_min_ff_dim, args.match_max_ff_dim + 1, args.match_ff_dim_step):
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
        raise ValueError("no valid Transformer candidates for parameter matching")

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
def evaluate_loader(
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


@torch.no_grad()
def evaluate_examples(
    model: torch.nn.Module,
    examples: Sequence[Dict[str, Any]],
    tokenizer: Tokenizer,
    label_to_id: Dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    correct = 0
    by_task: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    by_support_count: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    samples = []

    for example in examples:
        input_ids, attention_mask = tokenizer.encode(example["text"], args.max_len)
        logits = model(input_ids.unsqueeze(0).to(device), attention_mask.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=-1)
        predicted = id_to_label[int(probs.argmax(dim=-1).item())]
        ok = predicted == example["label"]
        correct += int(ok)

        task_key = str(example["task_id"])
        support_key = str(example["support_count"])
        by_task[task_key]["correct"] += int(ok)
        by_task[task_key]["total"] += 1
        by_support_count[support_key]["correct"] += int(ok)
        by_support_count[support_key]["total"] += 1

        if len(samples) < args.num_samples:
            samples.append(
                {
                    "text": example["text"],
                    "true": example["label"],
                    "predicted": predicted,
                    "confidence": round(float(probs.max().item()), 4),
                    "task_id": example["task_id"],
                    "support_count": example["support_count"],
                    "correct": ok,
                }
            )

    def summarize(groups: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float | int]]:
        return {
            group: {"acc": values["correct"] / max(1, values["total"]), "total": values["total"]}
            for group, values in sorted(groups.items(), key=lambda item: int(item[0]))
        }

    return {
        "acc": correct / max(1, len(examples)),
        "total": len(examples),
        "by_task": summarize(by_task),
        "by_support_count": summarize(by_support_count),
        "samples": samples,
    }


def train_one(
    kind: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, object, Dict[str, Any]]:
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
        val = evaluate_loader(model, kind, val_loader, args, device)
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

    return model, config, {
        "parameters": count_parameters(model),
        "best_val_acc": best_val_acc,
        "final_val": evaluate_loader(model, kind, val_loader, args, device),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--babi-data-dir", type=str, default="data_babi")
    parser.add_argument("--babi-language-dir", type=str, default="en")
    parser.add_argument("--babi-tasks", type=parse_csv_ints, default=parse_csv_ints("16"))
    parser.add_argument("--max-train-per-task", type=int, default=10000)
    parser.add_argument("--max-test-per-task", type=int, default=1000)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=25)
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
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="benchmark_results/pure_models_babi")
    args = parser.parse_args()
    args.resolved_transformer_dim = args.transformer_dim or args.dim
    args.resolved_transformer_heads = args.transformer_heads
    args.resolved_transformer_layers = args.transformer_layers
    args.resolved_transformer_ff_dim = args.transformer_ff_dim

    set_threads()
    seed_all(args.seed)
    device = torch.device(args.device)
    print("Device:", device, flush=True)

    train_examples, val_examples, test_examples = load_babi_examples(args)
    labels = sorted({example["label"] for example in train_examples + val_examples + test_examples})
    label_to_id = {label: index for index, label in enumerate(labels)}
    print("Labels:", labels, flush=True)

    tokenizer = Tokenizer(args.max_vocab)
    tokenizer.fit([example["text"] for example in train_examples])

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

    train_ds = BabiDataset(train_examples, tokenizer, label_to_id, args.max_len)
    val_ds = BabiDataset(val_examples, tokenizer, label_to_id, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "benchmark": "facebook_babi",
        "source_url": BABI_URL,
        "tasks": args.babi_tasks,
        "language_dir": args.babi_language_dir,
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
        "models": {},
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
        test = evaluate_examples(model, test_examples, tokenizer, label_to_id, args, device)
        result["test"] = test

        checkpoint_path = out_dir / f"{kind}_babi.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config,
                "tokenizer_stoi": tokenizer.stoi,
                "tokenizer_itos": tokenizer.itos,
                "label_to_id": label_to_id,
                "labels": labels,
                "args": vars(args),
                "result": result,
            },
            checkpoint_path,
        )
        summary["models"][kind] = {**result, "checkpoint": str(checkpoint_path)}
        print(f"{kind} held-out test acc={test['acc']:.4f}", flush=True)

    metrics_path = out_dir / "pure_models_babi.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nFinal bAbI comparison:")
    print(
        json.dumps(
            {
                kind: {
                    "parameters": result["parameters"],
                    "best_val_acc": result["best_val_acc"],
                    "test_acc": result["test"]["acc"],
                    "test_by_task": result["test"]["by_task"],
                    "test_by_support_count": result["test"]["by_support_count"],
                }
                for kind, result in summary["models"].items()
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Saved metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
