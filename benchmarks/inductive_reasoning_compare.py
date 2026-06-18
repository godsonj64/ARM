"""Inductive reasoning benchmarks for ARM and Transformer baselines.

This script runs:
1. CLUTRR train/validation/test with held-out test accuracy grouped by task
   chain length, e.g. task_1.4, task_1.5, ...
2. Auto-downloaded Facebook bAbI QA tasks grouped by supporting-fact count.

Run:
    python benchmarks/inductive_reasoning_compare.py --epochs 6 --models arm,transformer_small

Outputs:
    benchmark_results/inductive_reasoning_compare.json
    benchmark_results/inductive_reasoning_curves.png
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tarfile
import ssl
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from urllib.request import Request, urlopen

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from clutrr_compare import load_clutrr, pick_column, seed_all, set_threads, textify
from clutrr_multi_transformer_compare import TRANSFORMER_SPECS, TransformerClassifier
from arm import AlgebraicResonanceMemory, GRUTextEncoder, MemoryClassifier

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[.,!?;:()\-]")


class Tokenizer:
    def __init__(self, max_vocab: int):
        self.max_vocab = max_vocab
        self.stoi = {"[PAD]": 0, "[UNK]": 1}
        self.itos = ["[PAD]", "[UNK]"]

    def tokenize(self, text: str) -> List[str]:
        return TOKEN_RE.findall(text.lower())

    def fit(self, texts: Sequence[str]) -> None:
        counts = Counter()
        for text in texts:
            counts.update(self.tokenize(text))
        for token, _ in counts.most_common(self.max_vocab - len(self.itos)):
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
        print("Vocabulary size:", len(self.itos))

    def encode(self, text: str, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = [self.stoi.get(tok, 1) for tok in self.tokenize(text)[:max_len]]
        mask = [1.0] * len(ids)
        ids += [0] * (max_len - len(ids))
        mask += [0.0] * (max_len - len(mask))
        return torch.tensor(ids).long(), torch.tensor(mask).float()


class TextExampleDataset(Dataset):
    def __init__(self, examples: Sequence[Dict[str, Any]], tokenizer: Tokenizer, label_to_id: Dict[str, int], max_len: int):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        ids, mask = self.tokenizer.encode(ex["text"], self.max_len)
        return {"input_ids": ids, "attention_mask": mask, "label": torch.tensor(self.label_to_id[ex["label"]]).long()}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_model(name: str, vocab: int, labels: int, args: argparse.Namespace) -> nn.Module:
    if name == "arm":
        encoder = GRUTextEncoder(vocab, args.emb_dim, args.hidden_dim, args.latent_dim, dropout=args.dropout)
        memory = AlgebraicResonanceMemory(args.latent_dim, labels, args.num_operators, tau=args.tau)
        return MemoryClassifier(encoder, memory)
    if name.startswith("transformer_"):
        spec_name = name.removeprefix("transformer_")
        if spec_name not in TRANSFORMER_SPECS:
            raise ValueError(f"Unknown Transformer spec {spec_name!r}; choose from {sorted(TRANSFORMER_SPECS)}")
        return TransformerClassifier(vocab, labels, args.max_len, TRANSFORMER_SPECS[spec_name], args.dropout)
    raise ValueError(name)


def batch_loss(model: nn.Module, batch: Dict[str, torch.Tensor], name: str, args: argparse.Namespace, device: torch.device):
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    y = batch["label"].to(device)
    out = model(ids, mask)
    loss = F.cross_entropy(out["logits"], y)
    if name == "arm":
        loss = loss + args.cycle_weight * model.memory_layer.cycle_consistency_loss(4)
        loss = loss + args.op_weight * model.memory_layer.operator_regularization()
    with torch.no_grad():
        acc = (out["logits"].argmax(dim=-1) == y).float().mean().item()
    return loss, acc


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, name: str, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    seen = 0
    for batch in loader:
        loss, acc = batch_loss(model, batch, name, args, device)
        batch_size = batch["label"].shape[0]
        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        seen += batch_size
    return {"loss": total_loss / max(1, seen), "acc": total_acc / max(1, seen)}


@torch.no_grad()
def evaluate_examples(
    model: nn.Module,
    examples: Sequence[Dict[str, Any]],
    tokenizer: Tokenizer,
    label_to_id: Dict[str, int],
    name: str,
    args: argparse.Namespace,
    device: torch.device,
    group_key: str,
) -> Dict[str, Any]:
    model.eval()
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    groups: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    correct = 0
    samples = []
    for idx, ex in enumerate(examples):
        ids, mask = tokenizer.encode(ex["text"], args.max_len)
        out = model(ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))
        probs = F.softmax(out["logits"], dim=-1)
        pred = id_to_label[int(probs.argmax(dim=-1).item())]
        ok = pred == ex["label"]
        correct += int(ok)
        group = str(ex.get(group_key, "unknown"))
        groups[group]["correct"] += int(ok)
        groups[group]["total"] += 1
        if len(samples) < args.num_samples:
            samples.append(
                {
                    "text": ex["text"],
                    "true": ex["label"],
                    "predicted": pred,
                    "confidence": round(float(probs.max().item()), 4),
                    "group": group,
                    "correct": ok,
                }
            )
    by_group = {
        group: {"acc": vals["correct"] / max(1, vals["total"]), "total": vals["total"]}
        for group, vals in sorted(groups.items(), key=lambda item: item[0])
    }
    return {"acc": correct / max(1, len(examples)), "total": len(examples), "by_group": by_group, "samples": samples}


def train_one(
    name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab: int,
    labels: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    model = make_model(name, vocab, labels, args).to(device)
    params = count_parameters(model)
    print(f"{name} trainable parameters: {params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        seen = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, acc = batch_loss(model, batch, name, args, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            batch_size = batch["label"].shape[0]
            total_loss += loss.item() * batch_size
            total_acc += acc * batch_size
            seen += batch_size
        sched.step()
        val = evaluate(model, val_loader, name, args, device)
        train_loss = total_loss / max(1, seen)
        train_acc = total_acc / max(1, seen)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        best_val = max(best_val, val["acc"])
        print(f"{name:18s} epoch {epoch:03d}/{args.epochs} | train_acc={train_acc:.4f} | val_acc={val['acc']:.4f}")
    return model, {"parameters": params, "best_val_acc": best_val, "final_val": evaluate(model, val_loader, name, args, device), "history": history}


def clutrr_examples_from_split(split, limit: int) -> List[Dict[str, Any]]:
    columns = list(split[0].keys())
    story_col = pick_column(columns, ["story", "clean_story", "text", "context", "sentence"])
    query_col = pick_column(columns, ["query", "question"], required=False)
    label_col = pick_column(columns, ["target_text", "target", "answer", "relation", "target_label", "label"])
    task_col = pick_column(columns, ["task_name"], required=False)
    examples = []
    for idx in range(min(len(split), limit)):
        row = split[idx]
        story = textify(row[story_col])
        query = textify(row[query_col]) if query_col else ""
        label = textify(row[label_col])
        task_name = textify(row[task_col]) if task_col else "unknown"
        match = re.search(r"\.(\d+)$", task_name)
        chain_length = int(match.group(1)) if match else -1
        examples.append(
            {
                "text": (story + " [QUERY] " + query).strip(),
                "label": label,
                "task_name": task_name,
                "chain_length": chain_length,
            }
        )
    return examples


def load_clutrr_examples(args: argparse.Namespace):
    raw, used_config = load_clutrr(args.dataset_name, args.preferred_config)
    train = clutrr_examples_from_split(raw["train"], args.max_train)
    val = clutrr_examples_from_split(raw["validation"], args.max_eval)
    test = clutrr_examples_from_split(raw["test"], args.max_test)
    print(f"CLUTRR splits: train={len(train)}, validation={len(val)}, test={len(test)}")
    return used_config, train, val, test


RELATIONS = {
    "r0": lambda x, n: (x + 1) % n,
    "r1": lambda x, n: (x + 2) % n,
    "r2": lambda x, n: (2 * x + 1) % n,
    "r3": lambda x, n: (3 * x + 2) % n,
}


BABI_URL = "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz"


def read_url_bytes(url: str) -> bytes:
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = ssl.create_default_context()
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 ARM-benchmark-autodownload"})
    with urlopen(request, timeout=120, context=context) as response:
        return response.read()


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if not str(target).startswith(str(destination)):
            raise RuntimeError(f"Unsafe archive path blocked: {member.name}")
    tar.extractall(destination)


def download_babi(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted_root = data_dir / "tasks_1-20_v1-2"
    if extracted_root.exists():
        return extracted_root
    archive = data_dir / "en-valid-10k.tar.gz"
    if not archive.exists():
        print("Downloading official Facebook bAbI archive...")
        archive.write_bytes(read_url_bytes(BABI_URL))
    print("Extracting bAbI archive safely...")
    with tarfile.open(archive, "r:gz") as tar:
        safe_extract(tar, data_dir)
    if extracted_root.exists():
        return extracted_root
    candidates = list(data_dir.rglob("tasks_1-20_v1-2"))
    if not candidates:
        raise RuntimeError("Could not find extracted bAbI tasks_1-20_v1-2 directory.")
    return candidates[0]


def find_babi_task_files(base: Path, task_id: int) -> Tuple[Path, Path]:
    prefix = f"qa{task_id}_"
    train_files = sorted(base.rglob(f"{prefix}*train.txt"))
    test_files = sorted(base.rglob(f"{prefix}*test.txt"))
    if not train_files or not test_files:
        available = sorted(p.name for p in base.rglob("qa*_train.txt"))[:10]
        raise RuntimeError(f"Could not find bAbI task {task_id}. Available examples: {available}")
    return train_files[0], test_files[0]


def parse_babi_file(path: Path, limit: int, task_id: int) -> List[Dict[str, Any]]:
    examples = []
    story = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            idx_text = line.split(" ", 1)
            if len(idx_text) != 2:
                continue
            idx = int(idx_text[0])
            text = idx_text[1].strip().lower()
            if idx == 1:
                story = []
            if "\t" in text:
                question, answer, support = text.split("\t")
                context = " ".join(story)
                support_count = len([s for s in support.split(" ") if s])
                examples.append(
                    {
                        "text": context + " [QUERY] " + question,
                        "label": answer,
                        "task_id": task_id,
                        "chain_length": support_count,
                    }
                )
                if len(examples) >= limit:
                    break
            else:
                story.append(text)
    return examples


def split_train_val(examples: List[Dict[str, Any]], val_fraction: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = list(examples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    cut = max(1, int((1.0 - val_fraction) * len(shuffled)))
    return shuffled[:cut], shuffled[cut:]


def load_babi_examples(args: argparse.Namespace):
    base = download_babi(Path(args.babi_data_dir))
    train_all: List[Dict[str, Any]] = []
    test_all: List[Dict[str, Any]] = []
    for task_id in args.babi_tasks:
        train_file, test_file = find_babi_task_files(base, task_id)
        train_all.extend(parse_babi_file(train_file, args.max_train, task_id))
        test_all.extend(parse_babi_file(test_file, args.max_test, task_id))
    train, val = split_train_val(train_all, args.babi_val_fraction, args.seed)
    print(f"bAbI tasks={args.babi_tasks} splits: train={len(train)}, validation={len(val)}, test={len(test_all)}")
    return train, val, test_all


def make_chain_example(length: int, num_states: int, rng: random.Random) -> Dict[str, Any]:
    start = rng.randrange(num_states)
    state = start
    relations = [rng.choice(list(RELATIONS)) for _ in range(length)]
    for relation in relations:
        state = RELATIONS[relation](state, num_states)
    text = f"start state s{start} . " + " ".join(f"apply {relation} ." for relation in relations) + " [QUERY] final state ?"
    return {"text": text, "label": f"s{state}", "chain_length": length}


def make_synthetic_chain_examples(num_examples: int, lengths: Sequence[int], num_states: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    return [make_chain_example(rng.choice(list(lengths)), num_states, rng) for _ in range(num_examples)]


def parse_models(value: str) -> List[str]:
    names = [v.strip() for v in value.split(",") if v.strip()]
    valid = {"arm"} | {f"transformer_{name}" for name in TRANSFORMER_SPECS}
    unknown = [name for name in names if name not in valid]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown model(s): {unknown}. Choose from {sorted(valid)}")
    return names


def parse_lengths(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def run_text_benchmark(
    benchmark_name: str,
    train_examples: List[Dict[str, Any]],
    val_examples: List[Dict[str, Any]],
    test_examples: List[Dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    labels = sorted({ex["label"] for ex in train_examples + val_examples + test_examples})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    tokenizer = Tokenizer(args.max_vocab)
    tokenizer.fit([ex["text"] for ex in train_examples])
    train_ds = TextExampleDataset(train_examples, tokenizer, label_to_id, args.max_len)
    val_ds = TextExampleDataset(val_examples, tokenizer, label_to_id, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    results = {"labels": labels, "models": {}}
    for model_name in args.models:
        print(f"\n[{benchmark_name}] training {model_name}")
        seed_all(args.seed)
        model, metrics = train_one(model_name, train_loader, val_loader, len(tokenizer.itos), len(labels), args, device)
        test = evaluate_examples(model, test_examples, tokenizer, label_to_id, model_name, args, device, "chain_length")
        metrics["test"] = test
        results["models"][model_name] = metrics
        print(f"[{benchmark_name}] {model_name} held-out test acc={test['acc']:.4f}")
        print(f"[{benchmark_name}] {model_name} test by chain length:", json.dumps(test["by_group"], indent=2))
    return results


def plot_results(all_results: Dict[str, Any], out_dir: Path) -> None:
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for benchmark_name, result in all_results.items():
        if "models" not in result:
            continue
        plt.figure(figsize=(9, 5))
        for model_name, metrics in result["models"].items():
            plt.plot(metrics["history"]["val_acc"], label=f"{model_name} val acc")
        plt.title(f"{benchmark_name}: validation curves")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.savefig(out_dir / f"{benchmark_name}_curves.png", dpi=160, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=str, default="clutrr,babi")
    parser.add_argument("--models", type=parse_models, default=parse_models("arm,transformer_small"))
    parser.add_argument("--dataset-name", type=str, default="CLUTRR/v1")
    parser.add_argument("--preferred-config", type=str, default="gen_train234_test2to10")
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--max-eval", type=int, default=3000)
    parser.add_argument("--max-test", type=int, default=3000)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-operators", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.45)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cycle-weight", type=float, default=0.003)
    parser.add_argument("--op-weight", type=float, default=0.001)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--synthetic-train-samples", type=int, default=12000)
    parser.add_argument("--synthetic-val-samples", type=int, default=3000)
    parser.add_argument("--synthetic-test-samples", type=int, default=3000)
    parser.add_argument("--synthetic-states", type=int, default=18)
    parser.add_argument("--synthetic-train-lengths", type=parse_lengths, default=parse_lengths("2,3"))
    parser.add_argument("--synthetic-test-lengths", type=parse_lengths, default=parse_lengths("4,5,6,7,8"))
    parser.add_argument("--babi-data-dir", type=str, default="data_babi")
    parser.add_argument("--babi-tasks", type=parse_lengths, default=parse_lengths("2,3"))
    parser.add_argument("--babi-val-fraction", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    set_threads()
    seed_all(args.seed)
    device = torch.device(args.device)
    selected = {name.strip() for name in args.benchmarks.split(",") if name.strip()}
    print("Device:", device)
    print("Benchmarks:", ", ".join(sorted(selected)))
    print("Models:", ", ".join(args.models))

    all_results: Dict[str, Any] = {}
    if "clutrr" in selected:
        used_config, train, val, test = load_clutrr_examples(args)
        all_results["clutrr_test"] = run_text_benchmark("clutrr_test", train, val, test, args, device)
        all_results["clutrr_test"]["dataset_config"] = used_config
    if "synthetic_chain" in selected:
        train = make_synthetic_chain_examples(args.synthetic_train_samples, args.synthetic_train_lengths, args.synthetic_states, args.seed + 100)
        val = make_synthetic_chain_examples(args.synthetic_val_samples, args.synthetic_train_lengths, args.synthetic_states, args.seed + 200)
        test = make_synthetic_chain_examples(args.synthetic_test_samples, args.synthetic_test_lengths, args.synthetic_states, args.seed + 300)
        all_results["synthetic_chain"] = run_text_benchmark("synthetic_chain", train, val, test, args, device)
        all_results["synthetic_chain"]["train_lengths"] = args.synthetic_train_lengths
        all_results["synthetic_chain"]["test_lengths"] = args.synthetic_test_lengths
    if "babi" in selected:
        train, val, test = load_babi_examples(args)
        all_results["babi"] = run_text_benchmark("babi", train, val, test, args, device)
        all_results["babi"]["tasks"] = args.babi_tasks
        all_results["babi"]["source_url"] = BABI_URL

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "inductive_reasoning_compare.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    plot_results(all_results, out_dir)

    print("\nFinal inductive reasoning summary:")
    summary = {
        bench: {
            model: {
                "parameters": metrics["parameters"],
                "best_val_acc": metrics["best_val_acc"],
                "test_acc": metrics["test"]["acc"],
                "test_by_chain_length": metrics["test"]["by_group"],
            }
            for model, metrics in result["models"].items()
        }
        for bench, result in all_results.items()
        if "models" in result
    }
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {out_file}")


if __name__ == "__main__":
    main()
