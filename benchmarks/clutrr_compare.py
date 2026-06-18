"""Compare ARM against direct attention on CLUTRR kinship reasoning.

Run in Colab or locally:
    pip install -r requirements.txt
    python benchmarks/clutrr_compare.py --epochs 8

Outputs:
    benchmark_results/clutrr_compare.json
    benchmark_results/clutrr_learning_curves.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arm import AlgebraicResonanceMemory, AttentionMemory, GRUTextEncoder, MemoryClassifier

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from datasets import get_dataset_config_names, load_dataset
except Exception as exc:
    raise RuntimeError("Install Hugging Face datasets first: pip install datasets") from exc

TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[.,!?;:()\-]")


def set_threads() -> None:
    try:
        torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def textify(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(textify(v) for v in value)
    if isinstance(value, dict):
        return " ".join(f"{k} {textify(v)}" for k, v in value.items())
    return str(value)


def pick_column(columns: Sequence[str], names: Sequence[str], required: bool = True) -> str:
    lowered = {c.lower(): c for c in columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    for c in columns:
        for name in names:
            if name.lower() in c.lower():
                return c
    if required:
        raise KeyError(f"could not find any of {names} in columns {columns}")
    return ""


def load_clutrr(dataset_name: str, preferred_config: str):
    errors: List[str] = []
    configs: List[str] = []
    try:
        configs = get_dataset_config_names(dataset_name)
        print("Available CLUTRR configs:", configs[:12], "..." if len(configs) > 12 else "")
    except Exception as exc:
        errors.append(str(exc))
    candidates = [preferred_config] + [c for c in configs if c != preferred_config]
    if not candidates:
        candidates = [None]
    for config in candidates:
        try:
            ds = load_dataset(dataset_name) if config is None else load_dataset(dataset_name, config)
            print("Loaded dataset config:", config)
            return ds, config
        except Exception as exc:
            errors.append(f"{config}: {exc}")
    raise RuntimeError("Could not load CLUTRR. Errors:\n" + "\n".join(errors[-8:]))


def rows_from_split(split, story_col: str, query_col: str, label_col: str, limit: int) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for i in range(min(len(split), limit)):
        ex = split[i]
        story = textify(ex[story_col])
        query = textify(ex[query_col]) if query_col else ""
        label = textify(ex[label_col])
        rows.append(((story + " [QUERY] " + query).strip(), label))
    return rows


def prepare_rows(ds, max_train: int, max_eval: int) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    split_names = list(ds.keys())
    train_name = "train" if "train" in ds else split_names[0]
    eval_name = next((n for n in ["validation", "val", "test"] if n in ds and n != train_name), None)
    if eval_name is None:
        eval_name = next((n for n in split_names if n != train_name), None)

    columns = list(ds[train_name][0].keys())
    story_col = pick_column(columns, ["story", "clean_story", "text", "context", "sentence"])
    query_col = pick_column(columns, ["query", "question"], required=False)
    label_col = pick_column(columns, ["target_text", "target", "answer", "relation", "target_label", "label"])
    print("Column mapping:", {"story": story_col, "query": query_col or None, "label": label_col})

    train = rows_from_split(ds[train_name], story_col, query_col, label_col, max_train)
    if eval_name:
        val = rows_from_split(ds[eval_name], story_col, query_col, label_col, max_eval)
        print(f"Splits: {train_name}={len(train)}, {eval_name}={len(val)}")
    else:
        random.shuffle(train)
        cut = int(0.85 * len(train))
        train, val = train[:cut], train[cut:]
        print(f"Internal split: train={len(train)}, eval={len(val)}")
    return train, val


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
        for tok, _ in counts.most_common(self.max_vocab - 2):
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
        print("Vocabulary size:", len(self.itos))

    def encode(self, text: str, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = [self.stoi.get(t, 1) for t in self.tokenize(text)[:max_len]]
        mask = [1.0] * len(ids)
        ids += [0] * (max_len - len(ids))
        mask += [0.0] * (max_len - len(mask))
        return torch.tensor(ids).long(), torch.tensor(mask).float()


class TextRowsDataset(Dataset):
    def __init__(self, rows: Sequence[Tuple[str, str]], tokenizer: Tokenizer, label_to_id: Dict[str, int], max_len: int):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text, label = self.rows[idx]
        ids, mask = self.tokenizer.encode(text, self.max_len)
        return {"input_ids": ids, "attention_mask": mask, "label": torch.tensor(self.label_to_id[label]).long()}


def make_model(kind: str, vocab: int, labels: int, args: argparse.Namespace) -> MemoryClassifier:
    encoder = GRUTextEncoder(vocab, args.emb_dim, args.hidden_dim, args.latent_dim, dropout=args.dropout)
    if kind == "arm":
        memory = AlgebraicResonanceMemory(args.latent_dim, labels, args.num_operators, tau=args.tau)
    elif kind == "attention":
        memory = AttentionMemory(args.latent_dim, labels, dropout=args.dropout)
    else:
        raise ValueError(kind)
    return MemoryClassifier(encoder, memory)


def batch_loss(model: MemoryClassifier, batch: Dict[str, torch.Tensor], kind: str, args: argparse.Namespace, device: torch.device):
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    y = batch["label"].to(device)
    out = model(ids, mask)
    loss = F.cross_entropy(out["logits"], y)
    if kind == "arm":
        loss = loss + args.cycle_weight * model.memory_layer.cycle_consistency_loss(4) + args.op_weight * model.memory_layer.operator_regularization()
    with torch.no_grad():
        acc = (out["logits"].argmax(-1) == y).float().mean().item()
    return loss, acc


@torch.no_grad()
def evaluate(model: MemoryClassifier, loader: DataLoader, kind: str, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    seen = 0
    for batch in loader:
        loss, acc = batch_loss(model, batch, kind, args, device)
        bs = batch["label"].shape[0]
        total_loss += loss.item() * bs
        total_acc += acc * bs
        seen += bs
    return {"loss": total_loss / max(1, seen), "acc": total_acc / max(1, seen)}


def train_one(kind: str, train_loader: DataLoader, val_loader: DataLoader, vocab: int, labels: int, args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    model = make_model(kind, vocab, labels, args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        seen = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, acc = batch_loss(model, batch, kind, args, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            bs = batch["label"].shape[0]
            total_loss += loss.item() * bs
            total_acc += acc * bs
            seen += bs
        sched.step()
        val = evaluate(model, val_loader, kind, args, device)
        train_loss = total_loss / max(1, seen)
        train_acc = total_acc / max(1, seen)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        best = max(best, val["acc"])
        print(f"{kind:9s} epoch {epoch:03d}/{args.epochs} | train_acc={train_acc:.4f} | val_acc={val['acc']:.4f}")
    return {"best_val_acc": best, "final": evaluate(model, val_loader, kind, args, device), "history": history}


def plot_results(results: Dict[str, object], out_dir: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    for name, result in results.items():
        plt.plot(result["history"]["val_acc"], label=f"{name} val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CLUTRR benchmark: ARM vs direct attention")
    plt.grid(True)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "clutrr_learning_curves.png", dpi=160, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="clutrr")
    parser.add_argument("--preferred-config", type=str, default="gen_train234_test2to10")
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--max-eval", type=int, default=3000)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=8)
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    set_threads()
    seed_all(args.seed)
    device = torch.device(args.device)
    print("Device:", device)

    raw, used_config = load_clutrr(args.dataset_name, args.preferred_config)
    train_rows, val_rows = prepare_rows(raw, args.max_train, args.max_eval)
    labels = sorted({label for _, label in train_rows + val_rows})
    label_to_id = {label: i for i, label in enumerate(labels)}
    print("Labels:", labels)

    tokenizer = Tokenizer(args.max_vocab)
    tokenizer.fit([text for text, _ in train_rows])
    train_ds = TextRowsDataset(train_rows, tokenizer, label_to_id, args.max_len)
    val_ds = TextRowsDataset(val_rows, tokenizer, label_to_id, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    results: Dict[str, object] = {"dataset_config": used_config, "labels": labels}
    for kind in ["attention", "arm"]:
        seed_all(args.seed)
        results[kind] = train_one(kind, train_loader, val_loader, len(tokenizer.itos), len(labels), args, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "clutrr_compare.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    plot_results({k: v for k, v in results.items() if k in {"attention", "arm"}}, out_dir)
    print("\nFinal CLUTRR benchmark summary:")
    print(json.dumps({k: {"best_val_acc": v["best_val_acc"], "final": v["final"]} for k, v in results.items() if k in {"attention", "arm"}}, indent=2))


if __name__ == "__main__":
    main()
