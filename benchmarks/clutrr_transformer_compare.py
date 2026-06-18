"""Compare ARM against a Transformer encoder on CLUTRR kinship reasoning.

Run locally:
    pip install -r requirements.txt
    python benchmarks/clutrr_transformer_compare.py --epochs 8

Outputs:
    benchmark_results/clutrr_transformer_compare.json
    benchmark_results/clutrr_transformer_learning_curves.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from clutrr_compare import (
    TextRowsDataset,
    Tokenizer,
    load_clutrr,
    make_model as make_arm_model,
    prepare_rows,
    seed_all,
    set_threads,
)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        labels: int,
        max_len: int,
        dim: int,
        heads: int,
        layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"transformer_dim ({dim}) must be divisible by transformer_heads ({heads})")
        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, labels)
        self.scale = math.sqrt(dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        x = self.token_embedding(input_ids) * self.scale + self.position_embedding(positions)
        padding_mask = attention_mask == 0
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        mask = attention_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(self.norm(pooled))
        return {"logits": logits}


def make_model(kind: str, vocab: int, labels: int, args: argparse.Namespace) -> nn.Module:
    if kind == "arm":
        return make_arm_model("arm", vocab, labels, args)
    if kind == "transformer":
        return TransformerClassifier(
            vocab,
            labels,
            args.max_len,
            args.transformer_dim,
            args.transformer_heads,
            args.transformer_layers,
            args.transformer_ff_dim,
            args.dropout,
        )
    raise ValueError(kind)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batch_loss(model: nn.Module, batch: Dict[str, torch.Tensor], kind: str, args: argparse.Namespace, device: torch.device):
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    y = batch["label"].to(device)
    out = model(ids, mask)
    loss = F.cross_entropy(out["logits"], y)
    if kind == "arm":
        loss = loss + args.cycle_weight * model.memory_layer.cycle_consistency_loss(4)
        loss = loss + args.op_weight * model.memory_layer.operator_regularization()
    with torch.no_grad():
        acc = (out["logits"].argmax(dim=-1) == y).float().mean().item()
    return loss, acc


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, kind: str, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    seen = 0
    for batch in loader:
        loss, acc = batch_loss(model, batch, kind, args, device)
        batch_size = batch["label"].shape[0]
        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        seen += batch_size
    return {"loss": total_loss / max(1, seen), "acc": total_acc / max(1, seen)}


def train_one(kind: str, train_loader: DataLoader, val_loader: DataLoader, vocab: int, labels: int, args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    model = make_model(kind, vocab, labels, args).to(device)
    print(f"{kind} trainable parameters: {count_parameters(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best = -1.0
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
            batch_size = batch["label"].shape[0]
            total_loss += loss.item() * batch_size
            total_acc += acc * batch_size
            seen += batch_size
        sched.step()
        val = evaluate(model, val_loader, kind, args, device)
        train_loss = total_loss / max(1, seen)
        train_acc = total_acc / max(1, seen)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        best = max(best, val["acc"])
        print(f"{kind:11s} epoch {epoch:03d}/{args.epochs} | train_acc={train_acc:.4f} | val_acc={val['acc']:.4f}")
    return {
        "parameters": count_parameters(model),
        "best_val_acc": best,
        "final": evaluate(model, val_loader, kind, args, device),
        "history": history,
    }


def plot_results(results: Dict[str, object], out_dir: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    for name, result in results.items():
        plt.plot(result["history"]["val_acc"], label=f"{name} val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CLUTRR benchmark: ARM vs Transformer")
    plt.grid(True)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "clutrr_transformer_learning_curves.png", dpi=160, bbox_inches="tight")


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
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-operators", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.45)
    parser.add_argument("--transformer-dim", type=int, default=128)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ff-dim", type=int, default=512)
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
    for kind in ["arm", "transformer"]:
        seed_all(args.seed)
        results[kind] = train_one(kind, train_loader, val_loader, len(tokenizer.itos), len(labels), args, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "clutrr_transformer_compare.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    plot_results({k: v for k, v in results.items() if k in {"arm", "transformer"}}, out_dir)
    print("\nFinal CLUTRR ARM vs Transformer summary:")
    print(json.dumps({k: {"parameters": v["parameters"], "best_val_acc": v["best_val_acc"], "final": v["final"]} for k, v in results.items() if k in {"arm", "transformer"}}, indent=2))


if __name__ == "__main__":
    main()
