"""Compare ARM against multiple Transformer encoder baselines on CLUTRR.

Run locally:
    pip install -r requirements.txt
    python benchmarks/clutrr_multi_transformer_compare.py --epochs 6

Outputs:
    benchmark_results/clutrr_multi_transformer_compare.json
    benchmark_results/clutrr_multi_transformer_learning_curves.png

The JSON includes metrics, parameter counts, and sample predictions from each
trained model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

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


@dataclass(frozen=True)
class TransformerSpec:
    dim: int
    heads: int
    layers: int
    ff_dim: int


TRANSFORMER_SPECS: Dict[str, TransformerSpec] = {
    "tiny": TransformerSpec(dim=64, heads=4, layers=1, ff_dim=128),
    "small": TransformerSpec(dim=128, heads=4, layers=2, ff_dim=512),
    "wide": TransformerSpec(dim=256, heads=8, layers=2, ff_dim=1024),
    "deep": TransformerSpec(dim=128, heads=4, layers=4, ff_dim=512),
}


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, labels: int, max_len: int, spec: TransformerSpec, dropout: float):
        super().__init__()
        if spec.dim % spec.heads != 0:
            raise ValueError(f"dim ({spec.dim}) must be divisible by heads ({spec.heads})")
        self.token_embedding = nn.Embedding(vocab_size, spec.dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, spec.dim)
        layer = nn.TransformerEncoderLayer(
            d_model=spec.dim,
            nhead=spec.heads,
            dim_feedforward=spec.ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=spec.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(spec.dim)
        self.classifier = nn.Linear(spec.dim, labels)
        self.scale = math.sqrt(spec.dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        x = self.token_embedding(input_ids) * self.scale + self.position_embedding(positions)
        encoded = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        mask = attention_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return {"logits": self.classifier(self.norm(pooled))}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_model(name: str, vocab: int, labels: int, args: argparse.Namespace) -> nn.Module:
    if name == "arm":
        return make_arm_model("arm", vocab, labels, args)
    if not name.startswith("transformer_"):
        raise ValueError(name)
    spec_name = name.removeprefix("transformer_")
    return TransformerClassifier(vocab, labels, args.max_len, TRANSFORMER_SPECS[spec_name], args.dropout)


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


def train_one(name: str, train_loader: DataLoader, val_loader: DataLoader, vocab: int, labels: int, args: argparse.Namespace, device: torch.device):
    model = make_model(name, vocab, labels, args).to(device)
    params = count_parameters(model)
    print(f"{name} trainable parameters: {params:,}")
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
        best = max(best, val["acc"])
        print(f"{name:18s} epoch {epoch:03d}/{args.epochs} | train_acc={train_acc:.4f} | val_acc={val['acc']:.4f}")
    metrics = {
        "parameters": params,
        "best_val_acc": best,
        "final": evaluate(model, val_loader, name, args, device),
        "history": history,
    }
    return model, metrics


@torch.no_grad()
def sample_predictions(
    models: Dict[str, nn.Module],
    rows: Sequence[tuple[str, str]],
    tokenizer: Tokenizer,
    label_to_id: Dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, object]]:
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    samples: List[Dict[str, object]] = []
    for idx, (text, label) in enumerate(rows[: args.num_samples]):
        input_ids, attention_mask = tokenizer.encode(text, args.max_len)
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        predictions = {}
        for name, model in models.items():
            model.eval()
            logits = model(input_ids, attention_mask)["logits"]
            probs = F.softmax(logits, dim=-1)
            pred_id = int(probs.argmax(dim=-1).item())
            predictions[name] = {
                "predicted": id_to_label[pred_id],
                "confidence": round(float(probs.max().item()), 4),
                "correct": id_to_label[pred_id] == label,
            }
        samples.append(
            {
                "sample": idx,
                "text": text,
                "true": label,
                "predictions": predictions,
            }
        )
    return samples


def print_sample_predictions(samples: Sequence[Dict[str, object]]) -> None:
    print("\nSample predictions:")
    for sample in samples:
        print(f"\nsample={sample['sample']:02d} | true={sample['true']}")
        print(f"text={sample['text'][:220]}")
        for name, pred in sample["predictions"].items():
            marker = "OK" if pred["correct"] else "MISS"
            print(f"  {name:18s} pred={pred['predicted']:<16s} conf={pred['confidence']:.4f} {marker}")


def plot_results(results: Dict[str, object], out_dir: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(9, 5))
    for name, result in results.items():
        plt.plot(result["history"]["val_acc"], label=f"{name} val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CLUTRR: ARM vs Transformer Variants")
    plt.grid(True)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "clutrr_multi_transformer_learning_curves.png", dpi=160, bbox_inches="tight")


def parse_transformers(value: str) -> List[str]:
    names = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [name for name in names if name not in TRANSFORMER_SPECS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown transformer spec(s): {unknown}. Choose from {sorted(TRANSFORMER_SPECS)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="CLUTRR/v1")
    parser.add_argument("--preferred-config", type=str, default="gen_train234_test2to10")
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--max-eval", type=int, default=3000)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-operators", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.45)
    parser.add_argument("--transformers", type=parse_transformers, default=parse_transformers("tiny,small,deep"))
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cycle-weight", type=float, default=0.003)
    parser.add_argument("--op-weight", type=float, default=0.001)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    set_threads()
    seed_all(args.seed)
    device = torch.device(args.device)
    print("Device:", device)
    print("Transformer specs:", ", ".join(args.transformers))

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

    model_names = ["arm"] + [f"transformer_{name}" for name in args.transformers]
    trained_models: Dict[str, nn.Module] = {}
    result_metrics: Dict[str, object] = {}
    for name in model_names:
        seed_all(args.seed)
        model, metrics = train_one(name, train_loader, val_loader, len(tokenizer.itos), len(labels), args, device)
        trained_models[name] = model
        result_metrics[name] = metrics

    samples = sample_predictions(trained_models, val_rows, tokenizer, label_to_id, args, device)
    print_sample_predictions(samples)

    transformer_config = {name: asdict(spec) for name, spec in TRANSFORMER_SPECS.items() if name in args.transformers}
    results = {
        "dataset_config": used_config,
        "labels": labels,
        "transformer_config": transformer_config,
        "metrics": result_metrics,
        "sample_predictions": samples,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "clutrr_multi_transformer_compare.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    plot_results(result_metrics, out_dir)

    print("\nFinal CLUTRR ARM vs Transformer variants summary:")
    print(
        json.dumps(
            {
                name: {
                    "parameters": metrics["parameters"],
                    "best_val_acc": metrics["best_val_acc"],
                    "final": metrics["final"],
                }
                for name, metrics in result_metrics.items()
            },
            indent=2,
        )
    )
    print(f"\nWrote: {out_dir / 'clutrr_multi_transformer_compare.json'}")
    if plt is not None:
        print(f"Wrote: {out_dir / 'clutrr_multi_transformer_learning_curves.png'}")


if __name__ == "__main__":
    main()
