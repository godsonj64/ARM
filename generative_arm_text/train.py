from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative_arm_text.data import ByteTokenizer, NextTokenDataset, load_downloaded_dataset, load_text, preprocess_text
from generative_arm_text.model import GenerativeARMConfig, GenerativeARMLanguageModel


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def split_ids(ids: list[int], val_fraction: float) -> tuple[list[int], list[int]]:
    cut = max(1, int(len(ids) * (1.0 - val_fraction)))
    return ids[:cut], ids[max(0, cut - 1) :]


@torch.no_grad()
def evaluate(model: GenerativeARMLanguageModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    seen = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss = model(input_ids, labels)["loss"]
        batch_size = input_ids.shape[0]
        total_loss += float(loss.item()) * batch_size
        seen += batch_size
    return total_loss / max(1, seen)


def save_checkpoint(path: Path, model: GenerativeARMLanguageModel, args: argparse.Namespace, step: int) -> None:
    payload = {
        "config": model.config.to_dict(),
        "model": model.state_dict(),
        "args": vars(args),
        "step": step,
    }
    torch.save(payload, path)


def prepare_corpus(args: argparse.Namespace) -> tuple[str, str]:
    if args.data_path:
        raw_text = load_text(args.data_path, args.text_field)
        source = str(Path(args.data_path).expanduser())
    else:
        raw_text = load_downloaded_dataset(args.dataset, args.cache_dir)
        source = args.dataset
    return preprocess_text(raw_text, args.max_chars), source


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a standalone Generative ARM text model.")
    parser.add_argument("--dataset", choices=["tiny_shakespeare", "wikitext"], default="tiny_shakespeare", help="Real corpus to auto-download when --data-path is not set.")
    parser.add_argument("--data-path", type=str, default="", help="Text, markdown, JSONL, CSV, or directory of those files.")
    parser.add_argument("--text-field", type=str, default="text", help="JSONL/CSV field to read when present.")
    parser.add_argument("--cache-dir", type=str, default=str(HERE / "downloaded_data"), help="Dataset download/cache directory.")
    parser.add_argument("--max-chars", type=int, default=0, help="Optionally truncate preprocessed corpus for quick experiments.")
    parser.add_argument("--out-dir", type=str, default=str(HERE / "runs" / "tiny_arm_lm"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--arm-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-memory-atoms", type=int, default=256)
    parser.add_argument("--num-operators", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--op-weight", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_all(args.seed)
    tokenizer = ByteTokenizer()
    text, source = prepare_corpus(args)
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    train_ids, val_ids = split_ids(ids, args.val_fraction)
    train_loader = DataLoader(NextTokenDataset(train_ids, args.block_size), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(NextTokenDataset(val_ids, args.block_size), batch_size=args.batch_size, shuffle=False)

    config = GenerativeARMConfig(
        vocab_size=tokenizer.vocab_size,
        max_len=args.block_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        arm_dim=args.arm_dim,
        num_layers=args.num_layers,
        num_memory_atoms=args.num_memory_atoms,
        num_operators=args.num_operators,
        tau=args.tau,
        dropout=args.dropout,
    )
    device = torch.device(args.device)
    model = GenerativeARMLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preprocessed_corpus_preview.txt").write_text(text[:5000], encoding="utf-8")

    print(f"source={source} chars={len(text):,} tokens={len(ids):,}")
    print(f"train_windows={len(train_loader.dataset):,} val_windows={len(val_loader.dataset):,}")
    print(f"parameters={count_parameters(model):,} device={device}")

    history = []
    best_val = float("inf")
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids, labels)
            loss = out["loss"] + args.op_weight * model.arm.regularization_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            step += 1
            total += float(out["loss"].item()) * input_ids.shape[0]
            seen += input_ids.shape[0]
        train_loss = total / max(1, seen)
        val_loss = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(out_dir / "best.pt", model, args, step)
        save_checkpoint(out_dir / "last.pt", model, args, step)

    (out_dir / "metrics.json").write_text(json.dumps({"best_val_loss": best_val, "history": history}, indent=2), encoding="utf-8")
    print(f"saved checkpoints and metrics to {out_dir}")


if __name__ == "__main__":
    main()
