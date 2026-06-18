"""Compare single-hop ARM, multi-hop ARM, and Transformer baselines.

Multi-hop ARM is implemented separately in ``arm/multihop.py``. This benchmark
reuses the auto-downloaded CLUTRR and bAbI loaders from
``inductive_reasoning_compare.py``.

Run:
    python benchmarks/multihop_arm_compare.py --benchmarks clutrr --epochs 6
    python benchmarks/multihop_arm_compare.py --benchmarks babi --epochs 25

Outputs:
    benchmark_results/multihop_arm_compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from clutrr_multi_transformer_compare import TRANSFORMER_SPECS, TransformerClassifier
from inductive_reasoning_compare import (
    load_babi_examples,
    load_clutrr_examples,
    parse_lengths,
    run_text_benchmark,
    seed_all,
    set_threads,
)
from arm import AlgebraicResonanceMemory, GRUTextEncoder, MemoryClassifier, MultiHopAlgebraicResonanceMemory


def parse_models(value: str):
    names = [v.strip() for v in value.split(",") if v.strip()]
    valid = {"arm", "multihop_arm"} | {f"transformer_{name}" for name in TRANSFORMER_SPECS}
    unknown = [name for name in names if name not in valid]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown model(s): {unknown}. Choose from {sorted(valid)}")
    return names


def make_multihop_model(name: str, vocab: int, labels: int, args: argparse.Namespace) -> nn.Module:
    if name == "arm":
        encoder = GRUTextEncoder(vocab, args.emb_dim, args.hidden_dim, args.latent_dim, dropout=args.dropout)
        memory = AlgebraicResonanceMemory(args.latent_dim, labels, args.num_operators, tau=args.tau)
        return MemoryClassifier(encoder, memory)
    if name == "multihop_arm":
        encoder = GRUTextEncoder(vocab, args.emb_dim, args.hidden_dim, args.latent_dim, dropout=args.dropout)
        memory = MultiHopAlgebraicResonanceMemory(
            args.latent_dim,
            labels,
            num_operators=args.num_operators,
            max_hops=args.max_hops,
            beam_width=args.beam_width,
            tau=args.tau,
            score_intermediate=not args.final_hop_only,
            dropout=args.dropout,
        )
        return MemoryClassifier(encoder, memory)
    if name.startswith("transformer_"):
        spec_name = name.removeprefix("transformer_")
        if spec_name not in TRANSFORMER_SPECS:
            raise ValueError(f"Unknown Transformer spec {spec_name!r}; choose from {sorted(TRANSFORMER_SPECS)}")
        return TransformerClassifier(vocab, labels, args.max_len, TRANSFORMER_SPECS[spec_name], args.dropout)
    raise ValueError(name)


def patch_inductive_make_model() -> None:
    import inductive_reasoning_compare

    inductive_reasoning_compare.make_model = make_multihop_model


def run_benchmark(name: str, train, val, test, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    patch_inductive_make_model()
    return run_text_benchmark(name, train, val, test, args, device)


def save_table_png(rows, headers, title: str, path: Path) -> None:
    if plt is None:
        return
    height = max(2.5, 0.45 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(headers)), height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12)
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_diagnostic_pngs(results: Dict[str, Any], out_dir: Path) -> None:
    if plt is None:
        print("matplotlib unavailable; skipped PNG diagnostics.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for bench, result in results.items():
        if "models" not in result:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        for model, metrics in result["models"].items():
            ax.plot(metrics["history"]["val_acc"], marker="o", label=f"{model} val")
        ax.set_title(f"{bench}: validation accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{bench}_multihop_validation_curves.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

        groups = sorted({g for metrics in result["models"].values() for g in metrics["test"]["by_group"]}, key=lambda x: int(x) if str(x).isdigit() else str(x))
        fig, ax = plt.subplots(figsize=(10, 5))
        width = 0.8 / max(1, len(result["models"]))
        x = list(range(len(groups)))
        for offset, (model, metrics) in enumerate(result["models"].items()):
            vals = [metrics["test"]["by_group"].get(g, {}).get("acc", 0.0) for g in groups]
            xpos = [v - 0.4 + width / 2 + offset * width for v in x]
            ax.bar(xpos, vals, width=width, label=model)
        ax.set_title(f"{bench}: held-out accuracy by chain/support length")
        ax.set_xlabel("Chain/support length")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        if bench == "clutrr_test":
            caption = (
                "CLUTRR gen_train234_test2to10 trains only on chain lengths 2-4; "
                "held-out test includes lengths 2-10. The drop after length 4 "
                "therefore measures out-of-distribution longer-chain generalization."
            )
            fig.text(0.5, -0.03, caption, ha="center", va="top", fontsize=8, wrap=True)
        fig.tight_layout()
        fig.savefig(out_dir / f"{bench}_multihop_test_by_length.png", dpi=170, bbox_inches="tight", pad_inches=0.35)
        plt.close(fig)

        label_rows = [[idx, label] for idx, label in enumerate(result.get("labels", []))]
        save_table_png(label_rows, ["id", "label"], f"{bench}: label mapping", out_dir / f"{bench}_label_mapping.png")

        sample_rows = []
        for model, metrics in result["models"].items():
            for idx, sample in enumerate(metrics["test"].get("samples", [])):
                sample_rows.append(
                    [
                        model,
                        idx,
                        sample.get("group", ""),
                        sample.get("true", ""),
                        sample.get("predicted", ""),
                        sample.get("confidence", ""),
                        "OK" if sample.get("correct") else "MISS",
                        str(sample.get("text", ""))[:120],
                    ]
                )
        save_table_png(
            sample_rows,
            ["model", "sample", "group", "true", "predicted", "conf", "result", "text"],
            f"{bench}: sample predictions",
            out_dir / f"{bench}_sample_predictions.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=str, default="clutrr")
    parser.add_argument("--models", type=parse_models, default=parse_models("arm,multihop_arm,transformer_small"))
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
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--final-hop-only", action="store_true")
    parser.add_argument("--tau", type=float, default=0.45)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cycle-weight", type=float, default=0.003)
    parser.add_argument("--op-weight", type=float, default=0.001)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=5)
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
    print(f"Multi-hop config: max_hops={args.max_hops}, beam_width={args.beam_width}, final_hop_only={args.final_hop_only}")

    results: Dict[str, Any] = {}
    if "clutrr" in selected:
        used_config, train, val, test = load_clutrr_examples(args)
        results["clutrr_test"] = run_benchmark("clutrr_test", train, val, test, args, device)
        results["clutrr_test"]["dataset_config"] = used_config
    if "babi" in selected:
        train, val, test = load_babi_examples(args)
        results["babi"] = run_benchmark("babi", train, val, test, args, device)
        results["babi"]["tasks"] = args.babi_tasks

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "multihop_arm_compare.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    save_diagnostic_pngs(results, out_dir)

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
        for bench, result in results.items()
        if "models" in result
    }
    print("\nFinal multi-hop ARM summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {out_file}")
    if plt is not None:
        print(f"Wrote PNG diagnostics to: {out_dir}")


if __name__ == "__main__":
    main()
