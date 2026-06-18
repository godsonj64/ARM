"""Run the full multi-hop ARM evaluation suite.

This is an orchestration script for the experiment requested most often during
ARM development:

1. CLUTRR held-out test evaluation for 6 epochs.
2. bAbI task 2/3 evaluation for 25 epochs.
3. Models: single-hop ARM, multi-hop ARM, and transformer_small.
4. JSON metrics plus PNG diagnostics for curves, mappings, chain/support
   length accuracy, and sample predictions.

Run:
    python benchmarks/run_multihop_full_eval.py

Outputs:
    benchmark_results/multihop_full_eval/clutrr_6/
    benchmark_results/multihop_full_eval/babi_25/
    benchmark_results/multihop_full_eval/full_eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> None:
    print("\nRunning:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def load_result(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact_summary(result):
    summary = {}
    for benchmark, bench_result in result.items():
        if "models" not in bench_result:
            continue
        summary[benchmark] = {}
        for model, metrics in bench_result["models"].items():
            summary[benchmark][model] = {
                "parameters": metrics["parameters"],
                "best_val_acc": metrics["best_val_acc"],
                "test_acc": metrics["test"]["acc"],
                "test_by_chain_or_support_length": metrics["test"]["by_group"],
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="arm,multihop_arm,transformer_small")
    parser.add_argument("--out-dir", type=str, default="benchmark_results/multihop_full_eval")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--clutrr-epochs", type=int, default=6)
    parser.add_argument("--babi-epochs", type=int, default=25)
    parser.add_argument("--babi-tasks", type=str, default="2,3")
    args = parser.parse_args()

    root = Path(args.out_dir)
    clutrr_out = root / "clutrr_6"
    babi_out = root / "babi_25"
    root.mkdir(parents=True, exist_ok=True)

    common = [
        sys.executable,
        str(Path(__file__).resolve().parent / "multihop_arm_compare.py"),
        "--models",
        args.models,
        "--device",
        args.device,
        "--num-samples",
        str(args.num_samples),
        "--max-hops",
        str(args.max_hops),
        "--beam-width",
        str(args.beam_width),
    ]

    run_command(
        common
        + [
            "--benchmarks",
            "clutrr",
            "--epochs",
            str(args.clutrr_epochs),
            "--out-dir",
            str(clutrr_out),
        ]
    )

    run_command(
        common
        + [
            "--benchmarks",
            "babi",
            "--epochs",
            str(args.babi_epochs),
            "--babi-tasks",
            args.babi_tasks,
            "--out-dir",
            str(babi_out),
        ]
    )

    clutrr_result = load_result(clutrr_out / "multihop_arm_compare.json")
    babi_result = load_result(babi_out / "multihop_arm_compare.json")
    summary = {"clutrr_6": compact_summary(clutrr_result), "babi_25": compact_summary(babi_result)}
    summary_path = root / "full_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nFull multi-hop evaluation summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {summary_path}")
    print(f"CLUTRR outputs: {clutrr_out}")
    print(f"bAbI outputs: {babi_out}")


if __name__ == "__main__":
    main()
