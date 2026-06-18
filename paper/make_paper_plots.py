"""Render figures used by the ARM paper from the original CLUTRR benchmark results.

Run from the repository root:
    python paper/make_paper_plots.py

The script writes vector PDF and PNG figures into paper/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper" / "results_original_clutrr_15ep.json"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

data = json.loads(RESULTS.read_text(encoding="utf-8"))
val = data["validation_curves"]
summary = data["summary"]
labels = list(val.keys())
pretty = {
    "arm": "ARM",
    "enhanced_arm": "Enhanced ARM",
    "dot_memory": "Dot memory",
    "prototype": "Prototype",
    "rbf": "RBF",
    "hopfield": "Hopfield",
    "transformer": "Transformer",
}
epochs = np.arange(1, data["epochs"] + 1)

plt.figure(figsize=(8.6, 5.2))
for name in labels:
    lw = 2.4 if name in {"arm", "rbf", "prototype"} else 1.6
    plt.plot(epochs, val[name], marker="o", markersize=3.2, linewidth=lw, label=pretty[name])
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.title("Original CLUTRR validation learning curves")
plt.ylim(0.08, 1.01)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(OUT / "clutrr_val_curves.pdf")
plt.savefig(OUT / "clutrr_val_curves.png", dpi=220)
plt.close()

x = np.arange(len(labels))
width = 0.38
best = [summary[n]["best_val_acc"] for n in labels]
test = [summary[n]["test_acc"] for n in labels]
plt.figure(figsize=(9.0, 5.0))
plt.bar(x - width / 2, best, width, label="Best validation accuracy")
plt.bar(x + width / 2, test, width, label="Reported test accuracy")
plt.xticks(x, [pretty[n] for n in labels], rotation=35, ha="right")
plt.ylabel("Accuracy")
plt.title("CLUTRR summary across memory mechanisms")
plt.ylim(0, 1.03)
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "clutrr_summary_bars.pdf")
plt.savefig(OUT / "clutrr_summary_bars.png", dpi=220)
plt.close()

params = np.array([summary[n]["parameters"] / 1e6 for n in labels])
plt.figure(figsize=(8.4, 5.2))
for n, p, b in zip(labels, params, best):
    plt.scatter(p, b, s=60)
    plt.text(p + 0.004, b, pretty[n], fontsize=8, va="center")
plt.xlabel("Parameters (millions)")
plt.ylabel("Best validation accuracy")
plt.title("Parameter count vs validation performance")
plt.ylim(0.78, 1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "clutrr_params_val.pdf")
plt.savefig(OUT / "clutrr_params_val.png", dpi=220)
plt.close()

print(f"Wrote figures to {OUT}")
