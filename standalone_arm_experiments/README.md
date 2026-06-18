# Algebraic Resonance Memory Standalone Experiments

This folder is a self-contained experimental scaffold for comparing Algebraic
Resonance Memory (ARM) against related memory and sequence-classification
baselines.

ARM is a differentiable associative-memory layer. A query vector is transformed
by multiple learned operators and scored against learned memory atoms using a
learned positive metric distance:

```text
rho(q, m_i) = logsumexp_k(-||A_k q + b_k - m_i||^2_D / tau - c_k)
```

Unlike dot-product attention, ARM does not compute query-key compatibility over
sequence positions. It performs operator-mediated resonance retrieval over a
fixed memory bank.

## Baselines

The training script supports:

- `arm`: operator-mediated resonance memory.
- `dot_memory`: direct dot-product memory over learned atoms.
- `prototype`: learned class prototypes scored by negative squared distance.
- `rbf`: learned RBF memory scored by radial basis distances.
- `hopfield`: modern Hopfield-style associative memory over learned patterns.
- `transformer`: Transformer encoder classifier.

## Quick Start

From the repo root:

```bash
.venv/bin/python standalone_arm_experiments/train.py \
  --dataset synthetic \
  --models arm,dot_memory,prototype,rbf,hopfield,transformer \
  --epochs 5
```

Run on CLUTRR through Hugging Face/fallback CSV loader:

```bash
.venv/bin/python standalone_arm_experiments/train.py \
  --dataset clutrr \
  --models arm,dot_memory,prototype,rbf,hopfield,transformer \
  --epochs 8 \
  --max-train 12000 \
  --max-eval 3000
```

Run on a local CLUTTER/CLUTRR-style directory:

```bash
.venv/bin/python standalone_arm_experiments/train.py \
  --dataset local \
  --data-path /Users/godsonjohnson/CLUTTER2.0 \
  --models arm,dot_memory,prototype,rbf,hopfield,transformer \
  --epochs 20 \
  --max-len 512
```

Outputs are written to `standalone_arm_experiments/results/` by default:

- `metrics.json`
- one checkpoint per model
- printed sample predictions

## Design

For the memory models, the common text encoder is:

```text
tokens -> embedding -> BiGRU -> masked mean pooling -> projection -> query
```

The memory/retrieval layer is then swapped. The Transformer baseline uses token
and positional embeddings followed by a Transformer encoder and masked mean
pooling.

