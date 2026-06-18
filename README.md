# Algebraic Resonance Memory (ARM)

Algebraic Resonance Memory is a PyTorch research prototype for **transformation-mediated memory retrieval**. Standard attention retrieves by direct query-key similarity. ARM retrieves by applying a learned family of algebraic operators to the query and aggregating the resonance of all transformed paths that reach each memory atom.

## Core equation

For query `q`, memory atom `m_i`, and learned operators `A_k`, ARM computes:

```text
rho(q, m_i) = logsumexp_k( -||A_k q + b_k - m_i||^2_D / tau - c_k )
R(q)        = softmax_i(rho(q, m_i)) M
```

This allows different cue forms, relational paths, or temporal states to converge to the same latent memory.

## Repository structure

```text
arm/
  __init__.py              Public package exports
  models.py                ARM layer, attention baseline, encoders
  synthetic.py             Cyclic fan-state dataset

benchmarks/
  synthetic_compare.py     ARM vs direct attention on cyclic hidden-state retrieval
  clutrr_compare.py        ARM vs direct attention on CLUTRR kinship reasoning

arm_colab_runnable.py      Original synthetic Colab script
arm_clutrr_colab.py        Original CLUTRR Colab script
arm_colab_runnable.ipynb   Main Colab launcher for benchmark comparison
run_colab_benchmarks.ipynb Additional Colab benchmark launcher
requirements.txt           Python dependencies
```

## Install

```bash
pip install -r requirements.txt
```

## Benchmark 1: synthetic cyclic fan-state retrieval

This benchmark tests the original ARM idea: multiple cue types retrieve the same hidden cyclic state.

```bash
python benchmarks/synthetic_compare.py --epochs 20
```

Outputs:

```text
benchmark_results/synthetic_compare.json
benchmark_results/synthetic_learning_curves.png
```

The script trains two models under the same setup:

1. `attention`: direct query-memory attention baseline.
2. `arm`: Algebraic Resonance Memory with learned operator paths.

## Benchmark 2: CLUTRR real-dataset comparison

This benchmark downloads CLUTRR from Hugging Face and compares ARM against direct attention on kinship-relation reasoning.

```bash
python benchmarks/clutrr_compare.py --epochs 8
```

Outputs:

```text
benchmark_results/clutrr_compare.json
benchmark_results/clutrr_learning_curves.png
```

## Run in Google Colab

Open either notebook:

```text
arm_colab_runnable.ipynb
run_colab_benchmarks.ipynb
```

The notebooks clone this repository, install requirements, and run both benchmark scripts.

## Notes on interpretation

The benchmark scripts are intended to test whether ARM's transformed-query retrieval offers advantages over direct attention under controlled conditions. They do not prove general superiority over Transformer attention. Stronger claims require repeated runs, confidence intervals, tuned baselines, and larger-scale experiments.

## Paper

The theoretical manuscript frames ARM as a transformation-mediated retrieval scoring principle, not as a replacement for all attention mechanisms.

## License

MIT License.
