# Algebraic Resonance Memory (ARM)

Algebraic Resonance Memory is a PyTorch research prototype for **transformation-mediated memory retrieval**. Standard attention retrieves by direct query-key similarity. ARM retrieves by applying a learned family of algebraic operators to the query and aggregating the resonance of all transformed paths that reach each memory atom.

## Core equation

For query `q`, memory atom `m_i`, and learned operators `A_k`, ARM computes:

```text
rho(q, m_i) = logsumexp_k( -||A_k q + b_k - m_i||^2_D / tau - c_k )
R(q)        = softmax_i(rho(q, m_i)) M
```

This allows different cue forms, relational paths, or temporal states to converge to the same latent memory.

## Primary benchmark: CLUTRR real-data comparison

The default benchmark now uses **CLUTRR**, a real kinship-reasoning dataset downloaded automatically through Hugging Face `datasets`.

It compares:

1. `attention`: direct query-memory attention baseline.
2. `arm`: Algebraic Resonance Memory with learned operator paths.

Run locally:

```bash
pip install -r requirements.txt
python benchmarks/clutrr_compare.py --epochs 8
```

Outputs:

```text
benchmark_results/clutrr_compare.json
benchmark_results/clutrr_learning_curves.png
```

The script automatically:

- downloads CLUTRR,
- detects available CLUTRR configs,
- falls back across configs if needed,
- infers story/query/label columns defensively,
- builds a word-level tokenizer,
- trains both ARM and attention under the same encoder setup,
- writes JSON metrics and a learning-curve plot.

## Run in Google Colab

Open:

```text
run_clutrr_auto_download.ipynb
```

or:

```text
arm_colab_runnable.ipynb
```

Both notebooks clone the repo, install dependencies, auto-download CLUTRR, and run the real-data benchmark.

## Repository structure

```text
arm/
  __init__.py              Public package exports
  models.py                ARM layer, attention baseline, encoders
  synthetic.py             Legacy synthetic cyclic fan-state dataset

benchmarks/
  clutrr_compare.py        Primary real-data ARM vs attention benchmark
  synthetic_compare.py     Legacy synthetic cyclic hidden-state benchmark

run_clutrr_auto_download.ipynb  Main CLUTRR auto-download Colab notebook
arm_colab_runnable.ipynb        CLUTRR-only Colab launcher
run_colab_benchmarks.ipynb      Optional benchmark launcher
requirements.txt                Python dependencies
```

## Optional legacy synthetic benchmark

The synthetic cyclic benchmark is still available for controlled debugging, but it is no longer the default benchmark path.

```bash
python benchmarks/synthetic_compare.py --epochs 20
```

## Notes on interpretation

The CLUTRR benchmark tests ARM against a direct attention-memory baseline under the same text encoder. Results should be interpreted as an initial architecture comparison, not as proof of general superiority over Transformer attention. Stronger claims require repeated runs, tuned baselines, confidence intervals, and larger-scale experiments.

## License

MIT License.
