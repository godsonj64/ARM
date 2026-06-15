# Algebraic Resonance Memory (ARM)

Algebraic Resonance Memory is a PyTorch research prototype for memory retrieval through learned algebraic transformations rather than direct query-key similarity alone.

The central idea is that a single memory can be retrieved by multiple different cues when those cues converge to the same latent memory point through a family of learned operators. The motivating example is a cyclic pull-rope fan switch: each pull updates a hidden state, while the brain may retrieve the current state through motor rhythm, airflow, sound, partial count, or expectation.

## Core retrieval equation

For a query `q`, memory atom `m_i`, and learned operator family `A_k`, ARM computes:

```text
rho(q, m_i) = logsumexp_k( -||A_k q + b_k - m_i||^2_D / tau - c_k )
```

The retrieved memory is:

```text
R(q) = softmax_i(rho(q, m_i)) M
```

This lets different transformed versions of a query retrieve the same memory atom.

## Benchmarks included

### 1. Synthetic cyclic fan-state benchmark

`arm_colab_runnable.py` tests ARM on the original cyclic hidden-state idea. Multiple cues point to the same hidden fan state.

### 2. Real CLUTRR benchmark

`arm_clutrr_colab.py` trains ARM on CLUTRR, a real kinship-reasoning benchmark where a model must infer hidden family relations from short stories. This is a natural test for ARM because a target relation can be reached through different relational paths.

The CLUTRR script includes:

- Hugging Face dataset loading with config fallback
- Defensive column inference for CLUTRR variants
- Word-level tokenizer
- GRU text encoder
- ARM relation-memory classifier
- Learned algebraic operator bank
- Multi-path resonance retrieval
- Cycle-consistency regularization
- Validation accuracy reporting
- Checkpoint saving as `arm_clutrr_checkpoint.pt`

## Run CLUTRR in Google Colab

Open `arm_colab_runnable.ipynb` in Colab and run the single code cell.

The notebook installs dependencies, downloads `arm_clutrr_colab.py`, trains ARM on CLUTRR, plots learning curves, and saves a checkpoint.

## Run CLUTRR locally

```bash
pip install -r requirements.txt
python arm_clutrr_colab.py
```

## Run the synthetic benchmark locally

```bash
pip install -r requirements.txt
python arm_colab_runnable.py
```

Both scripts automatically use CUDA when available.

## Files

```text
arm_clutrr_colab.py         Real CLUTRR benchmark script
arm_colab_runnable.py       Synthetic cyclic fan-state benchmark
arm_colab_runnable.ipynb    Colab launcher for CLUTRR
requirements.txt            Python dependencies
.gitignore                  Ignore generated checkpoints and caches
LICENSE                     MIT license
```

## Research status

This is a research prototype. The synthetic benchmark tests whether ARM can retrieve one hidden memory from multiple cue forms. The CLUTRR benchmark tests whether ARM can learn relational memory retrieval on real text-based kinship reasoning. It is not yet a replacement for transformer attention on large-scale language tasks.

## Author

Concept and implementation direction: Godson Johnson
