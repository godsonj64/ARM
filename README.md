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

### 2. Real bAbI memory QA benchmark

`arm_babi_colab.py` trains ARM on the official Facebook bAbI task archive. The default task is Task 2, two supporting facts, which tests multi-hop memory retrieval from short stories.

The bAbI script includes:

- Official bAbI archive download
- Safe archive extraction
- bAbI task parser
- Word-level tokenizer
- GRU text encoder
- ARM answer-memory classifier
- Learned algebraic operator bank
- Multi-path resonance retrieval
- Cycle-consistency regularization
- Validation accuracy reporting
- Checkpoint saving as `arm_babi_checkpoint.pt`

### 3. CLUTRR experimental script

`arm_clutrr_colab.py` remains in the repository as an experimental script, but CLUTRR is not available through Hugging Face as `clutrr`. The Colab launcher therefore uses bAbI as the dependable real benchmark.

## Run bAbI in Google Colab

Open `arm_colab_runnable.ipynb` in Colab and run the single code cell.

The notebook clones this repository, runs `arm_babi_colab.py`, plots learning curves, and saves a checkpoint.

## Run bAbI locally

```bash
pip install -r requirements.txt
python arm_babi_colab.py
```

## Run the synthetic benchmark locally

```bash
pip install -r requirements.txt
python arm_colab_runnable.py
```

Both scripts automatically use CUDA when available.

## Files

```text
arm_babi_colab.py           Real bAbI memory QA benchmark
arm_clutrr_colab.py         Experimental CLUTRR script
arm_colab_runnable.py       Synthetic cyclic fan-state benchmark
arm_colab_runnable.ipynb    Colab launcher for bAbI
requirements.txt            Python dependencies
.gitignore                  Ignore generated checkpoints and caches
LICENSE                     MIT license
```

## Research status

This is a research prototype. The synthetic benchmark tests whether ARM can retrieve one hidden memory from multiple cue forms. The bAbI benchmark tests whether ARM can learn text-based multi-hop memory retrieval on a real public QA dataset. It is not yet a replacement for transformer attention on large-scale language tasks.

## Author

Concept and implementation direction: Godson Johnson
