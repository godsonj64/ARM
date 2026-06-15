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

## What is included

- Learned affine algebraic operator bank
- Multi-path resonance scoring with `logsumexp`
- Learned positive diagonal metric
- Learned nonnegative operator costs
- Cycle-consistency loss, such as `A^4 ≈ I`
- Equivalence loss for different cues of the same hidden state
- Synthetic cyclic fan-state benchmark
- Training loop, validation loop, tests, plots, and checkpoint saving
- Google Colab notebook and standalone Python script

## Run in Google Colab

Open `arm_colab_runnable.ipynb` in Colab and run the single code cell.

The notebook will:

1. Build the cyclic fan-state dataset.
2. Instantiate the ARM model.
3. Run shape, gradient, and finite-loss tests.
4. Train the model.
5. Evaluate retrieval accuracy.
6. Print memory retrieval examples.
7. Plot learning curves.
8. Save `arm_colab_checkpoint.pt`.

## Run locally

```bash
pip install -r requirements.txt
python arm_colab_runnable.py
```

The script automatically uses CUDA when available.

## Files

```text
arm_colab_runnable.py       Standalone PyTorch script
arm_colab_runnable.ipynb    Colab notebook
requirements.txt            Minimal dependencies
.gitignore                  Ignore generated checkpoints and caches
```

## Research status

This is a research prototype. It is intended to test the ARM mechanism on a controlled cyclic hidden-state benchmark. It is not yet a replacement for transformer attention on large-scale language tasks.

## Author

Concept and implementation direction: Godson Johnson
