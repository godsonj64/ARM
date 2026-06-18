# Generative ARM Text Model

This folder is a standalone causal text-generation scaffold for Algebraic
Resonance Memory (ARM). It trains a small byte-level language model that uses a
GRU for causal sequence state and an ARM block for operator-mediated memory
retrieval before predicting the next token.

The ARM block learns:

- memory atoms used as a differentiable associative store
- operator matrices that transform each token state into multiple paths
- a positive diagonal metric for resonance distance
- retrieval weights over memory atoms for every generated position

## Quick Start

From the repo root:

```bash
.venv/bin/python generative_arm_text/train.py --epochs 5
```

By default this auto-downloads the real Tiny Shakespeare corpus into
`generative_arm_text/downloaded_data/`, preprocesses it, and trains on the
byte-level next-token objective.

Generate from the trained checkpoint:

```bash
.venv/bin/python generative_arm_text/generate.py \
  --checkpoint generative_arm_text/runs/tiny_arm_lm/best.pt \
  --prompt "Algebraic resonance"
```

## Train On Your Own Text

Pass a `.txt`, `.md`, `.jsonl`, `.csv`, or a directory containing those files:

```bash
.venv/bin/python generative_arm_text/train.py \
  --data-path /path/to/corpus.txt \
  --epochs 20 \
  --block-size 256 \
  --batch-size 32 \
  --out-dir generative_arm_text/runs/my_corpus
```

For JSONL/CSV data, `--text-field text` is used by default.

## Downloaded Datasets

Use the default Tiny Shakespeare corpus:

```bash
.venv/bin/python generative_arm_text/train.py \
  --dataset tiny_shakespeare \
  --epochs 20
```

Or use WikiText-2 through Hugging Face `datasets`:

```bash
.venv/bin/python generative_arm_text/train.py \
  --dataset wikitext \
  --epochs 20 \
  --cache-dir generative_arm_text/downloaded_data
```

Preprocessing normalizes Unicode, standardizes line endings, removes control
characters, trims trailing whitespace, collapses excessive blank lines, and
writes a `preprocessed_corpus_preview.txt` file in the run directory.

For quick smoke tests on a subset:

```bash
.venv/bin/python generative_arm_text/train.py \
  --max-chars 50000 \
  --epochs 1
```

## Files

- `model.py`: `GenerativeARMLanguageModel` and the causal ARM block.
- `data.py`: byte tokenizer, real dataset downloading, preprocessing, and
  text/JSONL/CSV loading.
- `train.py`: standalone training entry point.
- `generate.py`: checkpoint loading and text sampling.

The model is byte-level, so it does not need a separate tokenizer file and can
generate arbitrary UTF-8 text.
