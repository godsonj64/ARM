from __future__ import annotations

import csv
import json
import re
import ssl
import unicodedata
from pathlib import Path
from typing import Any, Sequence
from urllib.request import urlopen

import torch
from torch.utils.data import Dataset

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)
MANY_BLANK_LINES_RE = re.compile(r"\n{3,}")


class ByteTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    byte_offset = 3
    vocab_size = 259

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [byte + self.byte_offset for byte in text.encode("utf-8", errors="replace")]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        byte_values = [idx - self.byte_offset for idx in ids if self.byte_offset <= int(idx) < self.vocab_size]
        return bytes(byte_values).decode("utf-8", errors="replace")


def _textify(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_textify(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key}: {_textify(item)}" for key, item in value.items())
    return str(value)


def preprocess_text(text: str, max_chars: int = 0) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_RE.sub("", text)
    text = TRAILING_SPACE_RE.sub("", text)
    lines = [line.strip() if not line.strip() else line for line in text.split("\n")]
    text = "\n".join(lines).strip()
    text = MANY_BLANK_LINES_RE.sub("\n\n", text)
    if max_chars > 0:
        text = text[:max_chars]
    if not text:
        raise ValueError("preprocessing produced an empty corpus")
    return text + "\n"


def load_text(path: str, text_field: str = "text") -> str:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        parts = [load_text(str(file), text_field) for file in sorted(source.rglob("*")) if file.suffix.lower() in {".txt", ".md", ".jsonl", ".csv"}]
        return "\n".join(part for part in parts if part.strip())

    suffix = source.suffix.lower()
    if suffix in {".txt", ".md"}:
        return source.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        lines = []
        with source.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                item = json.loads(line)
                lines.append(_textify(item.get(text_field, item)) if isinstance(item, dict) else _textify(item))
        return "\n".join(lines)
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            return ""
        field = text_field if text_field in rows[0] else next(iter(rows[0]))
        return "\n".join(_textify(row[field]) for row in rows)
    raise ValueError(f"unsupported data file type: {source.suffix}")


def download_tiny_shakespeare(cache_dir: str) -> Path:
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "tiny_shakespeare.txt"
    if not target.exists():
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            context = ssl.create_default_context()
        with urlopen(TINY_SHAKESPEARE_URL, timeout=120, context=context) as response:
            target.write_bytes(response.read())
    return target


def load_wikitext(cache_dir: str, split: str = "train") -> str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("install datasets or use --dataset tiny_shakespeare") from exc
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir)
    return "\n".join(row["text"] for row in dataset if row.get("text", "").strip())


def load_downloaded_dataset(name: str, cache_dir: str) -> str:
    if name == "tiny_shakespeare":
        return load_text(str(download_tiny_shakespeare(cache_dir)))
    if name == "wikitext":
        return load_wikitext(cache_dir)
    raise ValueError(f"unknown downloaded dataset: {name}")


class NextTokenDataset(Dataset):
    def __init__(self, token_ids: Sequence[int], block_size: int):
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.ids = torch.tensor(list(token_ids), dtype=torch.long)
        self.block_size = block_size
        if self.ids.numel() <= block_size:
            repeats = block_size // max(1, self.ids.numel()) + 2
            self.ids = self.ids.repeat(repeats)

    def __len__(self) -> int:
        return max(1, self.ids.numel() - self.block_size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        chunk = self.ids[index : index + self.block_size + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}
