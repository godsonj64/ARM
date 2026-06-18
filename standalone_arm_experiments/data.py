from __future__ import annotations

import csv
import json
import random
import re
import ssl
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Sequence
from urllib.request import urlopen

import torch
from torch.utils.data import Dataset


TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[.,!?;:()\-]")
CLUTRR_RAW_BASE = "https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/main"
CLUTRR_CONFIGS = ["gen_train23_test2to10", "gen_train234_test2to10"]


def textify(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(textify(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key} {textify(item)}" for key, item in value.items())
    return str(value)


def pick_column(columns: Sequence[str], names: Sequence[str], required: bool = True) -> str:
    lowered = {column.lower(): column for column in columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    for column in columns:
        for name in names:
            if name.lower() in column.lower():
                return column
    if required:
        raise KeyError(f"could not find any of {names} in columns {columns}")
    return ""


class Tokenizer:
    def __init__(self, max_vocab: int):
        self.max_vocab = max_vocab
        self.stoi = {"[PAD]": 0, "[UNK]": 1}
        self.itos = ["[PAD]", "[UNK]"]

    def tokenize(self, text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    def fit(self, texts: Sequence[str]) -> None:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(self.tokenize(text))
        for token, _ in counts.most_common(self.max_vocab - len(self.itos)):
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def encode(self, text: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self.stoi.get(token, 1) for token in self.tokenize(text)[:max_len]]
        mask = [1.0] * len(ids)
        ids += [0] * (max_len - len(ids))
        mask += [0.0] * (max_len - len(mask))
        return torch.tensor(ids).long(), torch.tensor(mask).float()


class TextDataset(Dataset):
    def __init__(self, examples: Sequence[Dict[str, Any]], tokenizer: Tokenizer, label_to_id: Dict[str, int], max_len: int):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        input_ids, attention_mask = self.tokenizer.encode(example["text"], self.max_len)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(self.label_to_id[example["label"]]).long(),
        }


def read_url_text(url: str) -> str:
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = ssl.create_default_context()
    with urlopen(url, timeout=60, context=context) as response:
        return response.read().decode("utf-8")


def load_clutrr_csvs(config: str) -> dict[str, list[dict[str, str]]]:
    if config not in CLUTRR_CONFIGS:
        raise ValueError(f"unknown CLUTRR config {config!r}")
    splits = {}
    for split in ["train", "validation", "test"]:
        text = read_url_text(f"{CLUTRR_RAW_BASE}/{config}/{split}.csv")
        splits[split] = list(csv.DictReader(StringIO(text)))
    return splits


def rows_from_records(records: Sequence[Dict[str, Any]], limit: int) -> list[Dict[str, Any]]:
    if not records:
        return []
    columns = list(records[0].keys())
    story_col = pick_column(columns, ["story", "clean_story", "text", "context", "sentence", "input", "passage"])
    query_col = pick_column(columns, ["query", "question", "prompt"], required=False)
    label_col = pick_column(columns, ["target_text", "target", "answer", "relation", "target_label", "label", "class"])
    group_col = pick_column(columns, ["task_name", "task", "group", "hops"], required=False)
    examples = []
    for index, row in enumerate(records[:limit]):
        story = textify(row[story_col])
        query = textify(row[query_col]) if query_col else ""
        examples.append(
            {
                "text": (story + " [QUERY] " + query).strip() if query else story.strip(),
                "label": textify(row[label_col]),
                "group": textify(row[group_col]) if group_col else "unknown",
                "row_index": index,
            }
        )
    return examples


def infer_split(path: Path) -> str:
    name = path.name.lower()
    if any(token in name for token in ["validation", "valid", "val", "dev"]):
        return "validation"
    if "test" in name:
        return "test"
    if "train" in name:
        return "train"
    return "unknown"


def direct_split_files(path: Path) -> list[Path]:
    return [
        path / name
        for name in ["train.csv", "validation.csv", "valid.csv", "val.csv", "dev.csv", "test.csv", "train.jsonl", "validation.jsonl", "test.jsonl"]
        if (path / name).is_file()
    ]


def load_records(path: Path) -> list[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    item = json.loads(line)
                    if isinstance(item, dict):
                        rows.append(item)
        return rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            obj = json.load(file)
        if isinstance(obj, list):
            return [item for item in obj if isinstance(item, dict)]
        if isinstance(obj, dict):
            for key in ["data", "examples", "rows", "records", "items"]:
                if isinstance(obj.get(key), list):
                    return [item for item in obj[key] if isinstance(item, dict)]
            return [obj]
    return []


def load_local(path: str, max_train: int, max_eval: int, max_test: int, seed: int) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    files = [root] if root.is_file() else direct_split_files(root)
    if not files and root.is_dir():
        files = sorted(file for file in root.rglob("*") if file.suffix.lower() in {".csv", ".json", ".jsonl"})
    split_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for file in files:
        split_records[infer_split(file)].extend(load_records(file))
    train = rows_from_records(split_records["train"], max_train)
    val = rows_from_records(split_records["validation"], max_eval)
    test = rows_from_records(split_records["test"], max_test)
    if not train:
        unknown = rows_from_records(split_records["unknown"], max_train + max_eval + max_test)
        random.Random(seed).shuffle(unknown)
        train_cut = int(0.8 * len(unknown))
        val_cut = int(0.9 * len(unknown))
        train, val, test = unknown[:train_cut], unknown[train_cut:val_cut], unknown[val_cut:]
    if train and not val:
        random.Random(seed).shuffle(train)
        val_size = max(1, int(0.15 * len(train)))
        train, val = train[:-val_size], train[-val_size:]
    if not test:
        test = val
    return train[:max_train], val[:max_eval], test[:max_test]


def load_synthetic(num_train: int, num_eval: int, num_test: int, seed: int) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    relations = {
        "r0": lambda x: (x + 1) % 8,
        "r1": lambda x: (x + 2) % 8,
        "r2": lambda x: (2 * x + 1) % 8,
        "r3": lambda x: (3 * x + 2) % 8,
    }

    def make(n: int, lengths: Sequence[int], offset: int) -> list[Dict[str, Any]]:
        rng = random.Random(seed + offset)
        examples = []
        for index in range(n):
            state = rng.randrange(8)
            start = state
            ops = [rng.choice(list(relations)) for _ in range(rng.choice(lengths))]
            for op in ops:
                state = relations[op](state)
            text = f"start s{start} . " + " ".join(f"apply {op} ." for op in ops) + " [QUERY] final state ?"
            examples.append({"text": text, "label": f"s{state}", "group": str(len(ops)), "row_index": index})
        return examples

    return make(num_train, [2, 3], 1), make(num_eval, [2, 3], 2), make(num_test, [4, 5, 6], 3)

