# ============================================================
# Algebraic Resonance Memory (ARM) on Facebook bAbI QA
# Colab runnable PyTorch script
# Concept direction: Godson Johnson
# ============================================================

import math
import os
import re
import random
import tarfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

try:
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def seed_all(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


@dataclass
class Config:
    data_url: str = "https://dl.fbaipublicfiles.com/babi/tasks_1-20_v1-2/en-valid-10k.tar.gz"
    data_dir: str = "data_babi"
    task_id: int = 2
    max_train: int = 10000
    max_eval: int = 2000
    max_vocab: int = 20000
    max_context_len: int = 220
    dim: int = 128
    emb: int = 128
    hidden: int = 128
    operators: int = 8
    tau: float = 0.45
    batch: int = 128
    epochs: int = 15
    lr: float = 2e-3
    wd: float = 1e-4
    cycle_w: float = 0.002
    op_w: float = 0.001
    clip: float = 1.0
    dropout: float = 0.15


cfg = Config()


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if not str(target).startswith(str(destination)):
            raise RuntimeError(f"Unsafe archive path blocked: {member.name}")
    tar.extractall(destination)


def download_babi(cfg: Config) -> Path:
    root = Path(cfg.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "tasks_1-20_v1-2" / "en-valid-10k"
    if marker.exists():
        return marker

    archive = root / "en-valid-10k.tar.gz"
    if not archive.exists():
        print("Downloading official Facebook bAbI archive...")
        urllib.request.urlretrieve(cfg.data_url, archive)

    print("Extracting bAbI archive safely...")
    with tarfile.open(archive, "r:gz") as tar:
        safe_extract(tar, root)

    if not marker.exists():
        candidates = list(root.rglob("en-valid-10k"))
        if not candidates:
            raise RuntimeError("Could not find extracted bAbI en-valid-10k directory.")
        marker = candidates[0]
    return marker


def find_task_files(base: Path, task_id: int) -> Tuple[Path, Path]:
    prefix = f"qa{task_id}_"
    train_files = sorted(base.glob(f"{prefix}*train.txt"))
    test_files = sorted(base.glob(f"{prefix}*test.txt"))
    if not train_files or not test_files:
        available = sorted(p.name for p in base.glob("qa*_train.txt"))[:10]
        raise RuntimeError(f"Could not find bAbI task {task_id}. Available examples: {available}")
    return train_files[0], test_files[0]


def normalize_text(s: str) -> str:
    return s.strip().lower()


def parse_babi_file(path: Path, limit: int) -> List[Tuple[str, str, str]]:
    rows = []
    story = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            idx_text = line.split(" ", 1)
            if len(idx_text) != 2:
                continue
            idx = int(idx_text[0])
            text = idx_text[1]
            if idx == 1:
                story = []
            if "\t" in text:
                q, ans, _support = text.split("\t")
                context = " ".join(story)
                rows.append((normalize_text(context), normalize_text(q), normalize_text(ans)))
                if len(rows) >= limit:
                    break
            else:
                story.append(normalize_text(text))
    return rows


TOKEN = re.compile(r"[A-Za-z0-9_']+|[.,!?;:()\-]")


class Tokenizer:
    def __init__(self, max_vocab: int):
        self.stoi = {"[PAD]": 0, "[UNK]": 1, "[SEP]": 2}
        self.itos = ["[PAD]", "[UNK]", "[SEP]"]
        self.max_vocab = max_vocab

    def tok(self, s: str) -> List[str]:
        return TOKEN.findall(s.lower())

    def fit(self, texts: List[str]):
        counts = Counter()
        for t in texts:
            counts.update(self.tok(t))
        for word, _ in counts.most_common(self.max_vocab - len(self.itos)):
            if word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
        print("Vocab size:", len(self.itos))

    def encode(self, s: str, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = [self.stoi.get(tok, 1) for tok in self.tok(s)[:max_len]]
        mask = [1.0] * len(ids)
        ids += [0] * (max_len - len(ids))
        mask += [0.0] * (max_len - len(mask))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


class BabiDataset(Dataset):
    def __init__(self, rows, tokenizer: Tokenizer, answer_to_id: Dict[str, int]):
        self.rows = rows
        self.tokenizer = tokenizer
        self.answer_to_id = answer_to_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        context, question, answer = self.rows[idx]
        text = context + " [SEP] " + question
        ids, mask = self.tokenizer.encode(text, cfg.max_context_len)
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "label": torch.tensor(self.answer_to_id[answer], dtype=torch.long),
            "answer_text": answer,
        }


class AlgebraicResonanceMemory(nn.Module):
    def __init__(self, dim: int, memories: int, operators: int, tau: float):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.memory = nn.Parameter(torch.randn(memories, dim) / math.sqrt(dim))
        eye = torch.eye(dim).unsqueeze(0).repeat(operators, 1, 1)
        self.ops = nn.Parameter(eye + 0.02 * torch.randn(operators, dim, dim))
        self.bias = nn.Parameter(torch.zeros(operators, dim))
        self.metric_raw = nn.Parameter(torch.zeros(dim))
        self.cost_raw = nn.Parameter(torch.zeros(operators))
        self.qproj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.05), nn.Linear(dim, dim), nn.LayerNorm(dim))

    def metric(self):
        return F.softplus(self.metric_raw) + 1e-5

    def cost(self):
        return F.softplus(self.cost_raw)

    def forward(self, q: torch.Tensor):
        q = self.qproj(q)
        z = torch.einsum("bd,ked->bke", q, self.ops) + self.bias.unsqueeze(0)
        diff = z.unsqueeze(2) - self.memory.unsqueeze(0).unsqueeze(0)
        dist = (diff.square() * self.metric().view(1, 1, 1, -1)).sum(-1)
        path_scores = -dist / self.tau - self.cost().view(1, -1, 1)
        logits = torch.logsumexp(path_scores, dim=1)
        weights = F.softmax(logits, dim=-1)
        return {"logits": logits, "weights": weights, "retrieved": weights @ self.memory, "path_scores": path_scores}

    def cycle_loss(self, order: int = 4):
        Ap = torch.matrix_power(self.ops[0], order)
        I = torch.eye(self.dim, device=Ap.device, dtype=Ap.dtype)
        return F.mse_loss(Ap, I)

    def op_reg(self):
        I = torch.eye(self.dim, device=self.ops.device, dtype=self.ops.dtype)
        return (self.ops - I.unsqueeze(0)).square().mean() + 0.1 * self.bias.square().mean()


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, cfg.emb, padding_idx=0)
        self.gru = nn.GRU(cfg.emb, cfg.hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.LayerNorm(cfg.hidden * 2), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden * 2, cfg.dim), nn.GELU(), nn.LayerNorm(cfg.dim))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        x = self.emb(ids)
        out, _ = self.gru(x)
        m = mask.unsqueeze(-1)
        pooled = (out * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.proj(pooled)


class ARMBabiQA(nn.Module):
    def __init__(self, vocab_size: int, num_answers: int):
        super().__init__()
        self.encoder = TextEncoder(vocab_size)
        self.arm = AlgebraicResonanceMemory(cfg.dim, num_answers, cfg.operators, cfg.tau)

    def forward(self, ids, mask):
        return self.arm(self.encoder(ids, mask))


def batch_loss(model: ARMBabiQA, batch: Dict[str, torch.Tensor]):
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    y = batch["label"].to(device)
    out = model(ids, mask)
    ce = F.cross_entropy(out["logits"], y)
    loss = ce + cfg.cycle_w * model.arm.cycle_loss(4) + cfg.op_w * model.arm.op_reg()
    with torch.no_grad():
        acc = (out["logits"].argmax(-1) == y).float().mean().item()
    return loss, acc


@torch.no_grad()
def evaluate(model: ARMBabiQA, loader: DataLoader):
    model.eval()
    total_loss = total_acc = total = 0.0
    for batch in loader:
        loss, acc = batch_loss(model, batch)
        bs = batch["label"].shape[0]
        total_loss += loss.item() * bs
        total_acc += acc * bs
        total += bs
    return total_loss / max(1, total), total_acc / max(1, total)


def run_tests(model: ARMBabiQA, loader: DataLoader, num_answers: int):
    print("\nRunning tests...")
    batch = next(iter(loader))
    out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
    bs = batch["label"].shape[0]
    assert out["logits"].shape == (bs, num_answers)
    assert out["weights"].shape == (bs, num_answers)
    assert out["retrieved"].shape == (bs, cfg.dim)
    assert out["path_scores"].shape[0] == bs
    assert torch.isfinite(out["logits"]).all()
    loss, _ = batch_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"
    model.zero_grad(set_to_none=True)
    print("All tests passed.")


@torch.no_grad()
def inspect(model: ARMBabiQA, dataset: BabiDataset, id_to_answer: Dict[int, str], n: int = 12):
    model.eval()
    print("\nExample predictions:")
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        out = model(item["input_ids"].unsqueeze(0).to(device), item["attention_mask"].unsqueeze(0).to(device))
        probs = out["weights"].squeeze(0).cpu()
        pred_id = int(probs.argmax().item())
        true_id = int(item["label"].item())
        print(f"sample={i:02d} | true={id_to_answer[true_id]} | pred={id_to_answer[pred_id]} | confidence={float(probs.max()):.3f}")


def plot_history(hist):
    if plt is None:
        return
    xs = range(1, len(hist["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, hist["train_loss"], label="train loss")
    plt.plot(xs, hist["eval_loss"], label="eval loss")
    plt.title("ARM on bAbI: Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.figure(figsize=(8, 5))
    plt.plot(xs, hist["train_acc"], label="train accuracy")
    plt.plot(xs, hist["eval_acc"], label="eval accuracy")
    plt.title("ARM on bAbI: Multi-hop Memory QA")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    base = download_babi(cfg)
    train_file, test_file = find_task_files(base, cfg.task_id)
    print("Using bAbI task:", cfg.task_id)
    print("Train file:", train_file.name)
    print("Test file:", test_file.name)
    train_rows = parse_babi_file(train_file, cfg.max_train)
    eval_rows = parse_babi_file(test_file, cfg.max_eval)
    print(f"Rows: train={len(train_rows)}, eval={len(eval_rows)}")
    all_answers = sorted({a for _, _, a in train_rows + eval_rows})
    answer_to_id = {a: i for i, a in enumerate(all_answers)}
    id_to_answer = {i: a for a, i in answer_to_id.items()}
    print("Answer classes:", all_answers)
    tokenizer = Tokenizer(cfg.max_vocab)
    tokenizer.fit([c + " " + q for c, q, _ in train_rows])
    train_ds = BabiDataset(train_rows, tokenizer, answer_to_id)
    eval_ds = BabiDataset(eval_rows, tokenizer, answer_to_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch, shuffle=False)
    model = ARMBabiQA(len(tokenizer.itos), len(all_answers)).to(device)
    print(model)
    run_tests(model, train_loader, len(all_answers))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    hist = {"train_loss": [], "train_acc": [], "eval_loss": [], "eval_acc": []}
    best = 0.0
    print("\nStarting training...")
    for ep in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = train_acc_sum = seen = 0.0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, acc = batch_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            bs = batch["label"].shape[0]
            train_loss_sum += loss.item() * bs
            train_acc_sum += acc * bs
            seen += bs
        sched.step()
        train_loss = train_loss_sum / max(1, seen)
        train_acc = train_acc_sum / max(1, seen)
        eval_loss, eval_acc = evaluate(model, eval_loader)
        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)
        hist["eval_loss"].append(eval_loss)
        hist["eval_acc"].append(eval_acc)
        if eval_acc > best:
            best = eval_acc
            torch.save({"model": model.state_dict(), "config": cfg.__dict__, "answer_to_id": answer_to_id, "id_to_answer": id_to_answer, "vocab": tokenizer.stoi, "history": hist, "best_eval_acc": best}, "arm_babi_checkpoint.pt")
        print(f"Epoch {ep:03d}/{cfg.epochs} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | eval_loss={eval_loss:.4f} | eval_acc={eval_acc:.4f}")
    print("\nBest bAbI eval accuracy:", round(best, 4))
    print("Checkpoint: arm_babi_checkpoint.pt")
    inspect(model, eval_ds, id_to_answer)
    plot_history(hist)


if __name__ == "__main__":
    main()
