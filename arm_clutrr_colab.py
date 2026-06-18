# ============================================================
# Algebraic Resonance Memory (ARM) on CLUTRR
# Colab runnable PyTorch script
# Concept direction: Godson Johnson
# ============================================================

import csv, math, os, re, random, ssl
from collections import Counter
from dataclasses import dataclass
from io import StringIO
from typing import Any, Dict, List, Tuple
from urllib.request import urlopen

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

try:
    from datasets import load_dataset, get_dataset_config_names
except Exception as exc:
    raise RuntimeError("Install dependencies first: pip install datasets matplotlib torch") from exc


def seed_all(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


@dataclass
class Config:
    dataset_name: str = "CLUTRR/v1"
    preferred_config: str = "gen_train234_test2to10"
    max_train: int = 12000
    max_eval: int = 3000
    max_vocab: int = 30000
    max_len: int = 192
    dim: int = 128
    emb: int = 128
    hidden: int = 128
    operators: int = 8
    tau: float = 0.45
    batch: int = 96
    epochs: int = 12
    lr: float = 2e-3
    wd: float = 1e-4
    cycle_w: float = 0.003
    op_w: float = 0.001
    clip: float = 1.0
    dropout: float = 0.15

cfg = Config()


# -----------------------------
# Dataset utilities
# -----------------------------

CLUTRR_RAW_BASE = "https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/main"
CLUTRR_CONFIGS = [
    "gen_train23_test2to10",
    "gen_train234_test2to10",
    "rob_train_clean_23_test_all_23",
    "rob_train_disc_23_test_all_23",
    "rob_train_irr_23_test_all_23",
    "rob_train_sup_23_test_all_23",
]


def read_url_text(url):
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = ssl.create_default_context()
    with urlopen(url, timeout=60, context=context) as response:
        return response.read().decode("utf-8")


def load_clutrr_csvs(config):
    if config not in CLUTRR_CONFIGS:
        raise ValueError(f"Unknown CLUTRR config {config!r}. Known configs: {CLUTRR_CONFIGS}")
    ds = {}
    for split in ["train", "validation", "test"]:
        url = f"{CLUTRR_RAW_BASE}/{config}/{split}.csv"
        text = read_url_text(url)
        ds[split] = list(csv.DictReader(StringIO(text)))
    print("Loaded CLUTRR CSV fallback config:", config)
    return ds


def load_clutrr():
    print("\nLoading CLUTRR from Hugging Face datasets...")
    errors, configs = [], []
    try:
        configs = get_dataset_config_names(cfg.dataset_name)
        print("Available configs:", configs[:10], "..." if len(configs) > 10 else "")
    except Exception as e:
        errors.append(str(e))
    candidates = [cfg.preferred_config] + [c for c in configs if c != cfg.preferred_config]
    if not candidates:
        candidates = [None]
    for c in candidates:
        try:
            ds = load_dataset(cfg.dataset_name) if c is None else load_dataset(cfg.dataset_name, c)
            print("Loaded config:", c)
            return ds, c
        except Exception as e:
            errors.append(f"{c}: {e}")
    if cfg.dataset_name.lower() in {"clutrr", "clutrr/v1"}:
        csv_candidates = [cfg.preferred_config] + [c for c in CLUTRR_CONFIGS if c != cfg.preferred_config]
        for c in csv_candidates:
            try:
                return load_clutrr_csvs(c), c
            except Exception as e:
                errors.append(f"csv fallback {c}: {e}")
    raise RuntimeError("Could not load CLUTRR. Last errors:\n" + "\n".join(errors[-6:]))


def pick(cols, names, required=True):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    for c in cols:
        for n in names:
            if n.lower() in c.lower():
                return c
    if required:
        raise KeyError(f"Could not find {names} in columns {cols}")
    return ""


def textify(x: Any) -> str:
    if isinstance(x, list): return " ".join(textify(v) for v in x)
    if isinstance(x, dict): return " ".join(f"{k} {textify(v)}" for k, v in x.items())
    return str(x)


def rows_from_split(split, story_col, query_col, label_col, limit):
    out, n = [], min(len(split), limit)
    for i in range(n):
        ex = split[i]
        story = textify(ex[story_col])
        query = textify(ex[query_col]) if query_col else ""
        label = textify(ex[label_col])
        out.append(((story + " [QUERY] " + query).strip(), label))
    return out


def prepare_rows(ds):
    names = list(ds.keys())
    train_name = "train" if "train" in ds else names[0]
    eval_name = next((n for n in ["validation", "val", "test"] if n in ds and n != train_name), None)
    if eval_name is None:
        eval_name = next((n for n in names if n != train_name), None)
    ex0 = ds[train_name][0]
    cols = list(ex0.keys())
    story_col = pick(cols, ["story", "clean_story", "text", "context", "sentence"])
    query_col = pick(cols, ["query", "question"], required=False)
    label_col = pick(cols, ["target_text", "target", "answer", "relation", "target_label", "label"])
    print("Columns:", {"story": story_col, "query": query_col or None, "label": label_col})
    train = rows_from_split(ds[train_name], story_col, query_col, label_col, cfg.max_train)
    if eval_name:
        ev = rows_from_split(ds[eval_name], story_col, query_col, label_col, cfg.max_eval)
        print(f"Splits: {train_name}={len(train)}, {eval_name}={len(ev)}")
    else:
        random.shuffle(train); cut = int(0.85 * len(train)); train, ev = train[:cut], train[cut:]
        print(f"Internal split: train={len(train)}, eval={len(ev)}")
    return train, ev


TOKEN = re.compile(r"[A-Za-z0-9_']+|[.,!?;:()\-]")

class Tokenizer:
    def __init__(self, max_vocab):
        self.stoi = {"[PAD]": 0, "[UNK]": 1}; self.itos = ["[PAD]", "[UNK]"]; self.max_vocab = max_vocab
    def tok(self, s): return TOKEN.findall(s.lower())
    def fit(self, texts):
        c = Counter(); [c.update(self.tok(t)) for t in texts]
        for w, _ in c.most_common(self.max_vocab - 2):
            if w not in self.stoi:
                self.stoi[w] = len(self.itos); self.itos.append(w)
        print("Vocab size:", len(self.itos))
    def encode(self, s, max_len):
        ids = [self.stoi.get(w, 1) for w in self.tok(s)[:max_len]]
        mask = [1] * len(ids)
        ids += [0] * (max_len - len(ids)); mask += [0] * (max_len - len(mask))
        return torch.tensor(ids), torch.tensor(mask, dtype=torch.float32)

class CLUTRRDataset(Dataset):
    def __init__(self, rows, tokenizer, label_to_id):
        self.rows, self.tok, self.label_to_id = rows, tokenizer, label_to_id
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        text, label = self.rows[i]
        ids, mask = self.tok.encode(text, cfg.max_len)
        return {"input_ids": ids.long(), "attention_mask": mask, "label": torch.tensor(self.label_to_id[label]).long()}


# -----------------------------
# ARM model
# -----------------------------

class AlgebraicResonanceMemory(nn.Module):
    def __init__(self, dim, memories, operators, tau):
        super().__init__(); self.dim, self.tau = dim, tau
        self.memory = nn.Parameter(torch.randn(memories, dim) / math.sqrt(dim))
        eye = torch.eye(dim).unsqueeze(0).repeat(operators, 1, 1)
        self.ops = nn.Parameter(eye + 0.02 * torch.randn(operators, dim, dim))
        self.bias = nn.Parameter(torch.zeros(operators, dim))
        self.metric_raw = nn.Parameter(torch.zeros(dim))
        self.cost_raw = nn.Parameter(torch.zeros(operators))
        self.qproj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.05), nn.Linear(dim, dim), nn.LayerNorm(dim))
    def metric(self): return F.softplus(self.metric_raw) + 1e-5
    def cost(self): return F.softplus(self.cost_raw)
    def forward(self, q):
        q = self.qproj(q)
        z = torch.einsum("bd,ked->bke", q, self.ops) + self.bias.unsqueeze(0)
        diff = z.unsqueeze(2) - self.memory.unsqueeze(0).unsqueeze(0)
        dist = (diff.square() * self.metric().view(1, 1, 1, -1)).sum(-1)
        path_scores = -dist / self.tau - self.cost().view(1, -1, 1)
        logits = torch.logsumexp(path_scores, dim=1)
        weights = F.softmax(logits, -1)
        return {"logits": logits, "weights": weights, "retrieved": weights @ self.memory, "path_scores": path_scores}
    def cycle_loss(self, order=4):
        Ap = torch.matrix_power(self.ops[0], order)
        return F.mse_loss(Ap, torch.eye(self.dim, device=Ap.device, dtype=Ap.dtype))
    def op_reg(self):
        I = torch.eye(self.dim, device=self.ops.device, dtype=self.ops.dtype)
        return (self.ops - I.unsqueeze(0)).square().mean() + 0.1 * self.bias.square().mean()

class AttentionMemory(nn.Module):
    def __init__(self, dim, memories):
        super().__init__(); self.dim = dim
        self.memory = nn.Parameter(torch.randn(memories, dim) / math.sqrt(dim))
        self.qproj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.scale = math.sqrt(dim)
    def forward(self, q):
        q = self.qproj(q)
        logits = q @ self.memory.t() / self.scale
        weights = F.softmax(logits, -1)
        return {"logits": logits, "weights": weights, "retrieved": weights @ self.memory}

class TextEncoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, cfg.emb, padding_idx=0)
        self.gru = nn.GRU(cfg.emb, cfg.hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.LayerNorm(cfg.hidden * 2), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden * 2, cfg.dim), nn.GELU(), nn.LayerNorm(cfg.dim))
    def forward(self, ids, mask):
        x = self.emb(ids); out, _ = self.gru(x); m = mask.unsqueeze(-1)
        pooled = (out * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.proj(pooled)

class ARMCLUTRR(nn.Module):
    def __init__(self, vocab, labels):
        super().__init__(); self.enc = TextEncoder(vocab); self.arm = AlgebraicResonanceMemory(cfg.dim, labels, cfg.operators, cfg.tau)
    def forward(self, ids, mask): return self.arm(self.enc(ids, mask))

class AttentionCLUTRR(nn.Module):
    def __init__(self, vocab, labels):
        super().__init__(); self.enc = TextEncoder(vocab); self.attn = AttentionMemory(cfg.dim, labels)
    def forward(self, ids, mask): return self.attn(self.enc(ids, mask))


def batch_loss(model, batch, kind="arm"):
    ids, mask, y = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
    out = model(ids, mask)
    ce = F.cross_entropy(out["logits"], y)
    loss = ce
    if kind == "arm":
        loss = loss + cfg.cycle_w * model.arm.cycle_loss(4) + cfg.op_w * model.arm.op_reg()
    with torch.no_grad(): acc = (out["logits"].argmax(-1) == y).float().mean().item()
    return loss, acc

@torch.no_grad()
def evaluate(model, loader, kind="arm"):
    model.eval(); tot_l = tot_a = n = 0
    for b in loader:
        loss, acc = batch_loss(model, b, kind); bs = b["label"].shape[0]
        tot_l += loss.item() * bs; tot_a += acc * bs; n += bs
    return tot_l / max(1, n), tot_a / max(1, n)


def run_tests(model, loader, kind="arm"):
    print(f"\nRunning {kind} tests...")
    b = next(iter(loader)); out = model(b["input_ids"].to(device), b["attention_mask"].to(device)); bs = b["label"].shape[0]
    assert out["logits"].shape[0] == bs and out["weights"].shape == out["logits"].shape
    assert out["retrieved"].shape == (bs, cfg.dim)
    if kind == "arm":
        assert out["path_scores"].shape[0] == bs
    loss, _ = batch_loss(model, b, kind); assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None: assert torch.isfinite(p.grad).all(), name
    model.zero_grad(set_to_none=True); print(f"{kind} tests passed.")


def make_model(kind, vocab, labels):
    if kind == "arm":
        return ARMCLUTRR(vocab, labels)
    if kind == "attention":
        return AttentionCLUTRR(vocab, labels)
    raise ValueError(kind)


def train_model(kind, train_loader, eval_loader, vocab, labels, used_config, label_to_id, id_to_label, tok):
    seed_all(42)
    model = make_model(kind, vocab, labels).to(device); print(model); run_tests(model, train_loader, kind)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    hist = {"train_loss": [], "train_acc": [], "eval_loss": [], "eval_acc": []}; best = -1.0
    checkpoint = f"{kind}_clutrr_checkpoint.pt"
    for ep in range(1, cfg.epochs + 1):
        model.train(); tl = ta = n = 0
        for b in train_loader:
            opt.zero_grad(set_to_none=True); loss, acc = batch_loss(model, b, kind); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip); opt.step()
            bs = b["label"].shape[0]; tl += loss.item() * bs; ta += acc * bs; n += bs
        sched.step(); tr_l, tr_a = tl / max(1, n), ta / max(1, n); ev_l, ev_a = evaluate(model, eval_loader, kind)
        hist["train_loss"].append(tr_l); hist["train_acc"].append(tr_a); hist["eval_loss"].append(ev_l); hist["eval_acc"].append(ev_a)
        if ev_a > best:
            best = ev_a; torch.save({"model": model.state_dict(), "kind": kind, "config": cfg.__dict__, "dataset_config": used_config, "label_to_id": label_to_id, "id_to_label": id_to_label, "vocab": tok.stoi, "history": hist, "best_eval_acc": best}, checkpoint)
        print(f"{kind:9s} Epoch {ep:03d}/{cfg.epochs} | train_loss={tr_l:.4f} | train_acc={tr_a:.4f} | eval_loss={ev_l:.4f} | eval_acc={ev_a:.4f}")
    print(f"\nBest {kind} CLUTRR eval accuracy:", round(best, 4)); print("Checkpoint:", checkpoint)
    return model, hist, best, checkpoint


def main():
    raw, used_config = load_clutrr(); train_rows, eval_rows = prepare_rows(raw)
    labels = sorted({y for _, y in train_rows + eval_rows}); label_to_id = {y: i for i, y in enumerate(labels)}; id_to_label = {i: y for y, i in label_to_id.items()}
    print("Relation labels:", labels)
    tok = Tokenizer(cfg.max_vocab); tok.fit([x for x, _ in train_rows])
    train_ds, eval_ds = CLUTRRDataset(train_rows, tok, label_to_id), CLUTRRDataset(eval_rows, tok, label_to_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, drop_last=True); eval_loader = DataLoader(eval_ds, batch_size=cfg.batch)
    results = {}
    for kind in ["attention", "arm"]:
        print(f"\n=== Training {kind} model ===")
        model, hist, best, checkpoint = train_model(kind, train_loader, eval_loader, len(tok.itos), len(labels), used_config, label_to_id, id_to_label, tok)
        results[kind] = {"model": model, "history": hist, "best": best, "checkpoint": checkpoint}
    print("\nComparison summary:")
    for kind, result in results.items():
        final_acc = result["history"]["eval_acc"][-1]
        print(f"{kind:9s} | best_eval_acc={result['best']:.4f} | final_eval_acc={final_acc:.4f} | checkpoint={result['checkpoint']}")
    inspect(results["arm"]["model"], eval_ds, id_to_label); plot_compare({k: v["history"] for k, v in results.items()})

@torch.no_grad()
def inspect(model, ds, id_to_label, n=10):
    model.eval(); print("\nExample predictions:")
    for i in range(min(n, len(ds))):
        item = ds[i]; out = model(item["input_ids"].unsqueeze(0).to(device), item["attention_mask"].unsqueeze(0).to(device)); pred = int(out["logits"].argmax(-1).item()); true = int(item["label"].item()); conf = float(out["weights"].max().item())
        print(f"sample={i:02d} | true={id_to_label[true]} | pred={id_to_label[pred]} | confidence={conf:.3f}")

def plot(hist):
    if plt is None: return
    xs = range(1, len(hist["train_loss"]) + 1)
    plt.figure(figsize=(8, 5)); plt.plot(xs, hist["train_loss"], label="train loss"); plt.plot(xs, hist["eval_loss"], label="eval loss"); plt.title("ARM on CLUTRR: Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.legend(); plt.show()
    plt.figure(figsize=(8, 5)); plt.plot(xs, hist["train_acc"], label="train acc"); plt.plot(xs, hist["eval_acc"], label="eval acc"); plt.title("ARM on CLUTRR: Kinship Retrieval"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(True); plt.legend(); plt.show()

def plot_compare(histories):
    if plt is None: return
    plt.figure(figsize=(8, 5))
    for kind, hist in histories.items():
        xs = range(1, len(hist["eval_acc"]) + 1)
        plt.plot(xs, hist["eval_acc"], label=f"{kind} eval acc")
    plt.title("CLUTRR: ARM vs Attention"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(True); plt.legend(); plt.show()

if __name__ == "__main__":
    main()
