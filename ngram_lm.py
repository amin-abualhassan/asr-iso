from __future__ import annotations

"""
A tiny n-gram language model implementation (word-level) used optionally by CTC beam decoding.

Important for this ISO:
- Main experiments are LM-free (lm_path=None).
- This module remains here for optional ablations / future work.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import math
import json
from pathlib import Path

from .scoring import normalize


@dataclass
class NGramLM:
    n: int
    # counts[(w_{i-n+1}, ..., w_{i-1})][w_i] = count
    counts: Dict[Tuple[str, ...], Dict[str, int]]
    context_totals: Dict[Tuple[str, ...], int]
    vocab: Dict[str, int]
    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    add_k: float = 0.1  # smoothing

    def log_prob_next(self, context: Tuple[str, ...], word: str) -> float:
        ctx = context[-(self.n - 1) :] if self.n > 1 else tuple()
        if word not in self.vocab:
            word = self.unk_token
        dist = self.counts.get(ctx)
        total = self.context_totals.get(ctx, 0)
        V = max(1, len(self.vocab))
        c = 0 if dist is None else dist.get(word, 0)
        p = (c + self.add_k) / (total + self.add_k * V)
        return math.log(p)

    def score_sentence(self, text: str) -> float:
        words = normalize(text).split()
        toks = [self.bos_token] * (self.n - 1) + words + [self.eos_token]
        s = 0.0
        for i in range(self.n - 1, len(toks)):
            ctx = tuple(toks[i - (self.n - 1) : i]) if self.n > 1 else tuple()
            s += self.log_prob_next(ctx, toks[i])
        return float(s)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "n": self.n,
            "counts": { " ".join(k): v for k, v in self.counts.items() },
            "context_totals": { " ".join(k): v for k, v in self.context_totals.items() },
            "vocab": self.vocab,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "add_k": self.add_k,
        }
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "NGramLM":
        obj = json.loads(path.read_text(encoding="utf-8"))
        counts = { tuple(k.split()) if k else tuple(): v for k, v in obj["counts"].items() }
        totals = { tuple(k.split()) if k else tuple(): int(v) for k, v in obj["context_totals"].items() }
        return NGramLM(
            n=int(obj["n"]),
            counts=counts,
            context_totals=totals,
            vocab={k: int(v) for k, v in obj["vocab"].items()},
            unk_token=str(obj.get("unk_token", "<unk>")),
            bos_token=str(obj.get("bos_token", "<s>")),
            eos_token=str(obj.get("eos_token", "</s>")),
            add_k=float(obj.get("add_k", 0.1)),
        )


def train_ngram(texts: Iterable[str], n: int = 3, add_k: float = 0.1) -> NGramLM:
    from collections import defaultdict

    counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: Dict[Tuple[str, ...], int] = defaultdict(int)
    vocab: Dict[str, int] = {}

    bos = "<s>"
    eos = "</s>"
    unk = "<unk>"

    for t in texts:
        words = normalize(t).split()
        toks = [bos] * (n - 1) + words + [eos]
        for w in words:
            vocab[w] = vocab.get(w, 0) + 1

        for i in range(n - 1, len(toks)):
            ctx = tuple(toks[i - (n - 1) : i]) if n > 1 else tuple()
            nxt = toks[i]
            counts[ctx][nxt] += 1
            totals[ctx] += 1

    vocab[unk] = vocab.get(unk, 0) + 1
    vocab[bos] = vocab.get(bos, 0) + 1
    vocab[eos] = vocab.get(eos, 0) + 1

    return NGramLM(n=n, counts=dict(counts), context_totals=dict(totals), vocab=vocab, add_k=float(add_k))
