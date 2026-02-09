from __future__ import annotations

"""
Scoring utilities: WER/CER/InsRate and Tail/GAP Non-Empty Rate.

The project already contains metrics modules, but experiments use these small, explicit
implementations to keep the evaluation logic easy to audit and debug.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json


# -----------------------------
# Text normalization
# -----------------------------

def _default_normalize(s: str) -> str:
    import re
    s = s.lower()
    # keep apostrophes inside words; remove other punctuation
    s = re.sub(r"[^\w\s']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize(s: str) -> str:
    """
    Use the repo's shared normalizer if available; else fall back to a conservative normalizer.
    """
    try:
        from asr_mvp.text import normalize_text  # type: ignore
        return normalize_text(s)
    except Exception:
        return _default_normalize(s)


# -----------------------------
# Edit distance with S/D/I
# -----------------------------

@dataclass
class EditCounts:
    sub: int
    delete: int
    insert: int
    ref_len: int

    @property
    def wer(self) -> float:
        return (self.sub + self.delete + self.insert) / max(1, self.ref_len)

    @property
    def ins_rate(self) -> float:
        return self.insert / max(1, self.ref_len)


def _edit_counts(ref: List[str], hyp: List[str]) -> EditCounts:
    """
    Compute S/D/I counts under minimum edit distance alignment (Levenshtein).
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = min edits for ref[:i], hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]  # "ok", "sub", "del", "ins"
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins"
    back[0][0] = "ok"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                best = dp[i - 1][j - 1]
                op = "ok"
            else:
                best = dp[i - 1][j - 1] + 1
                op = "sub"

            d = dp[i - 1][j] + 1
            if d < best:
                best = d
                op = "del"

            ins = dp[i][j - 1] + 1
            if ins < best:
                best = ins
                op = "ins"

            dp[i][j] = best
            back[i][j] = op

    # backtrace for counts
    i, j = n, m
    sub = delete = insert = 0
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "ok":
            i -= 1
            j -= 1
        elif op == "sub":
            sub += 1
            i -= 1
            j -= 1
        elif op == "del":
            delete += 1
            i -= 1
        elif op == "ins":
            insert += 1
            j -= 1
        else:
            # should not happen
            break

    return EditCounts(sub=sub, delete=delete, insert=insert, ref_len=n)


def wer_counts(ref_text: str, hyp_text: str) -> EditCounts:
    ref = normalize(ref_text).split()
    hyp = normalize(hyp_text).split()
    return _edit_counts(ref, hyp)


def cer_counts(ref_text: str, hyp_text: str) -> EditCounts:
    ref = list(normalize(ref_text).replace(" ", ""))
    hyp = list(normalize(hyp_text).replace(" ", ""))
    return _edit_counts(ref, hyp)


# -----------------------------
# Aggregation
# -----------------------------

@dataclass
class AggregateScores:
    wer: float
    cer: float
    ins_rate: float
    n_examples: int
    total_ref_words: int
    total_ref_chars: int
    total_S: int
    total_D: int
    total_I: int


def aggregate_scores(refs: List[str], hyps: List[str]) -> AggregateScores:
    assert len(refs) == len(hyps)
    total_words = 0
    total_chars = 0
    S = D = I = 0

    # WER counts
    for r, h in zip(refs, hyps):
        c = wer_counts(r, h)
        total_words += c.ref_len
        S += c.sub
        D += c.delete
        I += c.insert

    wer = (S + D + I) / max(1, total_words)
    ins_rate = I / max(1, total_words)

    # CER counts
    Sc = Dc = Ic = 0
    for r, h in zip(refs, hyps):
        c = cer_counts(r, h)
        total_chars += c.ref_len
        Sc += c.sub
        Dc += c.delete
        Ic += c.insert
    cer = (Sc + Dc + Ic) / max(1, total_chars)

    return AggregateScores(
        wer=float(wer),
        cer=float(cer),
        ins_rate=float(ins_rate),
        n_examples=len(refs),
        total_ref_words=int(total_words),
        total_ref_chars=int(total_chars),
        total_S=int(S),
        total_D=int(D),
        total_I=int(I),
    )


# -----------------------------
# Non-speech probing
# -----------------------------

def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_nonspeech_refs(ds_dir: Path) -> List[dict]:
    """
    Returns list of rows describing non-speech segments to probe.
    Expected file:
      ds_dir/nonspeech_refs.jsonl
    Rows:
      {"id": "...", "audio_path": "audio/<...>.wav", "kind": "tail"|"gap", "duration_sec": float}
    """
    p = ds_dir / "nonspeech_refs.jsonl"
    if not p.exists():
        return []
    return list(read_jsonl(p))


def nonempty_rate(hyps: List[str]) -> float:
    if not hyps:
        return 0.0
    ne = 0
    for h in hyps:
        if normalize(h) != "":
            ne += 1
    return ne / float(len(hyps))
