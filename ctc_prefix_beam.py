from __future__ import annotations

"""
CTC prefix beam search (LM-optional).

Main ISO experiments are LM-free (lm_path=None), but Conformer+CTC uses beam decoding.
NeMo provides built-in CTC beam decoding for many versions; this module remains as a
portable fallback and as a reference implementation.

Inputs:
- logp: (T, V) log-probabilities (natural log) over vocabulary including blank.
- vocab: list mapping token_id -> string (or single-character). For BPE tokens, these
  are decoded by simple concatenation; you can also pass a custom joiner.

Algorithm: standard prefix beam search tracking (p_blank, p_nonblank) for each prefix.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import math
import numpy as np


def logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


@dataclass(frozen=True)
class BeamSearchConfig:
    beam_size: int = 32
    blank_id: int = 0
    token_prune_topk: int = 0  # 0 => no prune
    lm_weight: float = 0.0
    # Optional LM with method score_sentence(text)->logprob (e.g., NGramLM)
    lm: Optional[object] = None
    # Optional token join function: tokens(list[str]) -> str
    joiner: Optional[Callable[[List[str]], str]] = None


def _default_joiner(tokens: List[str]) -> str:
    # For BPE-like tokens, simple concatenation is usually correct if tokens include spaces.
    return "".join(tokens).replace("▁", " ").strip()


def ctc_prefix_beam_search(logp: np.ndarray, vocab: List[str], cfg: BeamSearchConfig) -> str:
    """
    Return best decoded string.
    """
    T, V = logp.shape
    blank = int(cfg.blank_id)
    joiner = cfg.joiner or _default_joiner

    # beams: prefix(tuple token ids) -> (p_b, p_nb)
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {tuple(): (0.0, -math.inf)}  # log(1), log(0)

    for t in range(T):
        next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}

        # token pruning to reduce compute
        if cfg.token_prune_topk and cfg.token_prune_topk > 0 and cfg.token_prune_topk < V:
            topk = int(cfg.token_prune_topk)
            idx = np.argpartition(-logp[t], topk)[:topk]
            token_ids = idx.tolist()
        else:
            token_ids = list(range(V))

        for prefix, (p_b, p_nb) in beams.items():
            p_total = logsumexp(p_b, p_nb)

            # extend with blank
            pb2, pnb2 = next_beams.get(prefix, (-math.inf, -math.inf))
            pb2 = logsumexp(pb2, p_total + float(logp[t, blank]))
            next_beams[prefix] = (pb2, pnb2)

            for c in token_ids:
                if c == blank:
                    continue
                lp = float(logp[t, c])
                end = prefix[-1] if prefix else None

                if c == end:
                    # if repeating last token, only nonblank->nonblank (stay) and blank->nonblank (new)
                    pb2, pnb2 = next_beams.get(prefix, (-math.inf, -math.inf))
                    # stay in same prefix
                    pnb2 = logsumexp(pnb2, p_nb + lp)
                    next_beams[prefix] = (pb2, pnb2)

                    # extend prefix (from blank) yields same prefix under CTC collapse
                    new_pref = prefix + (c,)
                    pb3, pnb3 = next_beams.get(new_pref, (-math.inf, -math.inf))
                    pnb3 = logsumexp(pnb3, p_b + lp)
                    next_beams[new_pref] = (pb3, pnb3)
                else:
                    new_pref = prefix + (c,)
                    pb3, pnb3 = next_beams.get(new_pref, (-math.inf, -math.inf))
                    pnb3 = logsumexp(pnb3, p_total + lp)
                    next_beams[new_pref] = (pb3, pnb3)

        # prune to beam_size by total prob (+ optional LM)
        scored: List[Tuple[Tuple[int, ...], float]] = []
        for pref, (pb, pnb) in next_beams.items():
            total = logsumexp(pb, pnb)
            if cfg.lm is not None and cfg.lm_weight and pref:
                text = joiner([vocab[i] for i in pref])
                try:
                    total += float(cfg.lm_weight) * float(cfg.lm.score_sentence(text))  # type: ignore
                except Exception:
                    pass
            scored.append((pref, total))
        scored.sort(key=lambda x: x[1], reverse=True)
        beams = {p: next_beams[p] for p, _ in scored[: int(cfg.beam_size)]}

    # best
    best_pref = max(beams.items(), key=lambda kv: logsumexp(kv[1][0], kv[1][1]))[0]
    return joiner([vocab[i] for i in best_pref])
