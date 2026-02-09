from __future__ import annotations

"""
Parameter-efficient fine-tuning (LoRA) for OpenAI Whisper (PyTorch).

Used by adapt_whisper_lora.py to implement the adaptation protocol:
LoRA adaptation on speaker-change proxy concatenations, validated on a held-out
split with early stopping on WER (and optionally a non-speech non-empty probe).

Notes:
- Targets the `openai-whisper` PyTorch package (`pip install -U openai-whisper`).
- It does NOT fine-tune the CTranslate2/faster-whisper runtime directly.
- Written for readability + debuggability.

Outputs:
- best_lora.pt : a small "adapter" checkpoint containing ONLY LoRA weights + metadata
- manifest.json / best_metrics.json : run bookkeeping
"""

from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .jsonl_dataset import load_dataset_dir
from .scoring import aggregate_scores, nonempty_rate


# -----------------------------
# Helpers
# -----------------------------

def _normalize_openai_whisper_name(name: str) -> str:
    """
    openai-whisper expects names like: 'small.en', 'large-v3', etc.
    But we sometimes pass HF-style names like: 'openai/whisper-small.en'.
    This maps HF-style -> openai-whisper style.
    """
    n = (name or "").strip()
    if n.startswith("openai/whisper-"):
        n = n[len("openai/whisper-"):]
    if n.startswith("whisper-"):
        n = n[len("whisper-"):]
    return n


def _resolve_audio_path(ds_dir: Path, p: Path) -> Path:
    """Resolve relative audio paths against a dataset directory."""
    if p.is_absolute():
        return p
    cand = (ds_dir / p).resolve()
    return cand


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False)


# -----------------------------
# LoRA building blocks
# -----------------------------

class LoRALinear(nn.Module):
    """
    Wrap an existing nn.Linear with a trainable low-rank update:

        y = base(x) + scale * (drop(x) @ A^T @ B^T)

    where:
      A: (r, in_features)
      B: (out_features, r)

    Base weights are frozen; only A,B are trainable.
    """
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.r)
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        # Freeze base parameters
        for p in self.base.parameters():
            p.requires_grad = False

        in_f = int(base.in_features)
        out_f = int(base.out_features)

        # Force fp32 LoRA weights (avoids AMP dtype mismatches).
        dev = base.weight.device
        dt = torch.float32
        self.A = nn.Parameter(torch.zeros((self.r, in_f), device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros((out_f, self.r), device=dev, dtype=dt))

        # Init (common LoRA practice: A random, B zeros -> starts as no-op)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)

        # LoRA branch: compute in A's dtype (fp32), then cast to y dtype (fp16 under autocast)
        x2 = self.dropout(x).to(dtype=self.A.dtype)
        z = torch.matmul(x2, self.A.t())              # (..., r)
        dz = torch.matmul(z, self.B.t()) * self.scale # (..., out)
        return y + dz.to(dtype=y.dtype)


@dataclass(frozen=True)
class LoRAConfig:
    # low-rank update size
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0

    # If scope_substrings is non-empty, we ONLY apply LoRA to Linear layers whose module
    # name contains one of these substrings (e.g. "encoder.blocks", "decoder.blocks").
    scope_substrings: Tuple[str, ...] = ()

    # Additionally, if target_substrings is non-empty, the Linear module name must also
    # contain one of these substrings. If empty, we match ALL Linear layers within scope.
    target_substrings: Tuple[str, ...] = ("query", "key", "value", "out", "fc", "proj", "mlp")


def apply_lora(model: nn.Module, cfg: LoRAConfig) -> List[str]:
    """
    Replace selected nn.Linear modules with LoRALinear wrappers.
    Returns a list of replaced module names.
    """
    replaced: List[str] = []

    def _iter_named_parents(m: nn.Module, prefix: str = ""):
        for name, child in m.named_children():
            full = f"{prefix}.{name}" if prefix else name
            yield m, name, child, full
            yield from _iter_named_parents(child, full)

    for parent, attr, child, full_name in _iter_named_parents(model):
        if not isinstance(child, nn.Linear):
            continue

        n = full_name.lower()
        scope_ok = True
        if cfg.scope_substrings:
            scope_ok = any(s.lower() in n for s in cfg.scope_substrings)

        target_ok = True
        if cfg.target_substrings:
            target_ok = any(s.lower() in n for s in cfg.target_substrings)

        if scope_ok and target_ok:
            setattr(parent, attr, LoRALinear(child, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout))
            replaced.append(full_name)

    return replaced


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params.extend([m.A, m.B])
    return params


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Return ONLY LoRA weights from the model state dict.
    (Keeps checkpoints small and backend-agnostic.)
    """
    sd = model.state_dict()
    return {k: v.detach().cpu() for k, v in sd.items() if k.endswith(".A") or k.endswith(".B")}


def load_lora_adapter(model: nn.Module, adapter_path: Path, device: torch.device) -> Dict[str, Any]:
    """
    Load a LoRA adapter checkpoint saved by this trainer (best_lora.pt).
    Returns metadata dict and prints helpful debug info.
    """
    obj = torch.load(str(adapter_path), map_location="cpu")
    if isinstance(obj, dict) and "lora_state" in obj:
        state = obj["lora_state"]
        meta = {k: v for k, v in obj.items() if k != "lora_state"}
    else:
        # allow passing a raw state-dict file for convenience
        state = obj
        meta = {}

    missing, unexpected = model.load_state_dict(state, strict=False)

    # filter out base weights missing (expected) vs lora missing (not expected)
    lora_missing = [k for k in missing if k.endswith(".A") or k.endswith(".B")]

    print(f"[lora] loaded adapter: {adapter_path}")
    print(f"[lora] missing_keys={len(missing)} (lora_missing={len(lora_missing)}), unexpected_keys={len(unexpected)}")
    if lora_missing:
        print(f"[lora] WARNING: some LoRA keys were missing (first 10): {lora_missing[:10]}")
    if unexpected:
        print(f"[lora] WARNING: unexpected keys (first 10): {unexpected[:10]}")

    return meta


# -----------------------------
# Whisper dataset utilities
# -----------------------------

def _load_audio_mel(path: Path, device: torch.device) -> torch.Tensor:
    import whisper  # openai-whisper

    # load and resample to 16k
    audio = whisper.load_audio(str(path))

    # IMPORTANT: pad/trim the waveform to the 30s window first
    audio = whisper.pad_or_trim(audio)

    # then compute log-mel (80 x 3000 frames for 30s audio)
    mel = whisper.log_mel_spectrogram(audio)

    return mel.to(device)


def _tokenize_text(text: str, tokenizer) -> List[int]:
    ids = tokenizer.encode(text)
    return list(tokenizer.sot_sequence) + ids + [tokenizer.eot]


@dataclass
class Batch:
    mel: torch.Tensor        # (B, 80, T)
    tokens_in: torch.Tensor  # (B, L)
    targets: torch.Tensor    # (B, L)


# def _collate(batch: List[Tuple[torch.Tensor, List[int]]], pad_id: int, device: torch.device) -> Batch:
#     mels = torch.stack([b[0] for b in batch], dim=0)
#     lengths = [len(b[1]) for b in batch]
#     L = max(lengths)
#     tok = torch.full((len(batch), L), pad_id, dtype=torch.long)
#     tgt = torch.full((len(batch), L), -100, dtype=torch.long)
#     for i, (_, ids) in enumerate(batch):
#         tok[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
#         # teacher forcing: predict next token
#         if len(ids) >= 2:
#             tgt[i, 1 : len(ids)] = torch.tensor(ids[1:], dtype=torch.long)
#     return Batch(mel=mels.to(device), tokens_in=tok.to(device), targets=tgt.to(device))

def _collate(batch: List[Tuple[torch.Tensor, List[int]]], pad_id: int, device: torch.device) -> Batch:
    mels = torch.stack([b[0] for b in batch], dim=0)

    # inputs are ids[:-1], targets are ids[1:]
    ins = [b[1][:-1] for b in batch]
    tgts = [b[1][1:] for b in batch]
    lengths = [len(x) for x in ins]
    L = max(lengths)

    tok = torch.full((len(batch), L), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), L), -100, dtype=torch.long)

    for i, (ids_in, ids_tgt) in enumerate(zip(ins, tgts)):
        tok[i, : len(ids_in)] = torch.tensor(ids_in, dtype=torch.long)
        tgt[i, : len(ids_tgt)] = torch.tensor(ids_tgt, dtype=torch.long)

    return Batch(mel=mels.to(device), tokens_in=tok.to(device), targets=tgt.to(device))



# -----------------------------
# Training / evaluation
# -----------------------------

@dataclass(frozen=True)
class WhisperLoRATrainConfig:
    train_dir: Path
    dev_dir: Path
    dev_gap_dir: Optional[Path] = None
    out_dir: Path = Path("results/whisper_lora")

    whisper_name: str = "small.en"
    device: str = "cuda"

    lr: float = 5e-5
    weight_decay: float = 0.0
    batch_size: int = 4
    max_steps: int = 2000
    eval_every: int = 200
    patience: int = 5

    seed: int = 70072
    fp16: bool = True
    lora: LoRAConfig = LoRAConfig()


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@torch.no_grad()
def decode_openai_whisper(
    model,
    tokenizer,
    audio_paths: List[Path],
    device: torch.device,
    temperature: float = 0.0,
    beam_size: int = 5,
    condition_on_previous_text: bool = False,
) -> List[str]:
    import whisper
    import inspect

    outs: List[str] = []

    base_kwargs = {
        "language": "en",
        "task": "transcribe",
        "without_timestamps": True,
        "fp16": False,
        "temperature": float(temperature),
        # This kwarg is not supported in some whisper versions; we will filter it below.
        "condition_on_previous_text": bool(condition_on_previous_text),
    }

    if float(temperature) == 0.0:
        base_kwargs["beam_size"] = int(beam_size)

    # Filter kwargs based on what this installed whisper supports.
    sig = inspect.signature(whisper.DecodingOptions.__init__)
    base_kwargs = {k: v for k, v in base_kwargs.items() if k in sig.parameters}

    opts = whisper.DecodingOptions(**base_kwargs)

    for p in audio_paths:
        mel = _load_audio_mel(p, device)
        r = whisper.decode(model, mel, opts)
        outs.append((r.text or "").strip())

    return outs



def train_whisper_lora(cfg: WhisperLoRATrainConfig) -> Path:
    _set_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    log_path = cfg.out_dir / "train.log"
    # fresh log each run
    log_path.write_text("", encoding="utf-8")

    def _log(msg: str) -> None:
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _write_pairs_report(
        *,
        tag: str,
        refs: List[str],
        hyps: List[str],
        out_dir: Path,
        topk: int,
        sort_key,
        filename_prefix: str,
        header: str,
    ) -> Path:
        pairs = list(zip(refs, hyps))
        pairs.sort(key=sort_key, reverse=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        dbg_dir = out_dir / "debug"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        path = dbg_dir / f"{filename_prefix}_{tag}_{ts}.txt"

        with path.open("w", encoding="utf-8") as f:
            f.write(f"[debug] tag={tag}  n={len(pairs)}  topk={topk}\n")
            f.write(header + "\n\n")
            for i, (r, h) in enumerate(pairs[:topk], start=1):
                r_words = len(r.split())
                h_words = len(h.split())
                f.write(f"#{i}\n")
                f.write(f"ref_words={r_words} hyp_words={h_words}  (diff={h_words - r_words})\n")
                f.write(f"REF: {r}\n")
                f.write(f"HYP: {h}\n")
                f.write("-" * 80 + "\n")

        _log(f"[debug] wrote: {path}")
        return path

    def _dump_debug_views(tag: str, refs: List[str], hyps: List[str], out_dir: Path, topk: int = 25) -> None:
        # 1) longest hypotheses
        _write_pairs_report(
            tag=tag,
            refs=refs,
            hyps=hyps,
            out_dir=out_dir,
            topk=topk,
            sort_key=lambda x: len(x[1].split()),
            filename_prefix="debug_longest_hyps",
            header="Sorted by hyp word count (descending)",
        )
        # 2) worst insertion explosions (largest hyp-ref)
        _write_pairs_report(
            tag=tag,
            refs=refs,
            hyps=hyps,
            out_dir=out_dir,
            topk=topk,
            sort_key=lambda x: (len(x[1].split()) - len(x[0].split())),
            filename_prefix="debug_worst_worddiff",
            header="Sorted by (hyp_words - ref_words) (descending)",
        )

    try:
        import whisper
        from whisper.tokenizer import get_tokenizer
    except Exception as e:
        raise RuntimeError(
            "LoRA training requires the `openai-whisper` PyTorch package.\n"
            "Install with: pip install -U openai-whisper\n"
            f"Original error: {e}"
        )

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    use_amp = bool(cfg.fp16 and device.type == "cuda")

    # Tokenizer (needed before dataset tokenization + decoding)
    tokenizer = get_tokenizer(multilingual=False, language="en", task="transcribe")
    pad_id = tokenizer.eot

    # Load datasets
    train_exs = load_dataset_dir(cfg.train_dir)
    dev_exs = load_dataset_dir(cfg.dev_dir)
    dev_gap_exs = load_dataset_dir(cfg.dev_gap_dir) if cfg.dev_gap_dir else []

    # Precompute lists
    train_items = [
        (_resolve_audio_path(cfg.train_dir, e.audio_path), _tokenize_text(e.text, tokenizer))
        for e in train_exs
    ]
    dev_items = [(_resolve_audio_path(cfg.dev_dir, e.audio_path), e.text) for e in dev_exs]
    dev_gap_paths = (
        [_resolve_audio_path(cfg.dev_gap_dir, e.audio_path) for e in dev_gap_exs]
        if cfg.dev_gap_dir
        else []
    )

    rng = torch.Generator().manual_seed(int(cfg.seed))

    # Load base model
    model = whisper.load_model(_normalize_openai_whisper_name(cfg.whisper_name), device=str(device))

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Common dev lists
    dev_paths = [p for (p, _) in dev_items]
    dev_refs = [t for (_, t) in dev_items]

    # ---- baseline sanity eval (step0_base) ----
    model.eval()
    dev_hyps = decode_openai_whisper(model, tokenizer, dev_paths, device=device, temperature=0.0, beam_size=5)
    _dump_debug_views(tag="step0_base", refs=dev_refs, hyps=dev_hyps, out_dir=cfg.out_dir, topk=25)
    dev_scores = aggregate_scores(dev_refs, dev_hyps)
    _log(f"[eval step0_base] dev WER={dev_scores.wer:.4f} CER={dev_scores.cer:.4f} InsRate={dev_scores.ins_rate:.4f}")

    # Apply LoRA once
    replaced = apply_lora(model, cfg.lora)
    params = lora_parameters(model)
    if not params:
        raise RuntimeError(
            "No LoRA parameters were created.\n"
            "Check LoRAConfig.scope_substrings/target_substrings and that they match Whisper module names."
        )
    for p in params:
        p.requires_grad = True

    # ---- sanity eval after LoRA injection but before training (step0_lora_init) ----
    model.eval()
    dev_hyps = decode_openai_whisper(model, tokenizer, dev_paths, device=device, temperature=0.0, beam_size=5)
    _dump_debug_views(tag="step0_lora_init", refs=dev_refs, hyps=dev_hyps, out_dir=cfg.out_dir, topk=25)
    dev_scores = aggregate_scores(dev_refs, dev_hyps)
    _log(f"[eval step0_lora_init] dev WER={dev_scores.wer:.4f} CER={dev_scores.cer:.4f} InsRate={dev_scores.ins_rate:.4f}")

    model.train()

    # Optimizer + scaler
    opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"wer": float("inf"), "nonempty": float("inf"), "step": -1}
    bad_epochs = 0

    # debug: check LoRA actually updates
    with torch.no_grad():
        a0 = params[0].detach().float().clone()

    def _sample_batch() -> Batch:
        idx = torch.randint(low=0, high=len(train_items), size=(cfg.batch_size,), generator=rng).tolist()
        batch = []
        for i in idx:
            p, ids = train_items[i]
            mel = _load_audio_mel(p, device)
            batch.append((mel, ids))
        return _collate(batch, pad_id=pad_id, device=device)

    t0 = time.time()
    _log(f"[lora] replaced Linear modules: {len(replaced)}")
    if len(replaced) <= 20:
        for n in replaced:
            _log(f"  - {n}")
    else:
        _log(f"  (first 10) {replaced[:10]}")

    for step in range(1, int(cfg.max_steps) + 1):
        b = _sample_batch()
        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(b.mel, b.tokens_in)  # (B, L, vocab)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                b.targets.view(-1),
                ignore_index=-100,
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        if step % 50 == 0:
            with torch.no_grad():
                dA = (params[0].detach().float() - a0).abs().mean().item()
            g = params[0].grad
            gmean = float(g.detach().abs().mean().item()) if g is not None else 0.0
            _log(f"[step {step}] loss={loss.detach().float().item():.4f}  mean|ΔA|={dA:.3e}  mean|grad(A)|={gmean:.3e}")

        if step % int(cfg.eval_every) == 0 or step == int(cfg.max_steps):
            model.eval()

            dev_hyps = decode_openai_whisper(model, tokenizer, dev_paths, device=device, temperature=0.0, beam_size=5)
            _dump_debug_views(tag=f"step{step}", refs=dev_refs, hyps=dev_hyps, out_dir=cfg.out_dir, topk=25)
            dev_scores = aggregate_scores(dev_refs, dev_hyps)

            gap_nonempty = None
            if dev_gap_paths:
                gap_hyps = decode_openai_whisper(model, tokenizer, dev_gap_paths, device=device, temperature=0.0, beam_size=5)
                gap_nonempty = float(nonempty_rate(gap_hyps))

            _log(
                f"[eval step {step}] dev WER={dev_scores.wer:.4f} CER={dev_scores.cer:.4f} InsRate={dev_scores.ins_rate:.4f}"
                + (f" GapNonEmpty={gap_nonempty:.4f}" if gap_nonempty is not None else "")
            )

            improved = False
            if dev_scores.wer < best["wer"] - 1e-6:
                improved = True
            elif abs(dev_scores.wer - best["wer"]) <= 1e-6 and gap_nonempty is not None and gap_nonempty < best["nonempty"] - 1e-6:
                improved = True

            if improved:
                best = {
                    "wer": float(dev_scores.wer),
                    "nonempty": float(gap_nonempty) if gap_nonempty is not None else float("inf"),
                    "step": int(step),
                }
                bad_epochs = 0

                ckpt = cfg.out_dir / "best_lora.pt"
                torch.save(
                    {
                        "step": int(step),
                        "base_whisper_name": _normalize_openai_whisper_name(cfg.whisper_name),
                        "lora_cfg": _jsonify(asdict(cfg.lora)),
                        "lora_state": lora_state_dict(model),
                    },
                    ckpt,
                )

                (cfg.out_dir / "best_metrics.json").write_text(
                    json_dumps(
                        {
                            "step": int(step),
                            "dev": {
                                "wer": float(dev_scores.wer),
                                "cer": float(dev_scores.cer),
                                "ins_rate": float(dev_scores.ins_rate),
                                "gap_nonempty": float(gap_nonempty) if gap_nonempty is not None else None,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                _log(f"[eval] NEW BEST saved adapter to {ckpt}")
            else:
                bad_epochs += 1
                if bad_epochs >= int(cfg.patience):
                    _log(f"[early stop] no improvement for {bad_epochs} evals.")
                    break

            model.train()

    wall = time.time() - t0
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_dir": str(cfg.train_dir),
        "dev_dir": str(cfg.dev_dir),
        "dev_gap_dir": str(cfg.dev_gap_dir) if cfg.dev_gap_dir else None,
        "whisper_name": cfg.whisper_name,
        "device": cfg.device,
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "batch_size": int(cfg.batch_size),
        "max_steps": int(cfg.max_steps),
        "eval_every": int(cfg.eval_every),
        "patience": int(cfg.patience),
        "seed": int(cfg.seed),
        "fp16": bool(cfg.fp16),
        "lora": _jsonify(asdict(cfg.lora)),
        "replaced_modules": replaced,
        "best": best,
        "wall_sec": float(wall),
    }
    (cfg.out_dir / "manifest.json").write_text(json_dumps(manifest), encoding="utf-8")
    return cfg.out_dir



def _jsonify(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, Path):
        return str(x)
    if is_dataclass(x):
        return _jsonify(asdict(x))
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    return x





# def train_whisper_lora(cfg: WhisperLoRATrainConfig) -> Path:
#     _set_seed(cfg.seed)
#     cfg.out_dir.mkdir(parents=True, exist_ok=True)

#     log_path = cfg.out_dir / "train.log"

#     def _log(msg: str) -> None:
#         print(msg)
#         with log_path.open("a", encoding="utf-8") as f:
#             f.write(msg + "\n")

#     def _dump_longest_hyps(
#         tag: str,
#         refs: List[str],
#         hyps: List[str],
#         out_dir: Path,
#         topk: int = 25,
#     ) -> None:
#         """
#         Save the longest hypotheses (by hyp word count) to a txt file.
#         Useful for diagnosing insertion explosions / runaway decoding.
#         """
#         pairs = list(zip(refs, hyps))
#         pairs.sort(key=lambda x: len(x[1].split()), reverse=True)

#         ts = time.strftime("%Y%m%d_%H%M%S")
#         dbg_dir = out_dir / "debug"
#         dbg_dir.mkdir(parents=True, exist_ok=True)
#         path = dbg_dir / f"debug_longest_hyps_{tag}_{ts}.txt"

#         with path.open("w", encoding="utf-8") as f:
#             f.write(f"[debug] tag={tag}  n={len(pairs)}  topk={topk}\n")
#             f.write("Sorted by hyp word count (descending)\n\n")
#             for i, (r, h) in enumerate(pairs[:topk], start=1):
#                 r_words = len(r.split())
#                 h_words = len(h.split())
#                 f.write(f"#{i}\n")
#                 f.write(f"ref_words={r_words} hyp_words={h_words}  (diff={h_words - r_words})\n")
#                 f.write(f"REF: {r}\n")
#                 f.write(f"HYP: {h}\n")
#                 f.write("-" * 80 + "\n")

#         _log(f"[debug] wrote: {path}")

#     try:
#         import whisper
#         from whisper.tokenizer import get_tokenizer
#     except Exception as e:
#         raise RuntimeError(
#             "LoRA training requires the `openai-whisper` PyTorch package.\n"
#             "Install with: pip install -U openai-whisper\n"
#             f"Original error: {e}"
#         )

#     device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
#     use_amp = bool(cfg.fp16 and device.type == "cuda")

#     # Tokenizer (needed before dataset tokenization + decoding)
#     tokenizer = get_tokenizer(multilingual=False, language="en", task="transcribe")
#     pad_id = tokenizer.eot

#     # Load datasets
#     train_exs = load_dataset_dir(cfg.train_dir)
#     dev_exs = load_dataset_dir(cfg.dev_dir)
#     dev_gap_exs = load_dataset_dir(cfg.dev_gap_dir) if cfg.dev_gap_dir else []

#     # Precompute lists
#     train_items = [
#         (_resolve_audio_path(cfg.train_dir, e.audio_path), _tokenize_text(e.text, tokenizer))
#         for e in train_exs
#     ]
#     dev_items = [(_resolve_audio_path(cfg.dev_dir, e.audio_path), e.text) for e in dev_exs]
#     dev_gap_paths = (
#         [_resolve_audio_path(cfg.dev_gap_dir, e.audio_path) for e in dev_gap_exs]
#         if cfg.dev_gap_dir
#         else []
#     )

#     rng = torch.Generator().manual_seed(int(cfg.seed))

#     # Load base model
#     model = whisper.load_model(_normalize_openai_whisper_name(cfg.whisper_name), device=str(device))

#     # Freeze everything first
#     for p in model.parameters():
#         p.requires_grad = False

#     # ---- baseline sanity eval (step0_base) ----
#     model.eval()
#     dev_paths = [p for (p, _) in dev_items]
#     dev_refs = [t for (_, t) in dev_items]
#     dev_hyps = decode_openai_whisper(
#         model, tokenizer, dev_paths, device=device, temperature=0.0, beam_size=5
#     )
#     _dump_longest_hyps(tag="step0_base", refs=dev_refs, hyps=dev_hyps, out_dir=cfg.out_dir, topk=25)
#     dev_scores = aggregate_scores(dev_refs, dev_hyps)
#     _log(f"[eval step0_base] dev WER={dev_scores.wer:.4f} CER={dev_scores.cer:.4f} InsRate={dev_scores.ins_rate:.4f}")

#     # Apply LoRA once
#     replaced = apply_lora(model, cfg.lora)
#     params = lora_parameters(model)
#     if not params:
#         raise RuntimeError(
#             "No LoRA parameters were created.\n"
#             "Check LoRAConfig.scope_substrings/target_substrings and that they match Whisper module names."
#         )
#     for p in params:
#         p.requires_grad = True

#     # ---- sanity eval after LoRA injection but before training (step0_lora_init) ----
#     model.eval()
#     dev_hyps = decode_openai_whisper(
#         model, tokenizer, dev_paths, device=device, temperature=0.0, beam_size=5
#     )
#     _dump_longest_hyps(tag="step0_lora_init", refs=dev_refs, hyps=dev_hyps, out_dir=cfg.out_dir, topk=25)
#     dev_scores = aggregate_scores(dev_refs, dev_hyps)
#     _log(f"[eval step0_lora_init] dev WER={dev_scores.wer:.4f} CER={dev_scores.cer:.4f} InsRate={dev_scores.ins_rate:.4f}")

#     model.train()

#     # Optimizer + scaler
#     opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
#     scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

#     best = {"wer": float("inf"), "nonempty": float("inf"), "step": -1}
#     bad_epochs = 0

#     # debug: check LoRA actually updates
#     with torch.no_grad():
#         a0 = params[0].detach().float().clone()

#     def _sample_batch() -> Batch:
#         idx = torch.randint(low=0, high=len(train_items), size=(cfg.batch_size,), generator=rng).tolist()
#         batch = []
#         for i in idx:
#             p, ids = train_items[i]
#             mel = _load_audio_mel(p, device)
#             batch.append((mel, ids))
#         return _collate(batch, pad_id=pad_id, device=device)

#     t0 = time.time()
#     _log(f"[lora] replaced Linear modules: {len(replaced)}")
#     if len(replaced) <= 20:
#         for n in replaced:
#             _log(f"  - {n}")
#     else:
#         _log(f"  (first 10) {replaced[:10]}")

#     for step in range(1, int(cfg.max_steps) + 1):
#         b = _sample_batch()
#         opt.zero_grad(set_to_none=True)

#         with torch.amp.autocast("cuda", enabled=use_amp):
#             logits = model(b.mel, b.tokens_in)  # (B, L, vocab)
#             loss = F.cross_entropy(
#                 logits.view(-1, logits.size(-1)),
#                 b.targets.view(-1),
#                 ignore_index=-100,
#             )

#         if use_amp:
#             scaler.scale(loss).backward()
#             scaler.unscale_(opt)
#             torch.nn.utils.clip_grad_norm_(params, 1.0)
#             scaler.step(opt)
#             scaler.update()
#         else:
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(params, 1.0)
#             opt.step()

#         if step % 50 == 0:
#             with torch.no_grad():
#                 dA = (params[0].detach().float() - a0).abs().mean().item()
#             g = params[0].grad
#             gmean = float(g.detach().abs().mean().item()) if g is not None else 0.0
#             _log(f"[step {step}] loss={loss.detach().float().item():.4f}  mean|ΔA|={dA:.3e}  mean|grad(A)|={gmean:.3e}")

#         if step % int(cfg.eval_every) == 0 or step == int(cfg.max_steps):
#             model.eval()

#             dev_hyps = decode_openai_whisper(
#                 model, tokenizer, dev_paths, device=device, temperature=0.0, beam_size=5
#             )
#             _dump_longest_hyps(tag=f"step{step}", refs=dev_refs, hyps=dev_hyps, out_dir=cfg.out_dir, topk=25)
#             dev_scores = aggregate_scores(dev_refs, dev_hyps)

#             gap_nonempty = None
#             if dev_gap_paths:
#                 gap_hyps = decode_openai_whisper(
#                     model, tokenizer, dev_gap_paths, device=device, temperature=0.0, beam_size=5
#                 )
#                 gap_nonempty = float(nonempty_rate(gap_hyps))

#             _log(
#                 f"[eval step {step}] dev WER={dev_scores.wer:.4f} CER={dev_scores.cer:.4f} InsRate={dev_scores.ins_rate:.4f}"
#                 + (f" GapNonEmpty={gap_nonempty:.4f}" if gap_nonempty is not None else "")
#             )

#             improved = False
#             if dev_scores.wer < best["wer"] - 1e-6:
#                 improved = True
#             elif abs(dev_scores.wer - best["wer"]) <= 1e-6 and gap_nonempty is not None and gap_nonempty < best["nonempty"] - 1e-6:
#                 improved = True

#             if improved:
#                 best = {
#                     "wer": float(dev_scores.wer),
#                     "nonempty": float(gap_nonempty) if gap_nonempty is not None else float("inf"),
#                     "step": int(step),
#                 }
#                 bad_epochs = 0

#                 ckpt = cfg.out_dir / "best_lora.pt"
#                 torch.save(
#                     {
#                         "step": int(step),
#                         "base_whisper_name": _normalize_openai_whisper_name(cfg.whisper_name),
#                         "lora_cfg": _jsonify(asdict(cfg.lora)),
#                         "lora_state": lora_state_dict(model),
#                     },
#                     ckpt,
#                 )

#                 (cfg.out_dir / "best_metrics.json").write_text(
#                     json_dumps(
#                         {
#                             "step": int(step),
#                             "dev": {
#                                 "wer": float(dev_scores.wer),
#                                 "cer": float(dev_scores.cer),
#                                 "ins_rate": float(dev_scores.ins_rate),
#                                 "gap_nonempty": float(gap_nonempty) if gap_nonempty is not None else None,
#                             },
#                         }
#                     ),
#                     encoding="utf-8",
#                 )
#                 _log(f"[eval] NEW BEST saved adapter to {ckpt}")
#             else:
#                 bad_epochs += 1
#                 if bad_epochs >= int(cfg.patience):
#                     _log(f"[early stop] no improvement for {bad_epochs} evals.")
#                     break

#             model.train()

#     wall = time.time() - t0
#     manifest = {
#         "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#         "train_dir": str(cfg.train_dir),
#         "dev_dir": str(cfg.dev_dir),
#         "dev_gap_dir": str(cfg.dev_gap_dir) if cfg.dev_gap_dir else None,
#         "whisper_name": cfg.whisper_name,
#         "device": cfg.device,
#         "lr": float(cfg.lr),
#         "weight_decay": float(cfg.weight_decay),
#         "batch_size": int(cfg.batch_size),
#         "max_steps": int(cfg.max_steps),
#         "eval_every": int(cfg.eval_every),
#         "patience": int(cfg.patience),
#         "seed": int(cfg.seed),
#         "fp16": bool(cfg.fp16),
#         "lora": _jsonify(asdict(cfg.lora)),
#         "replaced_modules": replaced,
#         "best": best,
#         "wall_sec": float(wall),
#     }
#     (cfg.out_dir / "manifest.json").write_text(json_dumps(manifest), encoding="utf-8")
#     return cfg.out_dir




# def _jsonify(x: Any) -> Any:
#     if x is None:
#         return None
#     if isinstance(x, Path):
#         return str(x)
#     if is_dataclass(x):
#         return _jsonify(asdict(x))
#     if isinstance(x, dict):
#         return {str(k): _jsonify(v) for k, v in x.items()}
#     if isinstance(x, (list, tuple)):
#         return [_jsonify(v) for v in x]
#     return x



# def train_whisper_lora(cfg: WhisperLoRATrainConfig) -> Path:
#     _set_seed(cfg.seed)
#     cfg.out_dir.mkdir(parents=True, exist_ok=True)

#     def _dump_longest_hyps(
#         tag: str,
#         refs: list[str],
#         hyps: list[str],
#         out_dir: Path,
#         topk: int = 25,
#     ) -> Path:
#         """
#         Save the longest hypotheses (by hyp word count) to a txt file.
#         Useful for diagnosing insertion explosions / runaway decoding.
#         """
#         out_dir.mkdir(parents=True, exist_ok=True)
#         pairs = list(zip(refs, hyps))
#         pairs.sort(key=lambda x: len(x[1].split()), reverse=True)

#         ts = time.strftime("%Y%m%d_%H%M%S")
#         dbg_dir = out_dir / "debug"
#         dbg_dir.mkdir(parents=True, exist_ok=True)
#         path = dbg_dir / f"debug_longest_hyps_{tag}_{ts}.txt"

#         with path.open("w", encoding="utf-8") as f:
#             f.write(f"[debug] tag={tag}  n={len(pairs)}  topk={topk}\n")
#             f.write("Sorted by hyp word count (descending)\n\n")
#             for i, (r, h) in enumerate(pairs[:topk], start=1):
#                 r_words = len(r.split())
#                 h_words = len(h.split())
#                 f.write(f"#{i}\n")
#                 f.write(f"ref_words={r_words} hyp_words={h_words}  (diff={h_words - r_words})\n")
#                 f.write(f"REF: {r}\n")
#                 f.write(f"HYP: {h}\n")
#                 f.write("-" * 80 + "\n")

#         print(f"[debug] wrote: {path}")
#         return path

#     def _append_eval_jsonl(payload: dict) -> None:
#         """
#         Append evaluation metrics to a JSONL log for easy later plotting/inspection.
#         """
#         path = cfg.out_dir / "eval_log.jsonl"
#         with path.open("a", encoding="utf-8") as f:
#             f.write(json_dumps(payload) + "\n")

#     try:
#         import whisper
#         from whisper.tokenizer import get_tokenizer
#     except Exception as e:
#         raise RuntimeError(
#             "LoRA training requires the `openai-whisper` PyTorch package.\n"
#             "Install with: pip install -U openai-whisper\n"
#             f"Original error: {e}"
#         )

#     device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
#     model = whisper.load_model(_normalize_openai_whisper_name(cfg.whisper_name), device=str(device))

#     # Freeze everything first (we will unfreeze LoRA params after insertion)
#     for p in model.parameters():
#         p.requires_grad = False

#     # Tokenizer is needed for dataset tokenization + decoding
#     tokenizer = get_tokenizer(multilingual=False, language="en", task="transcribe")
#     pad_id = tokenizer.eot

#     # Load datasets
#     train_exs = load_dataset_dir(cfg.train_dir)
#     dev_exs = load_dataset_dir(cfg.dev_dir)
#     dev_gap_exs = load_dataset_dir(cfg.dev_gap_dir) if cfg.dev_gap_dir else []

#     # Precompute (abs_path, tokens/text) lists (mels computed on the fly)
#     train_items = [
#         (_resolve_audio_path(cfg.train_dir, e.audio_path), _tokenize_text(e.text, tokenizer))
#         for e in train_exs
#     ]
#     dev_items = [(_resolve_audio_path(cfg.dev_dir, e.audio_path), e.text) for e in dev_exs]
#     dev_gap_paths = [_resolve_audio_path(cfg.dev_gap_dir, e.audio_path) for e in dev_gap_exs] if cfg.dev_gap_dir else []

#     rng = torch.Generator().manual_seed(int(cfg.seed))

#     best = {"wer": float("inf"), "nonempty": float("inf"), "step": -1}
#     bad_epochs = 0

#     def _eval_dev(tag: str, dump_longest: bool = True) -> tuple[Any, list[str], list[str]]:
#         """
#         Run decoding on dev and return (scores, refs, hyps).
#         Also optionally dumps the longest hypotheses to file for debugging.
#         """
#         model.eval()
#         dev_paths = [p for (p, _) in dev_items]
#         dev_refs = [t for (_, t) in dev_items]
#         dev_hyps = decode_openai_whisper(
#             model,
#             tokenizer,
#             dev_paths,
#             device=device,
#             temperature=0.0,
#             beam_size=5,
#         )

#         dbg_path = None
#         if dump_longest:
#             dbg_path = str(_dump_longest_hyps(tag=tag, refs=dev_refs, hyps=dev_hyps, out_dir=cfg.out_dir, topk=25))

#         dev_scores = aggregate_scores(dev_refs, dev_hyps)

#         gap_nonempty = None
#         if dev_gap_paths:
#             gap_hyps = decode_openai_whisper(
#                 model,
#                 tokenizer,
#                 dev_gap_paths,
#                 device=device,
#                 temperature=0.0,
#                 beam_size=5,
#             )
#             gap_nonempty = float(nonempty_rate(gap_hyps))

#         msg = (
#             f"[eval {tag}] dev WER={dev_scores.wer:.4f} CER={dev_scores.cer:.4f} InsRate={dev_scores.ins_rate:.4f}"
#             + (f" GapNonEmpty={gap_nonempty:.4f}" if gap_nonempty is not None else "")
#         )
#         print(msg)

#         _append_eval_jsonl(
#             {
#                 "tag": tag,
#                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#                 "dev": {
#                     "wer": float(dev_scores.wer),
#                     "cer": float(dev_scores.cer),
#                     "ins_rate": float(dev_scores.ins_rate),
#                 },
#                 "gap_nonempty": float(gap_nonempty) if gap_nonempty is not None else None,
#                 "debug_longest_path": dbg_path,
#             }
#         )

#         model.train()
#         # return scores plus gap_nonempty (as attribute-like access convenience isn't guaranteed)
#         return (dev_scores, gap_nonempty), dev_refs, dev_hyps

#     # ---- baseline sanity eval (true baseline: BEFORE LoRA is applied) ----
#     _eval_dev(tag="step0_base", dump_longest=True)

#     # ---- Apply LoRA once ----
#     replaced = apply_lora(model, cfg.lora)
#     params = lora_parameters(model)
#     if not params:
#         raise RuntimeError(
#             "No LoRA parameters were created.\n"
#             "Check LoRAConfig.scope_substrings/target_substrings and that they match Whisper module names."
#         )
#     for p in params:
#         p.requires_grad = True

#     print(f"[lora] replaced Linear modules: {len(replaced)}")
#     if len(replaced) <= 20:
#         for n in replaced:
#             print(f"  - {n}")
#     else:
#         print(f"  (first 10) {replaced[:10]}")

#     # Optimizer / scaler
#     opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
#     scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.fp16 and device.type == "cuda"))

#     # Optional sanity: eval right after LoRA insertion (should match step0_base closely)
#     _eval_dev(tag="step0_lora_init", dump_longest=False)

#     # debug: check LoRA actually updates
#     with torch.no_grad():
#         a0 = params[0].detach().float().clone()

#     def _sample_batch() -> Batch:
#         idx = torch.randint(low=0, high=len(train_items), size=(cfg.batch_size,), generator=rng).tolist()
#         batch = []
#         for i in idx:
#             p, ids = train_items[i]
#             mel = _load_audio_mel(p, device)
#             batch.append((mel, ids))
#         return _collate(batch, pad_id=pad_id, device=device)

#     t0 = time.time()

#     for step in range(1, int(cfg.max_steps) + 1):
#         b = _sample_batch()
#         opt.zero_grad(set_to_none=True)

#         with torch.amp.autocast("cuda", enabled=bool(cfg.fp16 and device.type == "cuda")):
#             logits = model(b.mel, b.tokens_in)  # (B, L, vocab)
#             loss = F.cross_entropy(
#                 logits.view(-1, logits.size(-1)),
#                 b.targets.view(-1),
#                 ignore_index=-100,
#             )

#         scaler.scale(loss).backward()
#         scaler.unscale_(opt)
#         torch.nn.utils.clip_grad_norm_(params, 1.0)
#         scaler.step(opt)
#         scaler.update()

#         if step % 50 == 0:
#             with torch.no_grad():
#                 dA = (params[0].detach().float() - a0).abs().mean().item()
#             g = params[0].grad
#             gmean = float(g.detach().abs().mean().item()) if g is not None else 0.0
#             print(
#                 f"[step {step}] loss={loss.detach().float().item():.4f}  "
#                 f"mean|ΔA|={dA:.3e}  mean|grad(A)|={gmean:.3e}"
#             )

#         if step % int(cfg.eval_every) == 0 or step == int(cfg.max_steps):
#             # Dump longest hyps occasionally (first 2 evals + final), always log metrics.
#             dump_longest = step in (int(cfg.eval_every), 2 * int(cfg.eval_every), int(cfg.max_steps))
#             (dev_scores, gap_nonempty), _, _ = _eval_dev(tag=f"step{step}", dump_longest=dump_longest)

#             improved = False
#             if float(dev_scores.wer) < float(best["wer"]) - 1e-6:
#                 improved = True
#             elif (
#                 abs(float(dev_scores.wer) - float(best["wer"])) <= 1e-6
#                 and gap_nonempty is not None
#                 and float(gap_nonempty) < float(best["nonempty"]) - 1e-6
#             ):
#                 improved = True

#             if improved:
#                 best = {
#                     "wer": float(dev_scores.wer),
#                     "nonempty": float(gap_nonempty) if gap_nonempty is not None else float("inf"),
#                     "step": int(step),
#                 }
#                 bad_epochs = 0

#                 # Save a small adapter checkpoint (ONLY LoRA weights)
#                 ckpt = cfg.out_dir / "best_lora.pt"
#                 torch.save(
#                     {
#                         "step": int(step),
#                         "base_whisper_name": _normalize_openai_whisper_name(cfg.whisper_name),
#                         "lora_cfg": _jsonify(asdict(cfg.lora)),
#                         "lora_state": lora_state_dict(model),
#                     },
#                     ckpt,
#                 )

#                 (cfg.out_dir / "best_metrics.json").write_text(
#                     json_dumps(
#                         {
#                             "step": int(step),
#                             "dev": {
#                                 "wer": float(dev_scores.wer),
#                                 "cer": float(dev_scores.cer),
#                                 "ins_rate": float(dev_scores.ins_rate),
#                                 "gap_nonempty": float(gap_nonempty) if gap_nonempty is not None else None,
#                             },
#                         }
#                     ),
#                     encoding="utf-8",
#                 )
#                 print(f"[eval] NEW BEST saved adapter to {ckpt}")
#             else:
#                 bad_epochs += 1
#                 if bad_epochs >= int(cfg.patience):
#                     print(f"[early stop] no improvement for {bad_epochs} evals.")
#                     break

#     wall = time.time() - t0
#     manifest = {
#         "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#         "train_dir": str(cfg.train_dir),
#         "dev_dir": str(cfg.dev_dir),
#         "dev_gap_dir": str(cfg.dev_gap_dir) if cfg.dev_gap_dir else None,
#         "whisper_name": cfg.whisper_name,
#         "device": cfg.device,
#         "lr": float(cfg.lr),
#         "weight_decay": float(cfg.weight_decay),
#         "batch_size": int(cfg.batch_size),
#         "max_steps": int(cfg.max_steps),
#         "eval_every": int(cfg.eval_every),
#         "patience": int(cfg.patience),
#         "seed": int(cfg.seed),
#         "fp16": bool(cfg.fp16),
#         "lora": _jsonify(asdict(cfg.lora)),
#         "replaced_modules": replaced,
#         "best": best,
#         "wall_sec": float(wall),
#     }
#     (cfg.out_dir / "manifest.json").write_text(json_dumps(manifest), encoding="utf-8")
#     return cfg.out_dir




# def _jsonify(x: Any) -> Any:
#     if x is None:
#         return None
#     if isinstance(x, Path):
#         return str(x)
#     if is_dataclass(x):
#         return _jsonify(asdict(x))
#     if isinstance(x, dict):
#         return {str(k): _jsonify(v) for k, v in x.items()}
#     if isinstance(x, (list, tuple)):
#         return [_jsonify(v) for v in x]
#     return x


# def train_whisper_lora(cfg: WhisperLoRATrainConfig) -> Path:
#     _set_seed(cfg.seed)
#     cfg.out_dir.mkdir(parents=True, exist_ok=True)


#     def _dump_longest_hyps(tag: str, refs: list[str], hyps: list[str], out_dir: Path, topk: int = 25) -> None:
#         """
#         Save the longest hypotheses (by hyp word count) to a txt file.
#         Useful for diagnosing insertion explosions / runaway decoding.
#         """
#         out_dir.mkdir(parents=True, exist_ok=True)
#         pairs = list(zip(refs, hyps))
#         pairs.sort(key=lambda x: len(x[1].split()), reverse=True)

#         ts = time.strftime("%Y%m%d_%H%M%S")
#         dbg_dir = out_dir / "debug"
#         dbg_dir.mkdir(parents=True, exist_ok=True)
#         path = dbg_dir / f"debug_longest_hyps_{tag}_{ts}.txt"

#         with path.open("w", encoding="utf-8") as f:
#             f.write(f"[debug] tag={tag}  n={len(pairs)}  topk={topk}\n")
#             f.write("Sorted by hyp word count (descending)\n\n")
#             for i, (r, h) in enumerate(pairs[:topk], start=1):
#                 r_words = len(r.split())
#                 h_words = len(h.split())
#                 f.write(f"#{i}\n")
#                 f.write(f"ref_words={r_words} hyp_words={h_words}  (diff={h_words - r_words})\n")
#                 f.write(f"REF: {r}\n")
#                 f.write(f"HYP: {h}\n")
#                 f.write("-" * 80 + "\n")

#         print(f"[debug] wrote: {path}")


#     try:
#         import whisper
#         from whisper.tokenizer import get_tokenizer
#     except Exception as e:
#         raise RuntimeError(
#             "LoRA training requires the `openai-whisper` PyTorch package.\n"
#             "Install with: pip install -U openai-whisper\n"
#             f"Original error: {e}"
#         )

#     device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
#     model = whisper.load_model(_normalize_openai_whisper_name(cfg.whisper_name), device=str(device))

#     # Freeze everything first
#     for p in model.parameters():
#         p.requires_grad = False

#     # ---- baseline sanity eval (step 0) ----
#     model.eval()
#     dev_paths = [p for (p, _) in dev_items]
#     dev_refs  = [t for (_, t) in dev_items]
#     dev_hyps  = decode_openai_whisper(model, tokenizer, dev_paths, device=device, temperature=0.0, beam_size=5)

#     _dump_longest_hyps(tag="step0", refs=dev_refs, hyps=dev_hyps, out_dir=cfg.out_dir, topk=25)

#     dev_scores = aggregate_scores(dev_refs, dev_hyps)
#     print(f"[eval step 0] dev WER={dev_scores.wer:.4f} CER={dev_scores.cer:.4f} InsRate={dev_scores.ins_rate:.4f}")
#     model.train()



#     # Apply LoRA once
#     replaced = apply_lora(model, cfg.lora)
#     params = lora_parameters(model)
#     if not params:
#         raise RuntimeError(
#             "No LoRA parameters were created.\n"
#             "Check LoRAConfig.scope_substrings/target_substrings and that they match Whisper module names."
#         )
#     for p in params:
#         p.requires_grad = True

#     model.train()

#     tokenizer = get_tokenizer(multilingual=False, language="en", task="transcribe")
#     pad_id = tokenizer.eot

#     # Optimizer
#     opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

#     # Load datasets
#     train_exs = load_dataset_dir(cfg.train_dir)
#     dev_exs = load_dataset_dir(cfg.dev_dir)
#     dev_gap_exs = load_dataset_dir(cfg.dev_gap_dir) if cfg.dev_gap_dir else []

#     # Precompute (abs_path, tokens) lists (mels computed on the fly to reduce RAM)
#     train_items = [(_resolve_audio_path(cfg.train_dir, e.audio_path), _tokenize_text(e.text, tokenizer)) for e in train_exs]
#     dev_items = [(_resolve_audio_path(cfg.dev_dir, e.audio_path), e.text) for e in dev_exs]
#     dev_gap_paths = [_resolve_audio_path(cfg.dev_gap_dir, e.audio_path) for e in dev_gap_exs] if cfg.dev_gap_dir else []

#     rng = torch.Generator().manual_seed(int(cfg.seed))

#     best = {"wer": float("inf"), "nonempty": float("inf"), "step": -1}
#     bad_epochs = 0


#     # scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.fp16 and device.type == "cuda"))
#     scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.fp16 and device.type == "cuda"))


#     # ---- baseline sanity eval (step 0) ----
#     model.eval()
#     dev_paths = [p for (p, _) in dev_items]
#     dev_refs  = [t for (_, t) in dev_items]
#     dev_hyps  = decode_openai_whisper(model, tokenizer, dev_paths, device=device, temperature=0.0, beam_size=5)

#     _dump_longest_hyps(tag="step0", refs=dev_refs, hyps=dev_hyps, out_dir=cfg.out_dir, topk=25)

#     dev_scores = aggregate_scores(dev_refs, dev_hyps)
#     print(f"[eval step 0] dev WER={dev_scores.wer:.4f} CER={dev_scores.cer:.4f} InsRate={dev_scores.ins_rate:.4f}")
#     model.train()


#     # debug: check LoRA actually updates
#     with torch.no_grad():
#         a0 = params[0].detach().float().clone()

#     def _sample_batch() -> Batch:
#         idx = torch.randint(low=0, high=len(train_items), size=(cfg.batch_size,), generator=rng).tolist()
#         batch = []
#         for i in idx:
#             p, ids = train_items[i]
#             mel = _load_audio_mel(p, device)
#             batch.append((mel, ids))
#         return _collate(batch, pad_id=pad_id, device=device)

#     t0 = time.time()
#     print(f"[lora] replaced Linear modules: {len(replaced)}")
#     if len(replaced) <= 20:
#         for n in replaced:
#             print(f"  - {n}")
#     else:
#         print(f"  (first 10) {replaced[:10]}")

#     for step in range(1, int(cfg.max_steps) + 1):
#         b = _sample_batch()
#         opt.zero_grad(set_to_none=True)

#         # with torch.cuda.amp.autocast(enabled=bool(cfg.fp16 and device.type == "cuda")):
#         with torch.amp.autocast("cuda", enabled=bool(cfg.fp16 and device.type == "cuda")):
#             logits = model(b.mel, b.tokens_in)  # (B, L, vocab)
#             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), b.targets.view(-1), ignore_index=-100)

#         scaler.scale(loss).backward()
#         scaler.unscale_(opt)
#         torch.nn.utils.clip_grad_norm_(params, 1.0)
#         scaler.step(opt)
#         scaler.update()

#         if step % 50 == 0:
#             with torch.no_grad():
#                 dA = (params[0].detach().float() - a0).abs().mean().item()
#             g = params[0].grad
#             gmean = float(g.detach().abs().mean().item()) if g is not None else 0.0
#             print(f"[step {step}] loss={loss.detach().float().item():.4f}  mean|ΔA|={dA:.3e}  mean|grad(A)|={gmean:.3e}")

#         if step % int(cfg.eval_every) == 0 or step == int(cfg.max_steps):
#             model.eval()

#             dev_paths = [p for (p, _) in dev_items]
#             dev_refs = [t for (_, t) in dev_items]
#             dev_hyps = decode_openai_whisper(model, tokenizer, dev_paths, device=device, temperature=0.0, beam_size=5)
#             if step in (cfg.eval_every, 2*cfg.eval_every, cfg.max_steps):
#                 _dump_longest_hyps(tag=f"step{step}", refs=dev_refs, hyps=dev_hyps, out_dir=cfg.out_dir, topk=25)
#             dev_scores = aggregate_scores(dev_refs, dev_hyps)

#             gap_nonempty = None
#             if dev_gap_paths:
#                 gap_hyps = decode_openai_whisper(model, tokenizer, dev_gap_paths, device=device, temperature=0.0, beam_size=5)
#                 gap_nonempty = float(nonempty_rate(gap_hyps))

#             print(
#                 f"[eval step {step}] dev WER={dev_scores.wer:.4f} CER={dev_scores.cer:.4f} InsRate={dev_scores.ins_rate:.4f}"
#                 + (f" GapNonEmpty={gap_nonempty:.4f}" if gap_nonempty is not None else "")
#             )

#             improved = False
#             if dev_scores.wer < best["wer"] - 1e-6:
#                 improved = True
#             elif abs(dev_scores.wer - best["wer"]) <= 1e-6 and gap_nonempty is not None and gap_nonempty < best["nonempty"] - 1e-6:
#                 improved = True

#             if improved:
#                 best = {
#                     "wer": float(dev_scores.wer),
#                     "nonempty": float(gap_nonempty) if gap_nonempty is not None else float("inf"),
#                     "step": int(step),
#                 }
#                 bad_epochs = 0

#                 # Save a small adapter checkpoint (ONLY LoRA weights)
#                 ckpt = cfg.out_dir / "best_lora.pt"
#                 torch.save(
#                     {
#                         "step": int(step),
#                         "base_whisper_name": _normalize_openai_whisper_name(cfg.whisper_name),
#                         "lora_cfg": _jsonify(asdict(cfg.lora)),
#                         "lora_state": lora_state_dict(model),
#                     },
#                     ckpt,
#                 )

#                 (cfg.out_dir / "best_metrics.json").write_text(
#                     json_dumps(
#                         {
#                             "step": int(step),
#                             "dev": {
#                                 "wer": float(dev_scores.wer),
#                                 "cer": float(dev_scores.cer),
#                                 "ins_rate": float(dev_scores.ins_rate),
#                                 "gap_nonempty": float(gap_nonempty) if gap_nonempty is not None else None,
#                             },
#                         }
#                     ),
#                     encoding="utf-8",
#                 )
#                 print(f"[eval] NEW BEST saved adapter to {ckpt}")
#             else:
#                 bad_epochs += 1
#                 if bad_epochs >= int(cfg.patience):
#                     print(f"[early stop] no improvement for {bad_epochs} evals.")
#                     break

#             model.train()

#     wall = time.time() - t0
#     manifest = {
#         "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#         "train_dir": str(cfg.train_dir),
#         "dev_dir": str(cfg.dev_dir),
#         "dev_gap_dir": str(cfg.dev_gap_dir) if cfg.dev_gap_dir else None,
#         "whisper_name": cfg.whisper_name,
#         "device": cfg.device,
#         "lr": float(cfg.lr),
#         "weight_decay": float(cfg.weight_decay),
#         "batch_size": int(cfg.batch_size),
#         "max_steps": int(cfg.max_steps),
#         "eval_every": int(cfg.eval_every),
#         "patience": int(cfg.patience),
#         "seed": int(cfg.seed),
#         "fp16": bool(cfg.fp16),
#         "lora": _jsonify(asdict(cfg.lora)),
#         "replaced_modules": replaced,
#         "best": best,
#         "wall_sec": float(wall),
#     }
#     (cfg.out_dir / "manifest.json").write_text(json_dumps(manifest), encoding="utf-8")
#     return cfg.out_dir


# def _jsonify(x: Any) -> Any:
#     if x is None:
#         return None
#     if isinstance(x, Path):
#         return str(x)
#     if is_dataclass(x):
#         return _jsonify(asdict(x))
#     if isinstance(x, dict):
#         return {str(k): _jsonify(v) for k, v in x.items()}
#     if isinstance(x, (list, tuple)):
#         return [_jsonify(v) for v in x]
#     return x



# from __future__ import annotations

# """
# Parameter-efficient fine-tuning (LoRA) for OpenAI Whisper (PyTorch).

# Used by adapt_whisper_lora.py to implement the adaptation protocol:
# LoRA adaptation on speaker-change proxy concatenations, validated on a held-out
# split with early stopping on WER (and optionally a non-speech non-empty probe).

# Notes:
# - Targets the `openai-whisper` PyTorch package (`pip install -U openai-whisper`).
# - It does NOT fine-tune the CTranslate2/faster-whisper runtime directly.
# - Written for readability + debuggability.

# Outputs:
# - best_lora.pt : a small "adapter" checkpoint containing ONLY LoRA weights + metadata
# - manifest.json / best_metrics.json : run bookkeeping
# """

# from dataclasses import dataclass, asdict, is_dataclass
# from pathlib import Path
# from typing import Any, Dict, Iterable, List, Optional, Tuple

# import math
# import time

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .jsonl_dataset import load_dataset_dir
# from .scoring import aggregate_scores, nonempty_rate


# # -----------------------------
# # Helpers
# # -----------------------------

# def _normalize_openai_whisper_name(name: str) -> str:
#     """
#     openai-whisper expects names like: 'small.en', 'large-v3', etc.
#     But we sometimes pass HF-style names like: 'openai/whisper-small.en'.
#     This maps HF-style -> openai-whisper style.
#     """
#     n = (name or "").strip()
#     if n.startswith("openai/whisper-"):
#         n = n[len("openai/whisper-"):]
#     if n.startswith("whisper-"):
#         n = n[len("whisper-"):]
#     return n


# def _resolve_audio_path(ds_dir: Path, p: Path) -> Path:
#     """Resolve relative audio paths against a dataset directory."""
#     if p.is_absolute():
#         return p
#     cand = (ds_dir / p).resolve()
#     return cand


# def json_dumps(obj) -> str:
#     import json
#     return json.dumps(obj, indent=2, ensure_ascii=False)


# # -----------------------------
# # LoRA building blocks
# # -----------------------------

# class LoRALinear(nn.Module):
#     """
#     Wrap an existing nn.Linear with a trainable low-rank update:

#         y = base(x) + scale * (drop(x) @ A^T @ B^T)

#     where:
#       A: (r, in_features)
#       B: (out_features, r)

#     Base weights are frozen; only A,B are trainable.
#     """
#     def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
#         super().__init__()
#         if r <= 0:
#             raise ValueError("LoRA rank r must be > 0")
#         self.base = base
#         self.r = int(r)
#         self.alpha = float(alpha)
#         self.scale = self.alpha / float(self.r)
#         self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

#         # Freeze base parameters
#         for p in self.base.parameters():
#             p.requires_grad = False

#         in_f = int(base.in_features)
#         out_f = int(base.out_features)

#         # Force fp32 LoRA weights (avoids AMP dtype mismatches).
#         dev = base.weight.device
#         dt = torch.float32
#         self.A = nn.Parameter(torch.zeros((self.r, in_f), device=dev, dtype=dt))
#         self.B = nn.Parameter(torch.zeros((out_f, self.r), device=dev, dtype=dt))

#         # Init (common LoRA practice: A random, B zeros -> starts as no-op)
#         nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
#         nn.init.zeros_(self.B)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = self.base(x)

#         # LoRA branch: compute in A's dtype (fp32), then cast to y dtype (fp16 under autocast)
#         x2 = self.dropout(x).to(dtype=self.A.dtype)
#         z = torch.matmul(x2, self.A.t())              # (..., r)
#         dz = torch.matmul(z, self.B.t()) * self.scale # (..., out)
#         return y + dz.to(dtype=y.dtype)


# @dataclass(frozen=True)
# class LoRAConfig:
#     # low-rank update size
#     r: int = 8
#     alpha: float = 16.0
#     dropout: float = 0.0

#     # If scope_substrings is non-empty, we ONLY apply LoRA to Linear layers whose module
#     # name contains one of these substrings (e.g. "encoder.blocks", "decoder.blocks").
#     scope_substrings: Tuple[str, ...] = ()

#     # Additionally, if target_substrings is non-empty, the Linear module name must also
#     # contain one of these substrings. If empty, we match ALL Linear layers within scope.
#     target_substrings: Tuple[str, ...] = ("query", "key", "value", "out", "fc", "proj", "mlp")


# def apply_lora(model: nn.Module, cfg: LoRAConfig) -> List[str]:
#     """
#     Replace selected nn.Linear modules with LoRALinear wrappers.
#     Returns a list of replaced module names.
#     """
#     replaced: List[str] = []

#     def _iter_named_parents(m: nn.Module, prefix: str = ""):
#         for name, child in m.named_children():
#             full = f"{prefix}.{name}" if prefix else name
#             yield m, name, child, full
#             yield from _iter_named_parents(child, full)

#     for parent, attr, child, full_name in _iter_named_parents(model):
#         if not isinstance(child, nn.Linear):
#             continue

#         n = full_name.lower()
#         scope_ok = True
#         if cfg.scope_substrings:
#             scope_ok = any(s.lower() in n for s in cfg.scope_substrings)

#         target_ok = True
#         if cfg.target_substrings:
#             target_ok = any(s.lower() in n for s in cfg.target_substrings)

#         if scope_ok and target_ok:
#             setattr(parent, attr, LoRALinear(child, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout))
#             replaced.append(full_name)

#     return replaced


# def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
#     params: List[nn.Parameter] = []
#     for m in model.modules():
#         if isinstance(m, LoRALinear):
#             params.extend([m.A, m.B])
#     return params


# def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
#     """
#     Return ONLY LoRA weights from the model state dict.
#     (Keeps checkpoints small and backend-agnostic.)
#     """
#     sd = model.state_dict()
#     return {k: v.detach().cpu() for k, v in sd.items() if k.endswith(".A") or k.endswith(".B")}


# def load_lora_adapter(model: nn.Module, adapter_path: Path, device: torch.device) -> Dict[str, Any]:
#     """
#     Load a LoRA adapter checkpoint saved by this trainer (best_lora.pt).
#     Returns metadata dict and prints helpful debug info.
#     """
#     obj = torch.load(str(adapter_path), map_location="cpu")
#     if isinstance(obj, dict) and "lora_state" in obj:
#         state = obj["lora_state"]
#         meta = {k: v for k, v in obj.items() if k != "lora_state"}
#     else:
#         # allow passing a raw state-dict file for convenience
#         state = obj
#         meta = {}

#     missing, unexpected = model.load_state_dict(state, strict=False)

#     # filter out base weights missing (expected) vs lora missing (not expected)
#     lora_missing = [k for k in missing if k.endswith(".A") or k.endswith(".B")]

#     print(f"[lora] loaded adapter: {adapter_path}")
#     print(f"[lora] missing_keys={len(missing)} (lora_missing={len(lora_missing)}), unexpected_keys={len(unexpected)}")
#     if lora_missing:
#         print(f"[lora] WARNING: some LoRA keys were missing (first 10): {lora_missing[:10]}")
#     if unexpected:
#         print(f"[lora] WARNING: unexpected keys (first 10): {unexpected[:10]}")

#     return meta


# # -----------------------------
# # Whisper dataset utilities
# # -----------------------------

# def _load_audio_mel(path: Path, device: torch.device) -> torch.Tensor:
#     import whisper  # openai-whisper

#     # load and resample to 16k
#     audio = whisper.load_audio(str(path))

#     # IMPORTANT: pad/trim the waveform to the 30s window first
#     audio = whisper.pad_or_trim(audio)

#     # then compute log-mel (80 x 3000 frames for 30s audio)
#     mel = whisper.log_mel_spectrogram(audio)

#     return mel.to(device)


# def _tokenize_text(text: str, tokenizer) -> List[int]:
#     ids = tokenizer.encode(text)
#     return list(tokenizer.sot_sequence) + ids + [tokenizer.eot]


# @dataclass
# class Batch:
#     mel: torch.Tensor        # (B, 80, T)
#     tokens_in: torch.Tensor  # (B, L)
#     targets: torch.Tensor    # (B, L)


# def _collate(batch: List[Tuple[torch.Tensor, List[int]]], pad_id: int, device: torch.device) -> Batch:
#     mels = torch.stack([b[0] for b in batch], dim=0)
#     lengths = [len(b[1]) for b in batch]
#     L = max(lengths)
#     tok = torch.full((len(batch), L), pad_id, dtype=torch.long)
#     tgt = torch.full((len(batch), L), -100, dtype=torch.long)
#     for i, (_, ids) in enumerate(batch):
#         tok[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
#         # teacher forcing: predict next token
#         if len(ids) >= 2:
#             tgt[i, 1 : len(ids)] = torch.tensor(ids[1:], dtype=torch.long)
#     return Batch(mel=mels.to(device), tokens_in=tok.to(device), targets=tgt.to(device))


# # -----------------------------
# # Training / evaluation
# # -----------------------------

# @dataclass(frozen=True)
# class WhisperLoRATrainConfig:
#     train_dir: Path
#     dev_dir: Path
#     dev_gap_dir: Optional[Path] = None
#     out_dir: Path = Path("results/whisper_lora")

#     whisper_name: str = "small.en"
#     device: str = "cuda"

#     lr: float = 5e-5
#     weight_decay: float = 0.0
#     batch_size: int = 4
#     max_steps: int = 2000
#     eval_every: int = 200
#     patience: int = 5

#     seed: int = 70072
#     fp16: bool = True
#     lora: LoRAConfig = LoRAConfig()


# def _set_seed(seed: int) -> None:
#     torch.manual_seed(int(seed))
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(int(seed))


# @torch.no_grad()
# def decode_openai_whisper(
#     model,
#     tokenizer,
#     audio_paths: List[Path],
#     device: torch.device,
#     temperature: float = 0.0,
#     beam_size: int = 5,
#     condition_on_previous_text: bool = False,
# ) -> List[str]:
#     import whisper
#     outs: List[str] = []
#     opts = whisper.DecodingOptions(
#         language="en",
#         task="transcribe",
#         temperature=float(temperature),
#         beam_size=int(beam_size) if float(temperature) == 0.0 else None,
#         condition_on_previous_text=bool(condition_on_previous_text),
#     )
#     for p in audio_paths:
#         mel = _load_audio_mel(p, device)
#         r = whisper.decode(model, mel, opts)
#         outs.append(r.text or "")
#     return outs


# def train_whisper_lora(cfg: WhisperLoRATrainConfig) -> Path:
#     _set_seed(cfg.seed)
#     cfg.out_dir.mkdir(parents=True, exist_ok=True)

#     try:
#         import whisper
#         from whisper.tokenizer import get_tokenizer
#     except Exception as e:
#         raise RuntimeError(
#             "LoRA training requires the `openai-whisper` PyTorch package.\n"
#             "Install with: pip install -U openai-whisper\n"
#             f"Original error: {e}"
#         )

#     device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
#     model = whisper.load_model(_normalize_openai_whisper_name(cfg.whisper_name), device=str(device))

#     # Freeze everything first
#     for p in model.parameters():
#         p.requires_grad = False

#     # Apply LoRA once
#     replaced = apply_lora(model, cfg.lora)
#     params = lora_parameters(model)
#     if not params:
#         raise RuntimeError(
#             "No LoRA parameters were created.\n"
#             "Check LoRAConfig.scope_substrings/target_substrings and that they match Whisper module names."
#         )
#     for p in params:
#         p.requires_grad = True

#     model.train()

#     tokenizer = get_tokenizer(multilingual=False, language="en", task="transcribe")
#     pad_id = tokenizer.eot

#     # Optimizer
#     opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

#     # Load datasets
#     train_exs = load_dataset_dir(cfg.train_dir)
#     dev_exs = load_dataset_dir(cfg.dev_dir)
#     dev_gap_exs = load_dataset_dir(cfg.dev_gap_dir) if cfg.dev_gap_dir else []

#     # Precompute (abs_path, tokens) lists (mels computed on the fly to reduce RAM)
#     train_items = [(_resolve_audio_path(cfg.train_dir, e.audio_path), _tokenize_text(e.text, tokenizer)) for e in train_exs]
#     dev_items = [(_resolve_audio_path(cfg.dev_dir, e.audio_path), e.text) for e in dev_exs]
#     dev_gap_paths = [_resolve_audio_path(cfg.dev_gap_dir, e.audio_path) for e in dev_gap_exs] if cfg.dev_gap_dir else []

#     rng = torch.Generator().manual_seed(int(cfg.seed))

#     best = {"wer": float("inf"), "nonempty": float("inf"), "step": -1}
#     bad_epochs = 0

#     scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.fp16 and device.type == "cuda"))

#     # debug: check LoRA actually updates
#     with torch.no_grad():
#         a0 = params[0].detach().float().clone()

#     def _sample_batch() -> Batch:
#         idx = torch.randint(low=0, high=len(train_items), size=(cfg.batch_size,), generator=rng).tolist()
#         batch = []
#         for i in idx:
#             p, ids = train_items[i]
#             mel = _load_audio_mel(p, device)
#             batch.append((mel, ids))
#         return _collate(batch, pad_id=pad_id, device=device)

#     t0 = time.time()
#     print(f"[lora] replaced Linear modules: {len(replaced)}")
#     if len(replaced) <= 20:
#         for n in replaced:
#             print(f"  - {n}")
#     else:
#         print(f"  (first 10) {replaced[:10]}")

#     for step in range(1, int(cfg.max_steps) + 1):
#         b = _sample_batch()
#         opt.zero_grad(set_to_none=True)

#         with torch.cuda.amp.autocast(enabled=bool(cfg.fp16 and device.type == "cuda")):
#             logits = model(b.mel, b.tokens_in)  # (B, L, vocab)
#             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), b.targets.view(-1), ignore_index=-100)

#         scaler.scale(loss).backward()
#         scaler.unscale_(opt)
#         torch.nn.utils.clip_grad_norm_(params, 1.0)
#         scaler.step(opt)
#         scaler.update()

#         if step % 50 == 0:
#             with torch.no_grad():
#                 dA = (params[0].detach().float() - a0).abs().mean().item()
#             g = params[0].grad
#             gmean = float(g.detach().abs().mean().item()) if g is not None else 0.0
#             print(f"[step {step}] loss={loss.detach().float().item():.4f}  mean|ΔA|={dA:.3e}  mean|grad(A)|={gmean:.3e}")

#         if step % int(cfg.eval_every) == 0 or step == int(cfg.max_steps):
#             model.eval()

#             dev_paths = [p for (p, _) in dev_items]
#             dev_refs = [t for (_, t) in dev_items]
#             dev_hyps = decode_openai_whisper(model, tokenizer, dev_paths, device=device, temperature=0.0, beam_size=5)
#             dev_scores = aggregate_scores(dev_refs, dev_hyps)

#             gap_nonempty = None
#             if dev_gap_paths:
#                 gap_hyps = decode_openai_whisper(model, tokenizer, dev_gap_paths, device=device, temperature=0.0, beam_size=5)
#                 gap_nonempty = float(nonempty_rate(gap_hyps))

#             print(
#                 f"[eval step {step}] dev WER={dev_scores.wer:.4f} CER={dev_scores.cer:.4f} InsRate={dev_scores.ins_rate:.4f}"
#                 + (f" GapNonEmpty={gap_nonempty:.4f}" if gap_nonempty is not None else "")
#             )

#             improved = False
#             if dev_scores.wer < best["wer"] - 1e-6:
#                 improved = True
#             elif abs(dev_scores.wer - best["wer"]) <= 1e-6 and gap_nonempty is not None and gap_nonempty < best["nonempty"] - 1e-6:
#                 improved = True

#             if improved:
#                 best = {
#                     "wer": float(dev_scores.wer),
#                     "nonempty": float(gap_nonempty) if gap_nonempty is not None else float("inf"),
#                     "step": int(step),
#                 }
#                 bad_epochs = 0

#                 # Save a small adapter checkpoint (ONLY LoRA weights)
#                 ckpt = cfg.out_dir / "best_lora.pt"
#                 torch.save(
#                     {
#                         "step": int(step),
#                         "base_whisper_name": _normalize_openai_whisper_name(cfg.whisper_name),
#                         "lora_cfg": _jsonify(asdict(cfg.lora)),
#                         "lora_state": lora_state_dict(model),
#                     },
#                     ckpt,
#                 )

#                 (cfg.out_dir / "best_metrics.json").write_text(
#                     json_dumps(
#                         {
#                             "step": int(step),
#                             "dev": {
#                                 "wer": float(dev_scores.wer),
#                                 "cer": float(dev_scores.cer),
#                                 "ins_rate": float(dev_scores.ins_rate),
#                                 "gap_nonempty": float(gap_nonempty) if gap_nonempty is not None else None,
#                             },
#                         }
#                     ),
#                     encoding="utf-8",
#                 )
#                 print(f"[eval] NEW BEST saved adapter to {ckpt}")
#             else:
#                 bad_epochs += 1
#                 if bad_epochs >= int(cfg.patience):
#                     print(f"[early stop] no improvement for {bad_epochs} evals.")
#                     break

#             model.train()

#     wall = time.time() - t0
#     manifest = {
#         "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#         "train_dir": str(cfg.train_dir),
#         "dev_dir": str(cfg.dev_dir),
#         "dev_gap_dir": str(cfg.dev_gap_dir) if cfg.dev_gap_dir else None,
#         "whisper_name": cfg.whisper_name,
#         "device": cfg.device,
#         "lr": float(cfg.lr),
#         "weight_decay": float(cfg.weight_decay),
#         "batch_size": int(cfg.batch_size),
#         "max_steps": int(cfg.max_steps),
#         "eval_every": int(cfg.eval_every),
#         "patience": int(cfg.patience),
#         "seed": int(cfg.seed),
#         "fp16": bool(cfg.fp16),
#         "lora": _jsonify(asdict(cfg.lora)),
#         "replaced_modules": replaced,
#         "best": best,
#         "wall_sec": float(wall),
#     }
#     (cfg.out_dir / "manifest.json").write_text(json_dumps(manifest), encoding="utf-8")
#     return cfg.out_dir


# def _jsonify(x: Any) -> Any:
#     if x is None:
#         return None
#     if isinstance(x, Path):
#         return str(x)
#     if is_dataclass(x):
#         return _jsonify(asdict(x))
#     if isinstance(x, dict):
#         return {str(k): _jsonify(v) for k, v in x.items()}
#     if isinstance(x, (list, tuple)):
#         return [_jsonify(v) for v in x]
#     return x
