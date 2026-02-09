from __future__ import annotations

"""
LoRA adaptation runner for the speaker-change proxy experiment (matched same-speaker pairs + controlled augmentation).

This script:
1) Ensures LibriSpeech splits are cached (train/dev/test).
2) Samples SAME-SPEAKER pairs (x1, x2) from the <=10s core subset.
3) Builds four matched concatenation conditions with 0s gap:
   A: x1 || x2
   B: x1 || Aug(x2; p)
   C: Aug(x1; p) || x2
   D: Aug(x1; p) || Aug(x2; p)
   where p is deterministic from (seed, pair_id) and reused across B/C/D.
4) Merges selected conditions into one train/dev dataset directory (so WhisperLoRA trainer can read it).
5) Runs Whisper LoRA training using whisper_lora.py.
6) Evaluates (baseline vs LoRA-adapted) on spkproxy A/B/C/D for dev-clean and test-clean and writes a report.

Design goal: minimal disruption: adaptation is isolated here.
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .jsonl_dataset import JsonlExample, load_dataset_dir, write_jsonl
from .librispeech_cache import cache_librispeech
from .audio_transforms import read_wav, write_wav, ensure_sr, duration_sec

from .scoring import aggregate_scores
from .whisper_lora import (
    LoRAConfig,
    WhisperLoRATrainConfig,
    train_whisper_lora,
    apply_lora,
    load_lora_adapter,
    decode_openai_whisper,
    _normalize_openai_whisper_name,
)


# -----------------------------
# Determinism helpers
# -----------------------------

def _rng_for(seed: int, key: str) -> np.random.Generator:
    # stable int from key (FNV-1a-ish)
    h = 2166136261
    for ch in key.encode("utf-8"):
        h = (h ^ ch) * 16777619
        h &= 0xFFFFFFFF
    return np.random.default_rng(int((seed + h) % (2**32 - 1)))


def _peak_normalize_if_needed(x: np.ndarray, peak: float = 0.999) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m <= 1.0:
        return x.astype(np.float32)
    return (x / m * float(peak)).astype(np.float32)


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except Exception:
        try:
            dst.symlink_to(src)
        except Exception:
            import shutil
            shutil.copy2(src, dst)


# -----------------------------
# Core subset + pairing
# -----------------------------

def _load_core_subset(split_dir: Path, max_dur_sec: float = 10.0, limit: int = 0) -> List[JsonlExample]:
    exs = load_dataset_dir(split_dir)
    out = [e for e in exs if (e.duration_sec is not None and float(e.duration_sec) <= float(max_dur_sec))]
    # stable ordering for deterministic --limit-core
    out.sort(key=lambda e: str(e.id))
    if limit and limit > 0:
        out = out[: int(limit)]
    return out


def _group_by_speaker(exs: Sequence[JsonlExample]) -> Dict[str, List[JsonlExample]]:
    by: Dict[str, List[JsonlExample]] = {}
    for e in exs:
        spk = str(e.speaker) if e.speaker is not None else "UNKNOWN"
        by.setdefault(spk, []).append(e)
    return {k: v for k, v in by.items() if len(v) >= 2}


@dataclass(frozen=True)
class AugParams:
    pitch_steps: float     # semitones
    speed_rate: float      # librosa time_stretch rate (>1 faster, <1 slower)


def _sample_aug_params(seed: int, pair_id: str, pitch_k: float, speed_min: float, speed_max: float) -> AugParams:
    rng = _rng_for(seed, f"aug::{pair_id}")
    pitch = rng.uniform(-float(pitch_k), float(pitch_k))
    rate = rng.uniform(float(speed_min), float(speed_max))
    if abs(rate - 1.0) < 1e-4:
        rate = 1.0 + (1e-3 if rate >= 1.0 else -1e-3)
    return AugParams(pitch_steps=float(pitch), speed_rate=float(rate))


def _duration_after_speed(dur: float, speed_rate: float) -> float:
    # librosa.time_stretch(rate): output length ~ len / rate
    return float(dur) / float(speed_rate)


def _sample_same_speaker_pairs(
    exs: Sequence[JsonlExample],
    n_pairs: int,
    seed: int,
    max_total_sec: float,
    pitch_k: float,
    speed_min: float,
    speed_max: float,
) -> List[Tuple[JsonlExample, JsonlExample, AugParams, str]]:
    """
    Returns list of (x1, x2, aug_params, pair_id).
    Enforces max_total_sec for ALL A/B/C/D (worst-case) deterministically.
    """
    by_spk = _group_by_speaker(exs)
    speakers = sorted(by_spk.keys())
    if not speakers:
        raise RuntimeError("No speakers with >=2 utterances found in the provided subset.")

    rng = np.random.default_rng(int(seed))
    out: List[Tuple[JsonlExample, JsonlExample, AugParams, str]] = []

    attempts = 0
    max_attempts = max(10_000, int(n_pairs) * 50)

    while len(out) < int(n_pairs) and attempts < max_attempts:
        attempts += 1
        spk = speakers[int(rng.integers(0, len(speakers)))]
        pool = by_spk[spk]
        i1 = int(rng.integers(0, len(pool)))
        i2 = int(rng.integers(0, len(pool) - 1))
        if i2 >= i1:
            i2 += 1
        x1 = pool[i1]
        x2 = pool[i2]

        d1 = float(x1.duration_sec or 0.0)
        d2 = float(x2.duration_sec or 0.0)

        pair_id = f"{spk}::{x1.id}::{x2.id}"
        ap = _sample_aug_params(seed=seed, pair_id=pair_id, pitch_k=pitch_k, speed_min=speed_min, speed_max=speed_max)

        d1a = _duration_after_speed(d1, ap.speed_rate)
        d2a = _duration_after_speed(d2, ap.speed_rate)

        dA = d1 + d2
        dB = d1 + d2a
        dC = d1a + d2
        dD = d1a + d2a

        if max(dA, dB, dC, dD) > float(max_total_sec):
            continue

        out.append((x1, x2, ap, pair_id))

    if len(out) < int(n_pairs):
        raise RuntimeError(
            f"Could only sample {len(out)}/{n_pairs} valid same-speaker pairs under max_total_sec={max_total_sec}. "
            f"Try reducing n_pairs or tightening speed_min/speed_max."
        )
    return out


# -----------------------------
# Augmentation + dataset writing
# -----------------------------

def _augment_speaker_proxy(x: np.ndarray, sr: int, ap: AugParams) -> np.ndarray:
    """
    Speaker-like proxy augmentation = time-stretch + pitch-shift.
    Requires librosa.
    """
    try:
        import librosa  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Speaker-proxy augmentation requires librosa. Install it in your env (pip/conda). "
            f"Original import error: {e}"
        )

    y = x.astype(np.float32)

    # 1) speed / time-stretch (changes duration)
    if abs(ap.speed_rate - 1.0) > 1e-6:
        y = librosa.effects.time_stretch(y, rate=float(ap.speed_rate)).astype(np.float32)

    # 2) pitch shift (keeps duration)
    if abs(ap.pitch_steps) > 1e-6:
        y = librosa.effects.pitch_shift(y, sr=int(sr), n_steps=float(ap.pitch_steps)).astype(np.float32)

    return _peak_normalize_if_needed(y)


def _write_proxy_condition_dataset(
    pairs: Sequence[Tuple[JsonlExample, JsonlExample, AugParams, str]],
    out_dir: Path,
    cond: str,  # "A"|"B"|"C"|"D"
    sr: int = 16000,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    refs_rows: List[dict] = []

    for idx, (x1, x2, ap, pair_id) in enumerate(pairs):
        # load + resample
        w1, sr1 = read_wav(Path(x1.audio_path))
        w2, sr2 = read_wav(Path(x2.audio_path))
        w1, sr1 = ensure_sr(w1, sr1, target_sr=sr)
        w2, sr2 = ensure_sr(w2, sr2, target_sr=sr)
        assert sr1 == sr2 == sr

        if cond == "A":
            y1, y2 = w1, w2
        elif cond == "B":
            y1, y2 = w1, _augment_speaker_proxy(w2, sr=sr, ap=ap)
        elif cond == "C":
            y1, y2 = _augment_speaker_proxy(w1, sr=sr, ap=ap), w2
        elif cond == "D":
            y1, y2 = _augment_speaker_proxy(w1, sr=sr, ap=ap), _augment_speaker_proxy(w2, sr=sr, ap=ap)
        else:
            raise ValueError(f"Unknown condition: {cond}")

        y = np.concatenate([y1, y2]).astype(np.float32)
        y = _peak_normalize_if_needed(y)

        utt_id = f"pair_{idx:06d}__{cond}"
        wav_path = audio_dir / f"{utt_id}.wav"
        write_wav(wav_path, y, sr=sr)

        ref = f"{x1.text} {x2.text}".strip()
        refs_rows.append(
            {
                "id": utt_id,
                "audio_path": f"audio/{wav_path.name}",
                "text": ref,
                "duration_sec": float(duration_sec(y, sr)),
                "speaker": str(x1.speaker) if x1.speaker is not None else "UNKNOWN",
                "source_id_1": x1.id,
                "source_id_2": x2.id,
                "source_audio_path_1": str(x1.audio_path),
                "source_audio_path_2": str(x2.audio_path),
                "condition": cond,
                "pair_id": pair_id,
                "aug": {"pitch_steps": ap.pitch_steps, "speed_rate": ap.speed_rate},
            }
        )

    write_jsonl(out_dir / "refs.jsonl", refs_rows)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "condition": cond,
                "n_pairs": int(len(pairs)),
                "sr": int(sr),
                "gap_sec": 0.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


def _merge_datasets(cond_dirs: Sequence[Path], out_dir: Path) -> Path:
    """
    Merge multiple dataset dirs (each having refs.jsonl + audio/) into one dataset dir.
    We hardlink/copy audio into out_dir/audio and rewrite refs.jsonl accordingly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out = out_dir / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    merged: List[dict] = []
    for d in cond_dirs:
        rows = [json.loads(line) for line in (d / "refs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        for r in rows:
            src_rel = Path(r["audio_path"])
            src_abs = (d / src_rel).resolve()
            new_name = f"{d.name}__{Path(src_rel).name}"
            dst_abs = audio_out / new_name
            _safe_link_or_copy(src_abs, dst_abs)

            r2 = dict(r)
            r2["audio_path"] = f"audio/{new_name}"
            r2["merged_from"] = str(d.name)
            merged.append(r2)

    write_jsonl(out_dir / "refs.jsonl", merged)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "merged_from": [str(p.name) for p in cond_dirs],
                "n_examples": int(len(merged)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


# -----------------------------
# Evaluation helpers
# -----------------------------

def _resolve_ds_audio_path(ds_dir: Path, rel: Path) -> Path:
    if rel.is_absolute():
        return rel
    return (ds_dir / rel).resolve()


def _eval_one_ds(model, tokenizer, ds_dir: Path, device, temperature: float, beam_size: int) -> dict:
    exs = load_dataset_dir(ds_dir)
    audio_paths = [_resolve_ds_audio_path(ds_dir, e.audio_path) for e in exs]
    refs = [e.text for e in exs]
    hyps = decode_openai_whisper(
        model,
        tokenizer,
        audio_paths,
        device=device,
        temperature=float(temperature),
        beam_size=int(beam_size),
        condition_on_previous_text=False,
    )
    scores = aggregate_scores(refs, hyps)
    return {
        "wer": float(scores.wer),
        "cer": float(scores.cer),
        "ins_rate": float(scores.ins_rate),
        "n_examples": int(scores.n_examples),
        "total_ref_words": int(scores.total_ref_words),
    }


def _load_openai_whisper(base_name: str, device_str: str):
    import whisper
    from whisper.tokenizer import get_tokenizer

    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    model = whisper.load_model(_normalize_openai_whisper_name(base_name), device=str(device))
    tokenizer = get_tokenizer(multilingual=False, language="en", task="transcribe")
    return model, tokenizer, device


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", type=str, default="cache", help="Cache root (contains librispeech/, whisper/, nemo/, etc.)")
    p.add_argument("--out", type=str, required=True, help="Output directory for derived datasets + LoRA outputs")
    p.add_argument("--seed", type=int, default=70072)

    # splits
    p.add_argument("--train-split", type=str, default="train-clean-100")
    p.add_argument("--dev-split", type=str, default="dev-clean")
    p.add_argument("--test-split", type=str, default="test-clean")
    p.add_argument("--test-n-pairs", type=int, default=400)

    # subset + pairing
    p.add_argument("--max-core-dur-sec", type=float, default=10.0)
    p.add_argument("--max-total-sec", type=float, default=30.0)
    p.add_argument("--train-n-pairs", type=int, default=2000)
    p.add_argument("--dev-n-pairs", type=int, default=400)
    p.add_argument("--limit-core", type=int, default=0, help="Optional cap on core utterances (debug)")

    # augmentation ranges
    p.add_argument("--pitch-k", type=float, default=2.0, help="Pitch shift range in semitones: Uniform[-k, +k]")
    p.add_argument("--speed-min", type=float, default=0.90, help="Time-stretch rate min (<1 slower/longer)")
    p.add_argument("--speed-max", type=float, default=1.10, help="Time-stretch rate max (>1 faster/shorter)")

    # which conditions to train on (merged)
    p.add_argument("--train-conds", type=str, default="B,C,D", help="Comma-separated subset of A,B,C,D to merge for training")
    p.add_argument("--dev-conds", type=str, default="B,C,D", help="Comma-separated subset of A,B,C,D to merge for dev")

    # LoRA training knobs
    p.add_argument("--whisper-name", type=str, default="openai/whisper-small.en")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--fp16", action="store_true")

    # LoRA architecture choices (the key "encoder-only" vs "encoder+decoder" switch)
    p.add_argument("--lora-scope", type=str, default="encoder", choices=["encoder", "encdec"])
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument(
        "--lora-target",
        type=str,
        default="all",
        choices=["all", "attn"],
        help="Which Linear layers inside the chosen scope get LoRA. "
             "'all' is safer for getting improvements quickly; 'attn' is smaller.",
    )

    # Evaluation decode options (openai-whisper backend)
    p.add_argument("--eval-beam", type=int, default=5)
    p.add_argument("--eval-temperature", type=float, default=0.0)

    p.add_argument("--dry-run", action="store_true", help="Only build datasets + manifests, do not train/eval")
    args = p.parse_args()

    cache_root = Path(args.cache_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Ensure splits cached
    for split in (args.train_split, args.dev_split, args.test_split):
        split_dir = cache_root / "librispeech" / split
        if not split_dir.exists():
            cache_librispeech(cache_root, split, limit=0, seed=int(args.seed))

    # Load core subsets
    train_core = _load_core_subset(cache_root / "librispeech" / args.train_split, max_dur_sec=float(args.max_core_dur_sec), limit=int(args.limit_core))
    dev_core = _load_core_subset(cache_root / "librispeech" / args.dev_split, max_dur_sec=float(args.max_core_dur_sec), limit=int(args.limit_core))
    test_core = _load_core_subset(cache_root / "librispeech" / args.test_split, max_dur_sec=float(args.max_core_dur_sec), limit=int(args.limit_core))

    # Sample pairs
    train_pairs = _sample_same_speaker_pairs(
        train_core,
        n_pairs=int(args.train_n_pairs),
        seed=int(args.seed),
        max_total_sec=float(args.max_total_sec),
        pitch_k=float(args.pitch_k),
        speed_min=float(args.speed_min),
        speed_max=float(args.speed_max),
    )
    dev_pairs = _sample_same_speaker_pairs(
        dev_core,
        n_pairs=int(args.dev_n_pairs),
        seed=int(args.seed),
        max_total_sec=float(args.max_total_sec),
        pitch_k=float(args.pitch_k),
        speed_min=float(args.speed_min),
        speed_max=float(args.speed_max),
    )
    test_pairs = _sample_same_speaker_pairs(
        test_core,
        n_pairs=int(args.test_n_pairs),
        seed=int(args.seed),
        max_total_sec=float(args.max_total_sec),
        pitch_k=float(args.pitch_k),
        speed_min=float(args.speed_min),
        speed_max=float(args.speed_max),
    )

    # Build A/B/C/D datasets for train/dev/test
    derived_root = out_root / "proxy_datasets"
    derived_root.mkdir(parents=True, exist_ok=True)

    def _build_split(root: Path, pairs, split_tag: str) -> Dict[str, Path]:
        root.mkdir(parents=True, exist_ok=True)
        conds = ["A", "B", "C", "D"]
        out: Dict[str, Path] = {}
        for c in conds:
            out[c] = _write_proxy_condition_dataset(pairs, root / f"spkproxy_{c}", cond=c, sr=16000)
        # helpful alias names
        (root / "meta_split.json").write_text(json.dumps({"split": split_tag, "n_pairs": len(pairs)}, indent=2), encoding="utf-8")
        return out

    train_root = derived_root / args.train_split
    dev_root = derived_root / args.dev_split
    test_root = derived_root / args.test_split

    train_cond_dirs = _build_split(train_root, train_pairs, args.train_split)
    dev_cond_dirs = _build_split(dev_root, dev_pairs, args.dev_split)
    test_cond_dirs = _build_split(test_root, test_pairs, args.test_split)

    # Merge selected conditions for trainer
    conds = ["A", "B", "C", "D"]
    train_conds = [s.strip().upper() for s in str(args.train_conds).split(",") if s.strip()]
    dev_conds = [s.strip().upper() for s in str(args.dev_conds).split(",") if s.strip()]
    for s in train_conds + dev_conds:
        if s not in conds:
            raise SystemExit(f"Unknown condition in --train-conds/--dev-conds: {s}. Expected subset of A,B,C,D.")

    train_mix = _merge_datasets([train_cond_dirs[c] for c in train_conds], train_root / "train_mix")
    dev_mix = _merge_datasets([dev_cond_dirs[c] for c in dev_conds], dev_root / "dev_mix")

    # Write a run manifest
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(args.seed),
        "train_split": args.train_split,
        "dev_split": args.dev_split,
        "test_split": args.test_split,
        "max_core_dur_sec": float(args.max_core_dur_sec),
        "max_total_sec": float(args.max_total_sec),
        "train_n_pairs": int(args.train_n_pairs),
        "dev_n_pairs": int(args.dev_n_pairs),
        "test_n_pairs": int(args.test_n_pairs),
        "pitch_k": float(args.pitch_k),
        "speed_min": float(args.speed_min),
        "speed_max": float(args.speed_max),
        "train_conds": train_conds,
        "dev_conds": dev_conds,
        "train_dir": str(train_mix),
        "dev_dir": str(dev_mix),
        "whisper_name": str(args.whisper_name),
        "lora_scope": str(args.lora_scope),
        "lora_target": str(args.lora_target),
        "lora_r": int(args.lora_r),
        "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "train_hparams": {
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "max_steps": int(args.max_steps),
            "eval_every": int(args.eval_every),
            "patience": int(args.patience),
            "fp16": bool(args.fp16),
        },
        "eval_decode": {"beam": int(args.eval_beam), "temperature": float(args.eval_temperature)},
    }
    (out_root / "proxy_adapt_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"[dry-run] Built proxy datasets under: {derived_root}")
        print(f"[dry-run] Train mix: {train_mix}")
        print(f"[dry-run] Dev mix:   {dev_mix}")
        print(f"[dry-run] Test root: {test_root}")
        return 0

    # -----------------------------
    # Train LoRA
    # -----------------------------
    lora_out = out_root / "lora"
    lora_out.mkdir(parents=True, exist_ok=True)

    # Choose LoRA scope
    if args.lora_scope == "encoder":
        scope_subs = ("encoder.blocks",)
    else:
        scope_subs = ("encoder.blocks", "decoder.blocks")

    # Choose LoRA target
    if args.lora_target == "all":
        target_subs: Tuple[str, ...] = ()  # match ALL Linear layers within scope
    else:
        target_subs = ("query", "key", "value", "out")

    lora_cfg = LoRAConfig(
        r=int(args.lora_r),
        alpha=float(args.lora_alpha),
        dropout=float(args.lora_dropout),
        scope_substrings=scope_subs,
        target_substrings=target_subs,
    )

    cfg = WhisperLoRATrainConfig(
        train_dir=train_mix,
        dev_dir=dev_mix,
        dev_gap_dir=None,
        out_dir=lora_out,
        whisper_name=str(args.whisper_name),
        device=str(args.device),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        max_steps=int(args.max_steps),
        eval_every=int(args.eval_every),
        patience=int(args.patience),
        seed=int(args.seed),
        fp16=bool(args.fp16),
        lora=lora_cfg,
    )
    train_whisper_lora(cfg)

    adapter_path = lora_out / "best_lora.pt"
    if not adapter_path.exists():
        raise RuntimeError(f"Expected adapter checkpoint not found: {adapter_path}")

    # -----------------------------
    # Evaluate baseline vs LoRA on dev/test for A/B/C/D
    # -----------------------------
    import torch  # local import (keeps file import-time light)

    def _load_model(adapter: bool):
        import whisper
        from whisper.tokenizer import get_tokenizer

        device = torch.device(str(args.device) if (str(args.device) == "cpu" or torch.cuda.is_available()) else "cpu")
        base_name = _normalize_openai_whisper_name(str(args.whisper_name))
        model = whisper.load_model(base_name, device=str(device))

        tokenizer = get_tokenizer(multilingual=False, language="en", task="transcribe")

        if adapter:
            # Apply LoRA modules with SAME config, then load adapter weights
            apply_lora(model, lora_cfg)
            load_lora_adapter(model, adapter_path, device=device)

        model.eval()
        return model, tokenizer, device

    print("[eval] Loading baseline openai-whisper...")
    base_model, tok, device = _load_model(adapter=False)

    print("[eval] Loading LoRA-adapted openai-whisper...")
    lora_model, _, _ = _load_model(adapter=True)

    def _eval_split(split_name: str, cond_dirs: Dict[str, Path]) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for c in ["A", "B", "C", "D"]:
            ds_dir = cond_dirs[c]
            base_scores = _eval_one_ds(base_model, tok, ds_dir, device=device, temperature=float(args.eval_temperature), beam_size=int(args.eval_beam))
            lora_scores = _eval_one_ds(lora_model, tok, ds_dir, device=device, temperature=float(args.eval_temperature), beam_size=int(args.eval_beam))
            out[c] = {
                "baseline": base_scores,
                "lora": lora_scores,
                "delta": {k: float(lora_scores[k] - base_scores[k]) for k in ("wer", "cer", "ins_rate")},
            }
            print(
                f"[eval {split_name} {c}] "
                f"WER {base_scores['wer']:.4f} -> {lora_scores['wer']:.4f} (Δ {out[c]['delta']['wer']:+.4f})"
            )
        return out

    report = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "whisper_name": str(args.whisper_name),
        "adapter_path": str(adapter_path),
        "lora_cfg": asdict(lora_cfg),
        "train_mix": str(train_mix),
        "dev_mix": str(dev_mix),
        "eval_decode": {"beam": int(args.eval_beam), "temperature": float(args.eval_temperature)},
        "dev": _eval_split(args.dev_split, dev_cond_dirs),
        "test": _eval_split(args.test_split, test_cond_dirs),
    }

    (out_root / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] Wrote eval report: {out_root / 'eval_report.json'}")
    print(f"[done] LoRA adapter: {adapter_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())









# from __future__ import annotations

# """
# LoRA adaptation runner for the speaker-change proxy experiment (matched same-speaker pairs + controlled augmentation).

# This script:
# 1) Ensures LibriSpeech splits are cached (train/dev/test).
# 2) Samples SAME-SPEAKER pairs (x1, x2) from the <=10s core subset.
# 3) Builds four matched concatenation conditions with 0s gap:
#    A: x1 || x2
#    B: x1 || Aug(x2; p)
#    C: Aug(x1; p) || x2
#    D: Aug(x1; p) || Aug(x2; p)
#    where p is deterministic from (seed, pair_id) and reused across B/C/D.
# 4) Merges selected conditions into one train/dev dataset directory (so WhisperLoRA trainer can read it).
# 5) Runs Whisper LoRA training using whisper_lora.py.
# 6) Evaluates (baseline vs LoRA-adapted) on spkproxy A/B/C/D for dev-clean and test-clean and writes a report.

# Design goal: minimal disruption: adaptation is isolated here.
# """

# import argparse
# import json
# import time
# from dataclasses import dataclass, asdict
# from pathlib import Path
# from typing import Dict, List, Sequence, Tuple

# import numpy as np

# from .jsonl_dataset import JsonlExample, load_dataset_dir, write_jsonl
# from .librispeech_cache import cache_librispeech
# from .audio_transforms import read_wav, write_wav, ensure_sr, duration_sec

# from .scoring import aggregate_scores
# from .whisper_lora import (
#     LoRAConfig,
#     WhisperLoRATrainConfig,
#     train_whisper_lora,
#     apply_lora,
#     load_lora_adapter,
#     decode_openai_whisper,
#     _normalize_openai_whisper_name,
# )


# # -----------------------------
# # Determinism helpers
# # -----------------------------

# def _rng_for(seed: int, key: str) -> np.random.Generator:
#     # stable int from key (FNV-1a-ish)
#     h = 2166136261
#     for ch in key.encode("utf-8"):
#         h = (h ^ ch) * 16777619
#         h &= 0xFFFFFFFF
#     return np.random.default_rng(int((seed + h) % (2**32 - 1)))


# def _peak_normalize_if_needed(x: np.ndarray, peak: float = 0.999) -> np.ndarray:
#     m = float(np.max(np.abs(x))) if x.size else 0.0
#     if m <= 1.0:
#         return x.astype(np.float32)
#     return (x / m * float(peak)).astype(np.float32)


# def _safe_link_or_copy(src: Path, dst: Path) -> None:
#     dst.parent.mkdir(parents=True, exist_ok=True)
#     if dst.exists():
#         return
#     try:
#         dst.hardlink_to(src)
#     except Exception:
#         try:
#             dst.symlink_to(src)
#         except Exception:
#             import shutil
#             shutil.copy2(src, dst)


# # -----------------------------
# # Core subset + pairing
# # -----------------------------

# def _load_core_subset(split_dir: Path, max_dur_sec: float = 10.0, limit: int = 0) -> List[JsonlExample]:
#     exs = load_dataset_dir(split_dir)
#     out = [e for e in exs if (e.duration_sec is not None and float(e.duration_sec) <= float(max_dur_sec))]
#     # stable ordering for deterministic --limit-core
#     out.sort(key=lambda e: str(e.id))
#     if limit and limit > 0:
#         out = out[: int(limit)]
#     return out


# def _group_by_speaker(exs: Sequence[JsonlExample]) -> Dict[str, List[JsonlExample]]:
#     by: Dict[str, List[JsonlExample]] = {}
#     for e in exs:
#         spk = str(e.speaker) if e.speaker is not None else "UNKNOWN"
#         by.setdefault(spk, []).append(e)
#     return {k: v for k, v in by.items() if len(v) >= 2}


# @dataclass(frozen=True)
# class AugParams:
#     pitch_steps: float     # semitones
#     speed_rate: float      # librosa time_stretch rate (>1 faster, <1 slower)


# def _sample_aug_params(seed: int, pair_id: str, pitch_k: float, speed_min: float, speed_max: float) -> AugParams:
#     rng = _rng_for(seed, f"aug::{pair_id}")
#     pitch = rng.uniform(-float(pitch_k), float(pitch_k))
#     rate = rng.uniform(float(speed_min), float(speed_max))
#     if abs(rate - 1.0) < 1e-4:
#         rate = 1.0 + (1e-3 if rate >= 1.0 else -1e-3)
#     return AugParams(pitch_steps=float(pitch), speed_rate=float(rate))


# def _duration_after_speed(dur: float, speed_rate: float) -> float:
#     # librosa.time_stretch(rate): output length ~ len / rate
#     return float(dur) / float(speed_rate)


# def _sample_same_speaker_pairs(
#     exs: Sequence[JsonlExample],
#     n_pairs: int,
#     seed: int,
#     max_total_sec: float,
#     pitch_k: float,
#     speed_min: float,
#     speed_max: float,
# ) -> List[Tuple[JsonlExample, JsonlExample, AugParams, str]]:
#     """
#     Returns list of (x1, x2, aug_params, pair_id).
#     Enforces max_total_sec for ALL A/B/C/D (worst-case) deterministically.
#     """
#     by_spk = _group_by_speaker(exs)
#     speakers = sorted(by_spk.keys())
#     if not speakers:
#         raise RuntimeError("No speakers with >=2 utterances found in the provided subset.")

#     rng = np.random.default_rng(int(seed))
#     out: List[Tuple[JsonlExample, JsonlExample, AugParams, str]] = []

#     attempts = 0
#     max_attempts = max(10_000, int(n_pairs) * 50)

#     while len(out) < int(n_pairs) and attempts < max_attempts:
#         attempts += 1
#         spk = speakers[int(rng.integers(0, len(speakers)))]
#         pool = by_spk[spk]
#         i1 = int(rng.integers(0, len(pool)))
#         i2 = int(rng.integers(0, len(pool) - 1))
#         if i2 >= i1:
#             i2 += 1
#         x1 = pool[i1]
#         x2 = pool[i2]

#         d1 = float(x1.duration_sec or 0.0)
#         d2 = float(x2.duration_sec or 0.0)

#         pair_id = f"{spk}::{x1.id}::{x2.id}"
#         ap = _sample_aug_params(seed=seed, pair_id=pair_id, pitch_k=pitch_k, speed_min=speed_min, speed_max=speed_max)

#         d1a = _duration_after_speed(d1, ap.speed_rate)
#         d2a = _duration_after_speed(d2, ap.speed_rate)

#         dA = d1 + d2
#         dB = d1 + d2a
#         dC = d1a + d2
#         dD = d1a + d2a

#         if max(dA, dB, dC, dD) > float(max_total_sec):
#             continue

#         out.append((x1, x2, ap, pair_id))

#     if len(out) < int(n_pairs):
#         raise RuntimeError(
#             f"Could only sample {len(out)}/{n_pairs} valid same-speaker pairs under max_total_sec={max_total_sec}. "
#             f"Try reducing n_pairs or tightening speed_min/speed_max."
#         )
#     return out


# # -----------------------------
# # Augmentation + dataset writing
# # -----------------------------

# def _augment_speaker_proxy(x: np.ndarray, sr: int, ap: AugParams) -> np.ndarray:
#     """
#     Speaker-like proxy augmentation = time-stretch + pitch-shift.
#     Requires librosa.
#     """
#     try:
#         import librosa  # type: ignore
#     except Exception as e:
#         raise RuntimeError(
#             "Speaker-proxy augmentation requires librosa. Install it in your env (pip/conda). "
#             f"Original import error: {e}"
#         )

#     y = x.astype(np.float32)

#     # 1) speed / time-stretch (changes duration)
#     if abs(ap.speed_rate - 1.0) > 1e-6:
#         y = librosa.effects.time_stretch(y, rate=float(ap.speed_rate)).astype(np.float32)

#     # 2) pitch shift (keeps duration)
#     if abs(ap.pitch_steps) > 1e-6:
#         y = librosa.effects.pitch_shift(y, sr=int(sr), n_steps=float(ap.pitch_steps)).astype(np.float32)

#     return _peak_normalize_if_needed(y)


# def _write_proxy_condition_dataset(
#     pairs: Sequence[Tuple[JsonlExample, JsonlExample, AugParams, str]],
#     out_dir: Path,
#     cond: str,  # "A"|"B"|"C"|"D"
#     sr: int = 16000,
# ) -> Path:
#     out_dir.mkdir(parents=True, exist_ok=True)
#     audio_dir = out_dir / "audio"
#     audio_dir.mkdir(parents=True, exist_ok=True)

#     refs_rows: List[dict] = []

#     for idx, (x1, x2, ap, pair_id) in enumerate(pairs):
#         # load + resample
#         w1, sr1 = read_wav(Path(x1.audio_path))
#         w2, sr2 = read_wav(Path(x2.audio_path))
#         w1, sr1 = ensure_sr(w1, sr1, target_sr=sr)
#         w2, sr2 = ensure_sr(w2, sr2, target_sr=sr)
#         assert sr1 == sr2 == sr

#         if cond == "A":
#             y1, y2 = w1, w2
#         elif cond == "B":
#             y1, y2 = w1, _augment_speaker_proxy(w2, sr=sr, ap=ap)
#         elif cond == "C":
#             y1, y2 = _augment_speaker_proxy(w1, sr=sr, ap=ap), w2
#         elif cond == "D":
#             y1, y2 = _augment_speaker_proxy(w1, sr=sr, ap=ap), _augment_speaker_proxy(w2, sr=sr, ap=ap)
#         else:
#             raise ValueError(f"Unknown condition: {cond}")

#         y = np.concatenate([y1, y2]).astype(np.float32)
#         y = _peak_normalize_if_needed(y)

#         utt_id = f"pair_{idx:06d}__{cond}"
#         wav_path = audio_dir / f"{utt_id}.wav"
#         write_wav(wav_path, y, sr=sr)

#         ref = f"{x1.text} {x2.text}".strip()
#         refs_rows.append(
#             {
#                 "id": utt_id,
#                 "audio_path": f"audio/{wav_path.name}",
#                 "text": ref,
#                 "duration_sec": float(duration_sec(y, sr)),
#                 "speaker": str(x1.speaker) if x1.speaker is not None else "UNKNOWN",
#                 "source_id_1": x1.id,
#                 "source_id_2": x2.id,
#                 "source_audio_path_1": str(x1.audio_path),
#                 "source_audio_path_2": str(x2.audio_path),
#                 "condition": cond,
#                 "pair_id": pair_id,
#                 "aug": {"pitch_steps": ap.pitch_steps, "speed_rate": ap.speed_rate},
#             }
#         )

#     write_jsonl(out_dir / "refs.jsonl", refs_rows)
#     (out_dir / "meta.json").write_text(
#         json.dumps(
#             {
#                 "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#                 "condition": cond,
#                 "n_pairs": int(len(pairs)),
#                 "sr": int(sr),
#                 "gap_sec": 0.0,
#             },
#             indent=2,
#         ),
#         encoding="utf-8",
#     )
#     return out_dir


# def _merge_datasets(cond_dirs: Sequence[Path], out_dir: Path) -> Path:
#     """
#     Merge multiple dataset dirs (each having refs.jsonl + audio/) into one dataset dir.
#     We hardlink/copy audio into out_dir/audio and rewrite refs.jsonl accordingly.
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)
#     audio_out = out_dir / "audio"
#     audio_out.mkdir(parents=True, exist_ok=True)

#     merged: List[dict] = []
#     for d in cond_dirs:
#         rows = [json.loads(line) for line in (d / "refs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
#         for r in rows:
#             src_rel = Path(r["audio_path"])
#             src_abs = (d / src_rel).resolve()
#             new_name = f"{d.name}__{Path(src_rel).name}"
#             dst_abs = audio_out / new_name
#             _safe_link_or_copy(src_abs, dst_abs)

#             r2 = dict(r)
#             r2["audio_path"] = f"audio/{new_name}"
#             r2["merged_from"] = str(d.name)
#             merged.append(r2)

#     write_jsonl(out_dir / "refs.jsonl", merged)
#     (out_dir / "meta.json").write_text(
#         json.dumps(
#             {
#                 "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#                 "merged_from": [str(p.name) for p in cond_dirs],
#                 "n_examples": int(len(merged)),
#             },
#             indent=2,
#         ),
#         encoding="utf-8",
#     )
#     return out_dir


# # -----------------------------
# # Evaluation helpers
# # -----------------------------

# def _resolve_ds_audio_path(ds_dir: Path, rel: Path) -> Path:
#     if rel.is_absolute():
#         return rel
#     return (ds_dir / rel).resolve()


# def _eval_one_ds(model, tokenizer, ds_dir: Path, device, temperature: float, beam_size: int) -> dict:
#     exs = load_dataset_dir(ds_dir)
#     audio_paths = [_resolve_ds_audio_path(ds_dir, e.audio_path) for e in exs]
#     refs = [e.text for e in exs]
#     hyps = decode_openai_whisper(
#         model,
#         tokenizer,
#         audio_paths,
#         device=device,
#         temperature=float(temperature),
#         beam_size=int(beam_size),
#         condition_on_previous_text=False,
#     )
#     scores = aggregate_scores(refs, hyps)
#     return {
#         "wer": float(scores.wer),
#         "cer": float(scores.cer),
#         "ins_rate": float(scores.ins_rate),
#         "n_examples": int(scores.n_examples),
#         "total_ref_words": int(scores.total_ref_words),
#     }


# def _load_openai_whisper(base_name: str, device_str: str):
#     import whisper
#     from whisper.tokenizer import get_tokenizer

#     device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
#     model = whisper.load_model(_normalize_openai_whisper_name(base_name), device=str(device))
#     tokenizer = get_tokenizer(multilingual=False, language="en", task="transcribe")
#     return model, tokenizer, device


# # -----------------------------
# # Main
# # -----------------------------

# def main() -> int:
#     p = argparse.ArgumentParser()
#     p.add_argument("--cache-root", type=str, default="cache", help="Cache root (contains librispeech/, whisper/, nemo/, etc.)")
#     p.add_argument("--out", type=str, required=True, help="Output directory for derived datasets + LoRA outputs")
#     p.add_argument("--seed", type=int, default=70072)

#     # splits
#     p.add_argument("--train-split", type=str, default="train-clean-100")
#     p.add_argument("--dev-split", type=str, default="dev-clean")
#     p.add_argument("--test-split", type=str, default="test-clean")
#     p.add_argument("--test-n-pairs", type=int, default=400)

#     # subset + pairing
#     p.add_argument("--max-core-dur-sec", type=float, default=10.0)
#     p.add_argument("--max-total-sec", type=float, default=30.0)
#     p.add_argument("--train-n-pairs", type=int, default=2000)
#     p.add_argument("--dev-n-pairs", type=int, default=400)
#     p.add_argument("--limit-core", type=int, default=0, help="Optional cap on core utterances (debug)")

#     # augmentation ranges
#     p.add_argument("--pitch-k", type=float, default=2.0, help="Pitch shift range in semitones: Uniform[-k, +k]")
#     p.add_argument("--speed-min", type=float, default=0.90, help="Time-stretch rate min (<1 slower/longer)")
#     p.add_argument("--speed-max", type=float, default=1.10, help="Time-stretch rate max (>1 faster/shorter)")

#     # which conditions to train on (merged)
#     p.add_argument("--train-conds", type=str, default="B,C,D", help="Comma-separated subset of A,B,C,D to merge for training")
#     p.add_argument("--dev-conds", type=str, default="B,C,D", help="Comma-separated subset of A,B,C,D to merge for dev")

#     # LoRA training knobs
#     p.add_argument("--whisper-name", type=str, default="openai/whisper-small.en")
#     p.add_argument("--device", type=str, default="cuda")
#     p.add_argument("--lr", type=float, default=1e-4)
#     p.add_argument("--weight-decay", type=float, default=0.0)
#     p.add_argument("--batch-size", type=int, default=2)
#     p.add_argument("--max-steps", type=int, default=1500)
#     p.add_argument("--eval-every", type=int, default=200)
#     p.add_argument("--patience", type=int, default=4)
#     p.add_argument("--fp16", action="store_true")

#     # LoRA architecture choices (the key "encoder-only" vs "encoder+decoder" switch)
#     p.add_argument("--lora-scope", type=str, default="encoder", choices=["encoder", "encdec"])
#     p.add_argument("--lora-r", type=int, default=16)
#     p.add_argument("--lora-alpha", type=float, default=32.0)
#     p.add_argument("--lora-dropout", type=float, default=0.0)
#     p.add_argument(
#         "--lora-target",
#         type=str,
#         default="all",
#         choices=["all", "attn"],
#         help="Which Linear layers inside the chosen scope get LoRA. "
#              "'all' is safer for getting improvements quickly; 'attn' is smaller.",
#     )

#     # Evaluation decode options (openai-whisper backend)
#     p.add_argument("--eval-beam", type=int, default=5)
#     p.add_argument("--eval-temperature", type=float, default=0.0)

#     p.add_argument("--dry-run", action="store_true", help="Only build datasets + manifests, do not train/eval")
#     args = p.parse_args()

#     cache_root = Path(args.cache_root)
#     out_root = Path(args.out)
#     out_root.mkdir(parents=True, exist_ok=True)

#     # Ensure splits cached
#     for split in (args.train_split, args.dev_split, args.test_split):
#         split_dir = cache_root / "librispeech" / split
#         if not split_dir.exists():
#             cache_librispeech(cache_root, split, limit=0, seed=int(args.seed))

#     # Load core subsets
#     train_core = _load_core_subset(cache_root / "librispeech" / args.train_split, max_dur_sec=float(args.max_core_dur_sec), limit=int(args.limit_core))
#     dev_core = _load_core_subset(cache_root / "librispeech" / args.dev_split, max_dur_sec=float(args.max_core_dur_sec), limit=int(args.limit_core))
#     test_core = _load_core_subset(cache_root / "librispeech" / args.test_split, max_dur_sec=float(args.max_core_dur_sec), limit=int(args.limit_core))

#     # Sample pairs
#     train_pairs = _sample_same_speaker_pairs(
#         train_core,
#         n_pairs=int(args.train_n_pairs),
#         seed=int(args.seed),
#         max_total_sec=float(args.max_total_sec),
#         pitch_k=float(args.pitch_k),
#         speed_min=float(args.speed_min),
#         speed_max=float(args.speed_max),
#     )
#     dev_pairs = _sample_same_speaker_pairs(
#         dev_core,
#         n_pairs=int(args.dev_n_pairs),
#         seed=int(args.seed),
#         max_total_sec=float(args.max_total_sec),
#         pitch_k=float(args.pitch_k),
#         speed_min=float(args.speed_min),
#         speed_max=float(args.speed_max),
#     )
#     test_pairs = _sample_same_speaker_pairs(
#         test_core,
#         n_pairs=int(args.test_n_pairs),
#         seed=int(args.seed),
#         max_total_sec=float(args.max_total_sec),
#         pitch_k=float(args.pitch_k),
#         speed_min=float(args.speed_min),
#         speed_max=float(args.speed_max),
#     )

#     # Build A/B/C/D datasets for train/dev/test
#     derived_root = out_root / "proxy_datasets"
#     derived_root.mkdir(parents=True, exist_ok=True)

#     def _build_split(root: Path, pairs, split_tag: str) -> Dict[str, Path]:
#         root.mkdir(parents=True, exist_ok=True)
#         conds = ["A", "B", "C", "D"]
#         out: Dict[str, Path] = {}
#         for c in conds:
#             out[c] = _write_proxy_condition_dataset(pairs, root / f"spkproxy_{c}", cond=c, sr=16000)
#         # helpful alias names
#         (root / "meta_split.json").write_text(json.dumps({"split": split_tag, "n_pairs": len(pairs)}, indent=2), encoding="utf-8")
#         return out

#     train_root = derived_root / args.train_split
#     dev_root = derived_root / args.dev_split
#     test_root = derived_root / args.test_split

#     train_cond_dirs = _build_split(train_root, train_pairs, args.train_split)
#     dev_cond_dirs = _build_split(dev_root, dev_pairs, args.dev_split)
#     test_cond_dirs = _build_split(test_root, test_pairs, args.test_split)

#     # Merge selected conditions for trainer
#     conds = ["A", "B", "C", "D"]
#     train_conds = [s.strip().upper() for s in str(args.train_conds).split(",") if s.strip()]
#     dev_conds = [s.strip().upper() for s in str(args.dev_conds).split(",") if s.strip()]
#     for s in train_conds + dev_conds:
#         if s not in conds:
#             raise SystemExit(f"Unknown condition in --train-conds/--dev-conds: {s}. Expected subset of A,B,C,D.")

#     train_mix = _merge_datasets([train_cond_dirs[c] for c in train_conds], train_root / "train_mix")
#     dev_mix = _merge_datasets([dev_cond_dirs[c] for c in dev_conds], dev_root / "dev_mix")

#     # Write a run manifest
#     manifest = {
#         "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#         "seed": int(args.seed),
#         "train_split": args.train_split,
#         "dev_split": args.dev_split,
#         "test_split": args.test_split,
#         "max_core_dur_sec": float(args.max_core_dur_sec),
#         "max_total_sec": float(args.max_total_sec),
#         "train_n_pairs": int(args.train_n_pairs),
#         "dev_n_pairs": int(args.dev_n_pairs),
#         "test_n_pairs": int(args.test_n_pairs),
#         "pitch_k": float(args.pitch_k),
#         "speed_min": float(args.speed_min),
#         "speed_max": float(args.speed_max),
#         "train_conds": train_conds,
#         "dev_conds": dev_conds,
#         "train_dir": str(train_mix),
#         "dev_dir": str(dev_mix),
#         "whisper_name": str(args.whisper_name),
#         "lora_scope": str(args.lora_scope),
#         "lora_target": str(args.lora_target),
#         "lora_r": int(args.lora_r),
#         "lora_alpha": float(args.lora_alpha),
#         "lora_dropout": float(args.lora_dropout),
#         "train_hparams": {
#             "lr": float(args.lr),
#             "batch_size": int(args.batch_size),
#             "max_steps": int(args.max_steps),
#             "eval_every": int(args.eval_every),
#             "patience": int(args.patience),
#             "fp16": bool(args.fp16),
#         },
#         "eval_decode": {"beam": int(args.eval_beam), "temperature": float(args.eval_temperature)},
#     }
#     (out_root / "proxy_adapt_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

#     if args.dry_run:
#         print(f"[dry-run] Built proxy datasets under: {derived_root}")
#         print(f"[dry-run] Train mix: {train_mix}")
#         print(f"[dry-run] Dev mix:   {dev_mix}")
#         print(f"[dry-run] Test root: {test_root}")
#         return 0

#     # -----------------------------
#     # Train LoRA
#     # -----------------------------
#     lora_out = out_root / "lora"
#     lora_out.mkdir(parents=True, exist_ok=True)

#     # Choose LoRA scope
#     if args.lora_scope == "encoder":
#         scope_subs = ("encoder.blocks",)
#     else:
#         scope_subs = ("encoder.blocks", "decoder.blocks")

#     # Choose LoRA target
#     if args.lora_target == "all":
#         target_subs: Tuple[str, ...] = ()  # match ALL Linear layers within scope
#     else:
#         target_subs = ("query", "key", "value", "out")

#     lora_cfg = LoRAConfig(
#         r=int(args.lora_r),
#         alpha=float(args.lora_alpha),
#         dropout=float(args.lora_dropout),
#         scope_substrings=scope_subs,
#         target_substrings=target_subs,
#     )

#     cfg = WhisperLoRATrainConfig(
#         train_dir=train_mix,
#         dev_dir=dev_mix,
#         dev_gap_dir=None,
#         out_dir=lora_out,
#         whisper_name=str(args.whisper_name),
#         device=str(args.device),
#         lr=float(args.lr),
#         weight_decay=float(args.weight_decay),
#         batch_size=int(args.batch_size),
#         max_steps=int(args.max_steps),
#         eval_every=int(args.eval_every),
#         patience=int(args.patience),
#         seed=int(args.seed),
#         fp16=bool(args.fp16),
#         lora=lora_cfg,
#     )
#     train_whisper_lora(cfg)

#     adapter_path = lora_out / "best_lora.pt"
#     if not adapter_path.exists():
#         raise RuntimeError(f"Expected adapter checkpoint not found: {adapter_path}")

#     # -----------------------------
#     # Evaluate baseline vs LoRA on dev/test for A/B/C/D
#     # -----------------------------
#     import torch  # local import (keeps file import-time light)

#     def _load_model(adapter: bool):
#         import whisper
#         from whisper.tokenizer import get_tokenizer

#         device = torch.device(str(args.device) if (str(args.device) == "cpu" or torch.cuda.is_available()) else "cpu")
#         base_name = _normalize_openai_whisper_name(str(args.whisper_name))
#         model = whisper.load_model(base_name, device=str(device))

#         tokenizer = get_tokenizer(multilingual=False, language="en", task="transcribe")

#         if adapter:
#             # Apply LoRA modules with SAME config, then load adapter weights
#             apply_lora(model, lora_cfg)
#             load_lora_adapter(model, adapter_path, device=device)

#         model.eval()
#         return model, tokenizer, device

#     print("[eval] Loading baseline openai-whisper...")
#     base_model, tok, device = _load_model(adapter=False)

#     print("[eval] Loading LoRA-adapted openai-whisper...")
#     lora_model, _, _ = _load_model(adapter=True)

#     def _eval_split(split_name: str, cond_dirs: Dict[str, Path]) -> Dict[str, dict]:
#         out: Dict[str, dict] = {}
#         for c in ["A", "B", "C", "D"]:
#             ds_dir = cond_dirs[c]
#             base_scores = _eval_one_ds(base_model, tok, ds_dir, device=device, temperature=float(args.eval_temperature), beam_size=int(args.eval_beam))
#             lora_scores = _eval_one_ds(lora_model, tok, ds_dir, device=device, temperature=float(args.eval_temperature), beam_size=int(args.eval_beam))
#             out[c] = {
#                 "baseline": base_scores,
#                 "lora": lora_scores,
#                 "delta": {k: float(lora_scores[k] - base_scores[k]) for k in ("wer", "cer", "ins_rate")},
#             }
#             print(
#                 f"[eval {split_name} {c}] "
#                 f"WER {base_scores['wer']:.4f} -> {lora_scores['wer']:.4f} (Δ {out[c]['delta']['wer']:+.4f})"
#             )
#         return out

#     report = {
#         "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#         "whisper_name": str(args.whisper_name),
#         "adapter_path": str(adapter_path),
#         "lora_cfg": asdict(lora_cfg),
#         "train_mix": str(train_mix),
#         "dev_mix": str(dev_mix),
#         "eval_decode": {"beam": int(args.eval_beam), "temperature": float(args.eval_temperature)},
#         "dev": _eval_split(args.dev_split, dev_cond_dirs),
#         "test": _eval_split(args.test_split, test_cond_dirs),
#     }

#     (out_root / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
#     print(f"[done] Wrote eval report: {out_root / 'eval_report.json'}")
#     print(f"[done] LoRA adapter: {adapter_path}")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())
