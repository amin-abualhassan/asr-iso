from __future__ import annotations

"""
Sequential multi-speaker (concatenation) dataset construction.

This module implements Condition (6) from the methodology and is also used
by the adaptation scripts to build 1s-gap concatenation datasets.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import time

# from .audio_transforms import GapSpec, concat_with_gap, duration_sec, ensure_sr, read_wav, write_wav
from .audio_transforms import (
    GapSpec,
    concat_with_gap,
    duration_sec,
    ensure_sr,
    read_wav,
    write_wav,
    apply_speaker_aug,
    make_speaker_aug_params,
    sample_speaker_aug_params,
)



from .jsonl_dataset import JsonlExample, write_jsonl


@dataclass(frozen=True)
class ConcatSpec:
    same_speaker: bool
    gap_kind: str  # "silence" | "noise"
    gap_sec: float
    noise_snr_db: float = 10.0
    n_pairs: int = 1000
    max_total_sec: float = 30.0  # Whisper single-window constraint
    sr: int = 16000


def _group_by_speaker(exs: List[JsonlExample]) -> Dict[str, List[JsonlExample]]:
    m: Dict[str, List[JsonlExample]] = {}
    for e in exs:
        spk = e.speaker or "UNKNOWN"
        m.setdefault(spk, []).append(e)
    return m


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def sample_pairs(exs: List[JsonlExample], same_speaker: bool, n_pairs: int, seed: int) -> List[Tuple[JsonlExample, JsonlExample]]:
    """
    Sample utterance pairs with speaker constraint.
    Best-effort "without replacement" by tracking used examples, but will fall back
    to re-use if the pool is too small.
    """
    rng = _rng(seed)
    by_spk = _group_by_speaker(exs)

    pairs: List[Tuple[JsonlExample, JsonlExample]] = []
    used = set()

    if same_speaker:
        # Build candidate pairs within each speaker
        speakers = [s for s, lst in by_spk.items() if len(lst) >= 2]
        if not speakers:
            raise ValueError("No speaker with >=2 utterances available for same-speaker pairing.")
        rng.shuffle(speakers)

        # Round-robin: shuffle each speaker's utterances then pair adjacent.
        candidate_pairs: List[Tuple[JsonlExample, JsonlExample]] = []
        for spk in speakers:
            lst = by_spk[spk].copy()
            rng.shuffle(lst)
            for i in range(0, len(lst) - 1, 2):
                candidate_pairs.append((lst[i], lst[i + 1]))
        rng.shuffle(candidate_pairs)

        # Prefer without replacement
        for a, b in candidate_pairs:
            if len(pairs) >= n_pairs:
                break
            if a.id in used or b.id in used:
                continue
            pairs.append((a, b))
            used.add(a.id); used.add(b.id)

        # If not enough, allow re-use by random sampling within speakers
        while len(pairs) < n_pairs:
            spk = rng.choice(speakers)
            lst = by_spk[spk]
            a, b = rng.choice(lst, size=2, replace=False).tolist()
            pairs.append((a, b))

    else:
        # different-speaker pairs
        all_idx = np.arange(len(exs))
        rng.shuffle(all_idx)
        # greedy without replacement attempts
        for i in range(0, len(all_idx) - 1, 2):
            if len(pairs) >= n_pairs:
                break
            a = exs[int(all_idx[i])]
            b = exs[int(all_idx[i + 1])]
            if (a.speaker or "UNKNOWN") == (b.speaker or "UNKNOWN"):
                continue
            if a.id in used or b.id in used:
                continue
            pairs.append((a, b))
            used.add(a.id); used.add(b.id)

        # fallback: random sampling until enough
        while len(pairs) < n_pairs:
            a = exs[int(rng.integers(0, len(exs)))]
            b = exs[int(rng.integers(0, len(exs)))]
            if a.id == b.id:
                continue
            if (a.speaker or "UNKNOWN") == (b.speaker or "UNKNOWN"):
                continue
            pairs.append((a, b))

    return pairs


def build_concat_dataset(
    exs: List[JsonlExample],
    out_dir: Path,
    spec: ConcatSpec,
    seed: int,
) -> Path:
    """
    Build a dataset directory with concatenated audios and refs.jsonl.

    - exs: should already be the "core subset" (<=10s utterances) per methodology.
    - Gap audio (for gap_sec>0) is also written and referenced from nonspeech_refs.jsonl
      so Tail/GAP Non-Empty Rate can transcribe the gap segment in isolation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    pairs = sample_pairs(exs, same_speaker=spec.same_speaker, n_pairs=int(spec.n_pairs), seed=int(seed))

    gap = GapSpec(
        kind=str(spec.gap_kind),
        duration_sec=float(spec.gap_sec),
        noise_snr_db=float(spec.noise_snr_db) if spec.gap_kind == "noise" else None,
    )

    refs_rows: List[dict] = []
    nonspeech_rows: List[dict] = []

    n_written = 0
    for k, (a, b) in enumerate(pairs):
        # load audio
        x1, sr1 = read_wav(a.audio_path)
        x2, sr2 = read_wav(b.audio_path)
        x1, sr1 = ensure_sr(x1, sr1, target_sr=spec.sr)
        x2, sr2 = ensure_sr(x2, sr2, target_sr=spec.sr)
        assert sr1 == sr2 == spec.sr

        # concat with gap; noise gap SNR relative to first utterance RMS
        y, g = concat_with_gap(
            x1=x1,
            x2=x2,
            sr=spec.sr,
            gap=gap,
            seed=int(seed),
            key=f"{a.id}__{b.id}__{spec.gap_kind}{spec.gap_sec}",
            ref_rms_for_noise=None,
        )

        # enforce Whisper single-window constraint
        if duration_sec(y, spec.sr) > float(spec.max_total_sec):
            continue  # safe skip; spec should normally prevent this

        pair_id = f"pair_{k:05d}__{a.id}__{b.id}__{'same' if spec.same_speaker else 'diff'}__{spec.gap_kind}{int(spec.gap_sec)}s"
        wav_path = audio_dir / f"{pair_id}.wav"
        write_wav(wav_path, y, sr=spec.sr)

        # write gap-only probe if applicable (gap_sec>0)
        if g is not None and len(g) > 0:
            gap_path = audio_dir / f"{pair_id}__gap.wav"
            write_wav(gap_path, g, sr=spec.sr)
            nonspeech_rows.append(
                {
                    "id": pair_id,
                    "audio_path": f"audio/{gap_path.name}",
                    "kind": "gap",
                    "duration_sec": float(spec.gap_sec),
                }
            )

        refs_rows.append(
            {
                "id": pair_id,
                "audio_path": f"audio/{wav_path.name}",
                "text": f"{a.text} {b.text}".strip(),
                "duration_sec": float(duration_sec(y, spec.sr)),
                "speaker": f"{a.speaker or 'UNKNOWN'}+{b.speaker or 'UNKNOWN'}",
                "utt1_id": a.id,
                "utt2_id": b.id,
                "utt1_speaker": a.speaker,
                "utt2_speaker": b.speaker,
                "gap_kind": spec.gap_kind,
                "gap_sec": float(spec.gap_sec),
                "same_speaker": bool(spec.same_speaker),
            }
        )
        n_written += 1
        if n_written >= int(spec.n_pairs):
            break

    write_jsonl(out_dir / "refs.jsonl", refs_rows)
    if nonspeech_rows:
        write_jsonl(out_dir / "nonspeech_refs.jsonl", nonspeech_rows)

    (out_dir / "meta.json").write_text(
        json_dumps(
            {
                "condition": "concat",
                "same_speaker": bool(spec.same_speaker),
                "gap_kind": str(spec.gap_kind),
                "gap_sec": float(spec.gap_sec),
                "noise_snr_db": float(spec.noise_snr_db),
                "n_pairs_requested": int(spec.n_pairs),
                "n_written": int(n_written),
                "seed": int(seed),
                "max_total_sec": float(spec.max_total_sec),
                "sr": int(spec.sr),
            }
        ),
        encoding="utf-8",
    )
    return out_dir



# -----------------------------
# Speaker-change proxy via matched pairs + controlled augmentation (A/B/C/D)
# -----------------------------

@dataclass(frozen=True)
class SpeakerProxySpec:
    """Spec for the speaker-change proxy experiment family."""
    n_pairs: int = 1000
    max_total_sec: float = 30.0
    sr: int = 16000

    # speaker-like augmentation ranges
    pitch_max_semitones: float = 3.0
    rate_min: float = 0.9
    rate_max: float = 1.1


def build_speaker_proxy_datasets(
    exs: List[JsonlExample],
    out_root: Path,
    spec: SpeakerProxySpec,
    seed: int,
) -> Dict[str, Path]:
    """
    Build four *matched* concatenation datasets from same-speaker pairs:

      A: x1 || x2
      B: x1 || Aug(x2; p)
      C: Aug(x1; p) || x2
      D: Aug(x1; p) || Aug(x2; p)

    Notes:
    - No gap is inserted (0s), so there is no nonspeech_refs.jsonl.
    - For each pair, the same augmentation parameter set p is reused across B/C/D.
    - Pair acceptance is gated on *all four* variants satisfying max_total_sec to keep
      the evaluated pair set consistent regardless of which conditions are run later.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    # Condition directory names (these are the condition IDs used elsewhere).
    cond_names = {
        "A": "spkproxy__A__clean_clean",
        "B": "spkproxy__B__clean_aug",
        "C": "spkproxy__C__aug_clean",
        "D": "spkproxy__D__aug_aug",
    }

    # Prepare dirs
    audio_dirs: Dict[str, Path] = {}
    rows: Dict[str, List[dict]] = {k: [] for k in cond_names.keys()}

    for k, cname in cond_names.items():
        cdir = out_root / cname
        cdir.mkdir(parents=True, exist_ok=True)
        adir = cdir / "audio"
        adir.mkdir(parents=True, exist_ok=True)
        audio_dirs[k] = adir

    # Over-sample candidate pairs (same strategy as build_concat_dataset)
    candidates = sample_pairs(exs, same_speaker=True, n_pairs=int(spec.n_pairs) * 5, seed=int(seed))

    n_written = 0
    for a, b in candidates:
        if n_written >= int(spec.n_pairs):
            break

        x1, sr1 = read_wav(a.audio_path)
        x2, sr2 = read_wav(b.audio_path)
        x1, _ = ensure_sr(x1, sr1, target_sr=int(spec.sr))
        x2, _ = ensure_sr(x2, sr2, target_sr=int(spec.sr))
        sr = int(spec.sr)

        # Deterministic params per (seed, pair ids)
        pair_key = f"{a.id}__{b.id}"
        p = make_speaker_aug_params(
            seed=int(seed),
            key=pair_key,
            pitch_max_semitones=float(spec.pitch_max_semitones),
            rate_min=float(spec.rate_min),
            rate_max=float(spec.rate_max),
        )

        # Augmented segments
        x1a = apply_speaker_aug(x1, sr=sr, params=p)
        x2a = apply_speaker_aug(x2, sr=sr, params=p)

        # Construct all four variants
        yA = np.concatenate([x1, x2]).astype(np.float32)
        yB = np.concatenate([x1, x2a]).astype(np.float32)
        yC = np.concatenate([x1a, x2]).astype(np.float32)
        yD = np.concatenate([x1a, x2a]).astype(np.float32)

        # Gate acceptance on all four variants meeting the duration constraint
        if any(duration_sec(y, sr) > float(spec.max_total_sec) for y in (yA, yB, yC, yD)):
            continue

        base_id = f"pair_{n_written:05d}__{a.id}__{b.id}"
        ref_text = f"{a.text} {b.text}".strip()

        # write each condition
        for variant, y in (("A", yA), ("B", yB), ("C", yC), ("D", yD)):
            cname = cond_names[variant]
            utt_id = f"{base_id}__{cname}"
            wav_path = audio_dirs[variant] / f"{utt_id}.wav"
            write_wav(wav_path, y, sr=sr)

            rows[variant].append(
                {
                    "id": utt_id,
                    "audio_path": f"audio/{wav_path.name}",
                    "text": ref_text,
                    "duration_sec": float(duration_sec(y, sr)),
                    "speaker": a.speaker,  # same by construction
                    "utt1_id": a.id,
                    "utt2_id": b.id,
                    "utt1_audio_path": str(a.audio_path),
                    "utt2_audio_path": str(b.audio_path),
                    "aug_params": {"pitch_semitones": float(p.pitch_semitones), "rate": float(p.rate)},
                    "condition": cname,
                }
            )

        n_written += 1

    # Write jsonl + meta
    created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for variant, cname in cond_names.items():
        cdir = out_root / cname
        write_jsonl(cdir / "refs.jsonl", rows[variant])

        meta = {
            "created_utc": created_utc,
            "kind": "speaker_proxy_concat",
            "condition": cname,
            "seed": int(seed),
            "n_pairs_target": int(spec.n_pairs),
            "n_written": int(len(rows[variant])),
            "max_total_sec": float(spec.max_total_sec),
            "sr": int(spec.sr),
            "pitch_max_semitones": float(spec.pitch_max_semitones),
            "rate_min": float(spec.rate_min),
            "rate_max": float(spec.rate_max),
            "pairing": "same_speaker",
            "gap_sec": 0.0,
        }
        (cdir / "meta.json").write_text(json_dumps(meta), encoding="utf-8")

    # Return mapping from condition name to its directory path (for eval_runner)
    return {cname: (out_root / cname) for cname in cond_names.values()}




def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False)


# -----------------------------
# Speaker-change proxy datasets (matched same-speaker pairs + controlled augmentation)
# -----------------------------

@dataclass(frozen=True)
class SpkProxySpec:
    """Specification for the speaker-change proxy experiment family."""
    n_pairs: int = 1000
    max_total_sec: float = 30.0
    sr: int = 16000

    # Augmentation parameter ranges
    pitch_range: Tuple[float, float] = (-4.0, 4.0)  # semitones
    rate_range: Tuple[float, float] = (0.9, 1.1)    # tempo factor

    # Keep augmented segments the same length as the originals (recommended)
    keep_len: bool = True


def _sample_same_speaker_pairs_with_max_dur(
    exs: List[JsonlExample],
    n_pairs: int,
    seed: int,
    max_total_sec: float,
) -> List[Tuple[JsonlExample, JsonlExample]]:
    """Sample (x1, x2) from the same speaker, enforcing d(x1)+d(x2) <= max_total_sec."""
    rng = _rng(seed)
    by_spk = _group_by_speaker(exs)
    speakers = [s for s, lst in by_spk.items() if len(lst) >= 2]
    if not speakers:
        raise ValueError("No speakers with >=2 utterances available for spkproxy sampling.")

    pairs: List[Tuple[JsonlExample, JsonlExample]] = []
    attempts = 0
    max_attempts = max(10_000, int(n_pairs) * 50)

    # Best-effort without replacement.
    used = set()

    def _dur(e: JsonlExample) -> float:
        if e.duration_sec is not None:
            return float(e.duration_sec)
        x, sr = read_wav(e.audio_path)
        return float(duration_sec(x, sr))

    while len(pairs) < int(n_pairs) and attempts < max_attempts:
        attempts += 1
        spk = str(rng.choice(speakers))
        lst = by_spk[spk]

        cand = [e for e in lst if e.id not in used]
        pool = cand if len(cand) >= 2 else lst

        a, b = rng.choice(pool, size=2, replace=False).tolist()
        if float(_dur(a) + _dur(b)) > float(max_total_sec):
            continue

        pairs.append((a, b))
        used.add(a.id); used.add(b.id)

    if len(pairs) < int(n_pairs):
        raise RuntimeError(f"Could not sample enough pairs under duration constraint: {len(pairs)}/{n_pairs}.")
    return pairs


def build_spkproxy_datasets(
    exs: List[JsonlExample],
    out_root: Path,
    spec: SpkProxySpec,
    seed: int,
    variants: Tuple[str, ...] = ("A", "B", "C", "D"),
) -> Dict[str, Path]:
    """Build A/B/C/D speaker-change proxy datasets from SAME-speaker matched pairs.

    Conditions:
      A: x1 + x2
      B: x1 + Aug(x2; p)
      C: Aug(x1; p) + x2
      D: Aug(x1; p) + Aug(x2; p)

    The same (p) is reused across B/C/D for each pair.
    """
    import json, time  # local to avoid disturbing your existing imports

    out_root.mkdir(parents=True, exist_ok=True)

    pairs = _sample_same_speaker_pairs_with_max_dur(
        exs=exs,
        n_pairs=int(spec.n_pairs),
        seed=int(seed),
        max_total_sec=float(spec.max_total_sec),
    )

    # store sampled pairs + params for reproducibility
    pair_rows: List[dict] = []
    for i, (a, b) in enumerate(pairs):
        pair_id = f"pair{i:06d}__{a.id}__{b.id}"
        params = sample_speaker_aug_params(
            seed=int(seed),
            key=pair_id,
            pitch_range=tuple(spec.pitch_range),
            rate_range=tuple(spec.rate_range),
        )
        pair_rows.append(
            {
                "pair_id": pair_id,
                "speaker": str(a.speaker),
                "utt1_id": a.id,
                "utt2_id": b.id,
                "utt1_audio_path": str(a.audio_path),
                "utt2_audio_path": str(b.audio_path),
                "utt1_text": a.text,
                "utt2_text": b.text,
                "pitch_semitones": float(params.pitch_semitones),
                "rate": float(params.rate),
            }
        )

    write_jsonl(out_root / "pairs.jsonl", pair_rows)

    out_dirs: Dict[str, Path] = {}
    for v in variants:
        v = str(v).upper().strip()
        name = f"spkproxy_{v}"
        ds_dir = out_root / name
        audio_dir = ds_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        refs_rows: List[dict] = []
        n_written = 0

        for row in pair_rows:
            x1, sr1 = read_wav(Path(row["utt1_audio_path"]))
            x2, sr2 = read_wav(Path(row["utt2_audio_path"]))
            x1, sr1 = ensure_sr(x1, sr1, target_sr=int(spec.sr))
            x2, sr2 = ensure_sr(x2, sr2, target_sr=int(spec.sr))
            assert sr1 == sr2 == int(spec.sr)

            params = sample_speaker_aug_params(
                seed=int(seed),
                key=str(row["pair_id"]),
                pitch_range=tuple(spec.pitch_range),
                rate_range=tuple(spec.rate_range),
            )
            x1a = apply_speaker_aug(x1, sr=int(spec.sr), params=params, keep_len=bool(spec.keep_len))
            x2a = apply_speaker_aug(x2, sr=int(spec.sr), params=params, keep_len=bool(spec.keep_len))

            if v == "A":
                y = np.concatenate([x1, x2]).astype(np.float32)
            elif v == "B":
                y = np.concatenate([x1, x2a]).astype(np.float32)
            elif v == "C":
                y = np.concatenate([x1a, x2]).astype(np.float32)
            elif v == "D":
                y = np.concatenate([x1a, x2a]).astype(np.float32)
            else:
                raise ValueError(f"Unknown spkproxy variant: {v}")

            if float(duration_sec(y, int(spec.sr))) > float(spec.max_total_sec):
                continue

            ex_id = f"{row['pair_id']}__{name}"
            wav_path = audio_dir / f"{ex_id}.wav"
            write_wav(wav_path, y, sr=int(spec.sr))

            ref = f"{row['utt1_text']} {row['utt2_text']}"
            refs_rows.append(
                {
                    "id": ex_id,
                    "audio_path": f"audio/{wav_path.name}",
                    "text": ref,
                    "duration_sec": float(duration_sec(y, int(spec.sr))),
                    "speaker": str(row["speaker"]),
                    "source_id": f"{row['utt1_id']}+{row['utt2_id']}",
                    "condition": name,
                    "pair_id": row["pair_id"],
                    "variant": v,
                }
            )
            n_written += 1

        write_jsonl(ds_dir / "refs.jsonl", refs_rows)
        meta = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kind": "spkproxy",
            "variant": v,
            "seed": int(seed),
            "n_pairs_requested": int(spec.n_pairs),
            "n_written": int(n_written),
            "sr": int(spec.sr),
            "max_total_sec": float(spec.max_total_sec),
            "pitch_range": list(spec.pitch_range),
            "rate_range": list(spec.rate_range),
            "keep_len": bool(spec.keep_len),
        }
        (ds_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        out_dirs[name] = ds_dir

    return out_dirs
