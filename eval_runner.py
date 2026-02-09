from __future__ import annotations

"""
Evaluation runner for ISO experiments.

This module:
1) Builds derived datasets for each condition from cached LibriSpeech splits
   following the final Methodology chapter (baseline, AWGN noise, tails, concatenations).
2) Runs ASR inference for Whisper and Conformer+CTC in fixed decode configurations.
3) Computes WER/CER/InsRate for all conditions and Tail/GAP Non-Empty Rate for
   conditions that introduce explicit non-speech regions.

The code is intentionally explicit and file-backed (writes wavs and jsonl manifests)
so that runs are reproducible, inspectable, and debuggable.
"""

import json
import time
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import re  # add near your other imports at top (once)


import numpy as np

# from .audio_transforms import TailSpec, add_awgn_snr, append_tail, duration_sec, ensure_sr, read_wav, write_wav
from .audio_transforms import (
    TailSpec,
    add_awgn_snr,
    append_tail,
    apply_volume_ramp,
    duration_sec,
    ensure_sr,
    read_wav,
    write_wav,
    add_awgn_snr_gradual,
)



from .jsonl_dataset import JsonlExample, load_dataset_dir, write_jsonl
from .librispeech_cache import cache_librispeech
from .scoring import AggregateScores, aggregate_scores, load_nonspeech_refs, nonempty_rate
# from .seq_multispeaker import ConcatSpec, build_concat_dataset
from .seq_multispeaker import ConcatSpec, build_concat_dataset, SpeakerProxySpec, build_speaker_proxy_datasets


# -----------------------------
# Condition specs (mirrors methodology)
# -----------------------------

# @dataclass(frozen=True)
# class SingleUttCondition:
#     name: str
#     # kind: baseline | noise | tail
#     kind: str
#     snr_db: Optional[float] = None  # for noise (full utterance)
#     tail: Optional[TailSpec] = None  # for appended tail
#     max_total_sec: float = 30.0
#     sr: int = 16000

@dataclass(frozen=True)
class SingleUttCondition:
    name: str
    # kind: baseline | noise | tail | volume
    kind: str

    # noise (full utterance)
    snr_db: Optional[float] = None
    noise_mode: str = "full"   # "full" | "gradual"

    # tails
    tail: Optional[TailSpec] = None

    # volume ramp
    volume_end_gain: Optional[float] = None
    volume_floor_db: float = -80.0
    volume_curve_power_nonzero: float = 0.6

    max_total_sec: float = 30.0
    sr: int = 16000




def _now_tag() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S", time.localtime())


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


# def _load_core_subset(split_dir: Path, max_dur_sec: float = 10.0, limit: int = 0) -> List[JsonlExample]:
#     exs = load_dataset_dir(split_dir)
#     out = [e for e in exs if (e.duration_sec is not None and float(e.duration_sec) <= float(max_dur_sec))]
#     if limit and limit > 0:
#         out = out[: int(limit)]
#     return out


# def _load_core_subset(split_dir: Path, max_dur_sec: float = 10.0, limit: int = 0) -> List[JsonlExample]:
#     exs = load_dataset_dir(split_dir)
#     out = [e for e in exs if (e.duration_sec is not None and float(e.duration_sec) <= float(max_dur_sec))]

#     # IMPORTANT: deterministic ordering so --limit-core selects the same utterances every run
#     out.sort(
#         key=lambda e: (
#             str(getattr(e, "speaker", "")),
#             str(getattr(e, "id", "")),
#             str(getattr(e, "audio_path", "")),
#         )
#     )

#     if limit and limit > 0:
#         out = out[: int(limit)]
#     return out

def _load_core_subset(split_dir: Path, max_dur_sec: float = 10.0, limit: int = 0) -> List[JsonlExample]:
    exs = load_dataset_dir(split_dir)
    out = [e for e in exs if (e.duration_sec is not None and float(e.duration_sec) <= float(max_dur_sec))]

    # ✅ critical: enforce stable ordering so --limit-core selects the same items every run
    out.sort(key=lambda e: str(e.id))

    if limit and limit > 0:
        out = out[: int(limit)]
    return out



def build_single_utt_dataset(
    base_split_dir: Path,
    out_dir: Path,
    cond: SingleUttCondition,
    seed: int,
    max_dur_sec: float = 10.0,
    limit: int = 0,
) -> Path:
    """
    Build a derived dataset for baseline/noise/tails from a cached LibriSpeech split.

    Writes:
      out_dir/refs.jsonl
      out_dir/audio/*.wav
      out_dir/nonspeech_refs.jsonl (for tails only)
      out_dir/meta.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    core = _load_core_subset(base_split_dir, max_dur_sec=max_dur_sec, limit=limit)

    refs_rows: List[dict] = []
    nonspeech_rows: List[dict] = []
    n_written = 0

    for ex in core:
        x, sr = read_wav(ex.audio_path)
        x, sr = ensure_sr(x, sr, target_sr=cond.sr)
        key = f"{cond.name}__{ex.id}"

        if cond.kind == "baseline":
            y = x
            y_ns = None

        # elif cond.kind == "noise":
        #     assert cond.snr_db is not None
        #     y = add_awgn_snr(x, sr=sr, snr_db=float(cond.snr_db), seed=int(seed), key=key)
        #     y_ns = None

        # elif cond.kind == "tail":
        #     assert cond.tail is not None
        #     y, tail = append_tail(x, sr=sr, spec=cond.tail, seed=int(seed), key=key)
        #     y_ns = tail
        # else:
        #     raise ValueError(f"Unknown condition kind: {cond.kind}")



        elif cond.kind == "noise":
            assert cond.snr_db is not None
            if cond.noise_mode == "full":
                y = add_awgn_snr(x, sr=sr, snr_db=float(cond.snr_db), seed=int(seed), key=key)
            elif cond.noise_mode == "gradual":
                y = add_awgn_snr_gradual(x, sr=sr, snr_db_end=float(cond.snr_db), seed=int(seed), key=key)
            else:
                raise ValueError(f"Unknown noise_mode: {cond.noise_mode}")
            y_ns = None

        # elif cond.kind == "volume":
        #     assert cond.volume_end_gain is not None
        #     y = apply_volume_ramp(x, end_gain=float(cond.volume_end_gain))
        #     y_ns = None

        elif cond.kind == "tail":
            assert cond.tail is not None
            y, tail = append_tail(x, sr=sr, spec=cond.tail, seed=int(seed), key=key)
            y_ns = tail

        elif cond.kind == "volume":
            assert cond.volume_end_gain is not None
            y = apply_volume_ramp(
                x,
                end_gain=float(cond.volume_end_gain),
                floor_db=float(cond.volume_floor_db),
                curve_power_nonzero=float(cond.volume_curve_power_nonzero),
            )
            y_ns = None

        else:
            raise ValueError(f"Unknown condition kind: {cond.kind}")




        if duration_sec(y, sr) > float(cond.max_total_sec):
            continue

        utt_id = f"{ex.id}__{cond.name}"
        wav_path = audio_dir / f"{utt_id}.wav"

        if cond.kind == "baseline":
            # link/copy original file to keep dataset self-contained
            _safe_link_or_copy(ex.audio_path, wav_path)
        else:
            write_wav(wav_path, y, sr=sr)

        refs_rows.append(
            {
                "id": utt_id,
                "audio_path": f"audio/{wav_path.name}",
                "text": ex.text,
                "duration_sec": float(duration_sec(y, sr)),
                "speaker": ex.speaker,
                "source_id": ex.id,
                "source_audio_path": str(ex.audio_path),
                "condition": cond.name,
            }
        )

        # tail probe in isolation (last D seconds by construction)
        if y_ns is not None and cond.tail is not None:
            ns_path = audio_dir / f"{utt_id}__tail.wav"
            write_wav(ns_path, y_ns, sr=sr)
            nonspeech_rows.append(
                {
                    "id": utt_id,
                    "audio_path": f"audio/{ns_path.name}",
                    "kind": "tail",
                    "duration_sec": float(cond.tail.duration_sec),
                }
            )

        n_written += 1

    write_jsonl(out_dir / "refs.jsonl", refs_rows)
    if nonspeech_rows:
        write_jsonl(out_dir / "nonspeech_refs.jsonl", nonspeech_rows)

    meta = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_split_dir": str(base_split_dir),
        "condition": cond.name,
        "kind": cond.kind,
        "snr_db": cond.snr_db,
        "noise_mode": getattr(cond, "noise_mode", None),
        "volume_end_gain": cond.volume_end_gain,
        "volume_floor_db": float(cond.volume_floor_db),
        "volume_curve_power_nonzero": float(cond.volume_curve_power_nonzero),
        "tail": None if cond.tail is None else cond.tail.__dict__,
        "seed": int(seed),
        "max_dur_sec_core_subset": float(max_dur_sec),
        "limit": int(limit),
        "n_written": int(n_written),
        "sr": int(cond.sr),
        "max_total_sec": float(cond.max_total_sec),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


# -----------------------------
# ASR runners (thin wrappers)
# -----------------------------

@dataclass(frozen=True)
class WhisperDecode:
    name: str
    beam_size: int
    temperature: float
    condition_on_previous_text: bool = False


@dataclass(frozen=True)
class ConformerDecode:
    name: str
    mode: str  # "greedy" | "beam"
    beam_size: int = 32
    token_prune_topk: int = 20

    lm_path: Optional[Path] = None
    beam_alpha: float = 0.0
    beam_beta: float = 0.0

    # NEW: unigrams file for pyctcdecode when using KenLM binaries
    lm_unigrams_path: Optional[Path] = None




def _build_whisper_asr(model_name: str, device: str, compute_type: str, decode: WhisperDecode, cache_dir: Path):
    try:
        from asr_mvp.models.whisper_asr import WhisperASR, WhisperConfig  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import WhisperASR wrapper. Ensure your package is installed and Whisper deps are available.\n"
            f"Original error: {e}"
        )
    cfg = WhisperConfig(
        name=model_name,
        device=device,
        compute_type=compute_type,
        language="en",
        task="transcribe",
        beam_size=int(decode.beam_size),
        temperature=float(decode.temperature),
        condition_on_previous_text=bool(decode.condition_on_previous_text),
    )
    return WhisperASR(cfg, cache_dir=cache_dir)


def _build_conformer_runner(model_name: str, device: str, cache_dir: Path, decode: ConformerDecode):
    from .conformer_ctc_runner import ConformerCTCRunner, ConformerRunnerConfig

    cfg = ConformerRunnerConfig(
        pretrained_name=model_name,
        cache_dir=cache_dir,
        device=device,
        decode_mode=decode.mode,
        beam_size=int(decode.beam_size),
        token_prune_topk=int(decode.token_prune_topk),
        lm_path=decode.lm_path,
        beam_alpha=float(decode.beam_alpha),
        beam_beta=float(decode.beam_beta),
        lm_unigrams_path=decode.lm_unigrams_path,
    )
    return ConformerCTCRunner(cfg)



@dataclass(frozen=True)
class _AudioItem:
    audio_path: Path


def _transcribe_in_batches(asr, items: List[Any], batch_size: int = 16) -> List[str]:
    hyps: List[str] = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        out = asr.transcribe(batch)  # expected List[str]
        hyps.extend([str(s) for s in out])
    return hyps



# -----------------------------
# Evaluation loop
# -----------------------------

@dataclass(frozen=True)
class EvalConfig:
    seed: int = 70072
    max_core_dur_sec: float = 10.0
    whisper_max_total_sec: float = 30.0
    concat_n_pairs: int = 1000
    concat_gap_secs: Tuple[float, float] = (0.0, 5.0)

    # Default single-model names (kept for backwards compatibility).
    whisper_model: str = "small.en"
    conformer_model: str = "stt_en_conformer_ctc_large"

    device: str = "cuda"
    whisper_compute_type: str = "float16"

    # batching
    batch_size: int = 16
    limit_core: int = 0  # 0 => full core subset

    # fnmatch patterns; run only matching conditions (e.g. "baseline", "tail_*", "concat__diff__silence__5s")
    only_conditions: Optional[List[str]] = None

    # --- NEW: experiment selection (all optional; defaults mimic old behaviour) ---
    systems: Tuple[str, ...] = ("whisper", "conformer")

    # Whisper selection
    whisper_models: Optional[List[str]] = None          # e.g. ["small.en", "large-v3"]
    whisper_decode_names: Optional[List[str]] = None    # e.g. ["deterministic", "sampling_t0p8"]

    # Conformer selection
    conformer_models: Optional[List[str]] = None        # e.g. ["stt_en_conformer_ctc_large"]
    conformer_decode_names: Optional[List[str]] = None  # e.g. ["greedy", "beam", "beam_lm"]

    # Conformer decode params
    conformer_beam_size: int = 32
    conformer_token_prune_topk: int = 20

    # Conformer external LM (KenLM) shallow fusion params (used by decode name "beam_lm")
    conformer_lm_path: Optional[Path] = None
    conformer_beam_alpha: float = 0.0
    conformer_beam_beta: float = 0.0

    # NEW
    conformer_lm_unigrams_path: Optional[Path] = None



def _resolve_audio_path(ds_dir: Path, p: Path) -> Path:
    if p.is_absolute():
        return p

    # If load_dataset_dir already returned a path under ds_dir (but still relative),
    # don't prepend ds_dir again.
    if p.is_relative_to(ds_dir):
        return p

    cand = ds_dir / p
    if cand.exists():
        return cand

    # fallback: maybe p is relative to CWD
    if p.exists():
        return p

    return cand


def run_full_evaluation(
    split_name: str,
    cache_root: Path,
    out_dir: Path,
    cfg: EvalConfig,
) -> Path:
    """
    Main entry used by experiments/run.py.

    - split_name: "dev-clean" or "test-clean" (methodology: test-clean for final)
    - cache_root: where cached LibriSpeech and model caches live
    - out_dir: where results and derived datasets are written
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional condition filtering (run only a subset of conditions).
    from fnmatch import fnmatch
    patterns = [p.strip() for p in (getattr(cfg, 'only_conditions', None) or []) if p and p.strip()]

    def want(name: str) -> bool:
        if not patterns:
            return True
        return any(fnmatch(name, pat) for pat in patterns)

    ds_cache_root = cache_root / "datasets"
    ds_cache_root.mkdir(parents=True, exist_ok=True)

    # 1) Ensure base LibriSpeech split exists
    base_split_dir = cache_root / "librispeech" / split_name
    if not base_split_dir.exists():
        cache_librispeech(cache_root, split_name, limit=0, seed=int(cfg.seed))

    # 2) Build derived datasets for conditions (1)-(5)
    derived_root = out_dir / "derived_datasets" / split_name
    derived_root.mkdir(parents=True, exist_ok=True)

    conds: List[SingleUttCondition] = [
        SingleUttCondition(name="baseline", kind="baseline", max_total_sec=float(cfg.whisper_max_total_sec)),
        SingleUttCondition(name="noise_snr20", kind="noise", snr_db=20.0, max_total_sec=float(cfg.whisper_max_total_sec)),
        SingleUttCondition(name="noise_snr5", kind="noise", snr_db=5.0, max_total_sec=float(cfg.whisper_max_total_sec)),


        # ✅ harsher noise (worse SNR)
        SingleUttCondition(name="noise_snr0", kind="noise", snr_db=0.0, max_total_sec=float(cfg.whisper_max_total_sec)),
        SingleUttCondition(name="noise_snr-5", kind="noise", snr_db=-5.0, max_total_sec=float(cfg.whisper_max_total_sec)),


        # ✅ NEW: gradual noise ramp (0 noise at start -> target SNR noise at end)
        SingleUttCondition(
            name="gradual_noise_snr20",
            kind="noise",
            snr_db=20.0,
            noise_mode="gradual",
            max_total_sec=float(cfg.whisper_max_total_sec),
        ),
        SingleUttCondition(
            name="gradual_noise_snr5",
            kind="noise",
            snr_db=5.0,
            noise_mode="gradual",
            max_total_sec=float(cfg.whisper_max_total_sec),
        ),
        SingleUttCondition(
            name="gradual_noise_snr0",
            kind="noise",
            snr_db=0.0,
            noise_mode="gradual",
            max_total_sec=float(cfg.whisper_max_total_sec),
        ),
        SingleUttCondition(
            name="gradual_noise_snr-5",
            kind="noise",
            snr_db=-5.0,
            noise_mode="gradual",
            max_total_sec=float(cfg.whisper_max_total_sec),
        ),



        # ✅ NEW: gradual volume ramp (full volume at start -> target gain at end)
        SingleUttCondition(
            name="gradual_volume_end0",
            kind="volume",
            volume_end_gain=0.0,  # extreme: ends silent
            max_total_sec=float(cfg.whisper_max_total_sec),
        ),
        SingleUttCondition(
            name="gradual_volume_end0.0001",
            kind="volume",
            volume_end_gain=0.0001,
            max_total_sec=float(cfg.whisper_max_total_sec),
        ),
        SingleUttCondition(
            name="gradual_volume_end0.0005",
            kind="volume",
            volume_end_gain=0.0005,
            max_total_sec=float(cfg.whisper_max_total_sec),
        ),
        SingleUttCondition(
            name="gradual_volume_end0.001",
            kind="volume",
            volume_end_gain=0.001,
            max_total_sec=float(cfg.whisper_max_total_sec),
        ),
        SingleUttCondition(
            name="gradual_volume_end0.01",
            kind="volume",
            volume_end_gain=0.01,
            max_total_sec=float(cfg.whisper_max_total_sec),
        ),

            



        # tails
        SingleUttCondition(
            name="tail_silence_3s",
            kind="tail",
            tail=TailSpec(kind="silence", duration_sec=3.0, noise_snr_db=None, fade_ms=10.0),
            max_total_sec=float(cfg.whisper_max_total_sec),
        ),
        SingleUttCondition(
            name="tail_silence_10s",
            kind="tail",
            tail=TailSpec(kind="silence", duration_sec=10.0, noise_snr_db=None, fade_ms=10.0),
            max_total_sec=float(cfg.whisper_max_total_sec),
        ),
        SingleUttCondition(
            name="tail_noise10db_3s",
            kind="tail",
            tail=TailSpec(kind="noise", duration_sec=3.0, noise_snr_db=10.0, fade_ms=10.0),
            max_total_sec=float(cfg.whisper_max_total_sec),
        ),
        SingleUttCondition(
            name="tail_noise10db_10s",
            kind="tail",
            tail=TailSpec(kind="noise", duration_sec=10.0, noise_snr_db=10.0, fade_ms=10.0),
            max_total_sec=float(cfg.whisper_max_total_sec),
        ),
    ]

    condition_dirs: Dict[str, Path] = {}
    for c in conds:
        if not want(c.name):
            continue
        c_dir = derived_root / c.name
        build_single_utt_dataset(
            base_split_dir=base_split_dir,
            out_dir=c_dir,
            cond=c,
            seed=int(cfg.seed),
            max_dur_sec=float(cfg.max_core_dur_sec),
            limit=int(cfg.limit_core),
        )
        condition_dirs[c.name] = c_dir

    # 3) Build Condition (6) concatenation datasets
    core = _load_core_subset(base_split_dir, max_dur_sec=float(cfg.max_core_dur_sec), limit=int(cfg.limit_core))

    concat_specs: List[ConcatSpec] = []
    for same in (True, False):
        for gap_kind in ("silence", "noise"):
            for gsec in cfg.concat_gap_secs:
                concat_specs.append(
                    ConcatSpec(
                        same_speaker=bool(same),
                        gap_kind=str(gap_kind),
                        gap_sec=float(gsec),
                        noise_snr_db=10.0,
                        n_pairs=int(cfg.concat_n_pairs),
                        max_total_sec=float(cfg.whisper_max_total_sec),
                        sr=16000,
                    )
                )

    for cs in concat_specs:
        name = f"concat__{'same' if cs.same_speaker else 'diff'}__{cs.gap_kind}__{int(cs.gap_sec)}s"
        if not want(name):
            continue
        c_dir = derived_root / name
        build_concat_dataset(core, out_dir=c_dir, spec=cs, seed=int(cfg.seed))
        condition_dirs[name] = c_dir



    # 3b) Speaker-change proxy datasets (matched same-speaker pairs, 0s gap): A/B/C/D
    spkproxy_names = [
        "spkproxy__A__clean_clean",
        "spkproxy__B__clean_aug",
        "spkproxy__C__aug_clean",
        "spkproxy__D__aug_aug",
    ]

    # Only build if requested (or if no filtering is active)
    if (not patterns) or any(want(n) for n in spkproxy_names):
        spk_spec = SpeakerProxySpec(
            n_pairs=int(cfg.concat_n_pairs),
            max_total_sec=float(cfg.whisper_max_total_sec),
            sr=16000,
            pitch_max_semitones=3.0,
            rate_min=0.9,
            rate_max=1.1,
        )
        built = build_speaker_proxy_datasets(core, out_root=derived_root, spec=spk_spec, seed=int(cfg.seed))
        for cname, cdir in built.items():
            if want(cname):
                condition_dirs[cname] = cdir




    # --- FAIL FAST if user-selected patterns matched nothing ---
    if not condition_dirs:
        available_single = [c.name for c in conds]
        available_concat = [
            f"concat__{'same' if cs.same_speaker else 'diff'}__{cs.gap_kind}__{int(cs.gap_sec)}s"
            for cs in concat_specs
        ]
        available_spkproxy = spkproxy_names

        raise SystemExit(
            "No conditions matched --only-condition patterns.\n"
            f"Patterns: {patterns}\n"
            f"Available single-utt: {available_single}\n"
            f"Available concat: {available_concat}\n"
            f"Available speaker-proxy: {available_spkproxy}"
        )


            
    whisper_cache = cache_root / "whisper"
    conformer_cache = cache_root / "nemo"

    # Available decode presets (filterable via cfg.*_decode_names)
    whisper_decodes_all = [
        WhisperDecode(name="deterministic", beam_size=5, temperature=0.0, condition_on_previous_text=False),
        WhisperDecode(name="sampling_t0p8", beam_size=1, temperature=0.8, condition_on_previous_text=False),
    ]

    conformer_decodes_all: List[ConformerDecode] = [
        ConformerDecode(name="greedy", mode="greedy"),
        ConformerDecode(
            name="beam",
            mode="beam",
            beam_size=int(cfg.conformer_beam_size),
            token_prune_topk=int(cfg.conformer_token_prune_topk),
        ),
    ]

    if cfg.conformer_lm_path is not None:
        conformer_decodes_all.append(
            ConformerDecode(
                name="beam_lm",
                mode="beam",
                beam_size=int(cfg.conformer_beam_size),
                token_prune_topk=int(cfg.conformer_token_prune_topk),
                lm_path=cfg.conformer_lm_path,
                beam_alpha=float(cfg.conformer_beam_alpha),
                beam_beta=float(cfg.conformer_beam_beta),
                lm_unigrams_path=cfg.conformer_lm_unigrams_path,
            )
        )


    # Apply optional decode filtering
    whisper_decodes = (
        [d for d in whisper_decodes_all if d.name in set(cfg.whisper_decode_names)]
        if cfg.whisper_decode_names
        else whisper_decodes_all
    )
    conformer_decodes = (
        [d for d in conformer_decodes_all if d.name in set(cfg.conformer_decode_names)]
        if cfg.conformer_decode_names
        else conformer_decodes_all
    )

    # Apply optional model filtering
    whisper_models = cfg.whisper_models or [cfg.whisper_model]
    conformer_models = cfg.conformer_models or [cfg.conformer_model]

    systems = tuple(s.lower().strip() for s in (cfg.systems or ("whisper", "conformer")))
    systems_set = set(systems)

    def _safe_tag(s: str) -> str:
        return s.replace("/", "_").replace(" ", "").replace(":", "_")

    # Build a flat job list so we can run job-by-job (only one model on GPU at a time)
    jobs: List[Dict[str, Any]] = []
    if "whisper" in systems_set:
        for mname in whisper_models:
            for d in whisper_decodes:
                jobs.append({"system": "whisper", "model": str(mname), "decode": d})
    if "conformer" in systems_set:
        for mname in conformer_models:
            for d in conformer_decodes:
                jobs.append({"system": "conformer", "model": str(mname), "decode": d})

    # 5) Evaluate all selected conditions for each job (model loaded lazily per job)
    results_root = out_dir / "results" / split_name
    results_root.mkdir(parents=True, exist_ok=True)

    def _release_model(obj: Any) -> None:
        # Best-effort GPU memory release between jobs
        try:
            del obj
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    for job in jobs:
        sys_name = str(job["system"])
        model_name = str(job["model"])
        decode = job["decode"]

        # Load *one* runner for this job (kept alive across all conditions), then release it.
        if sys_name == "whisper":
            asr = _build_whisper_asr(model_name, cfg.device, cfg.whisper_compute_type, decode, whisper_cache)
        else:
            asr = _build_conformer_runner(model_name, cfg.device, conformer_cache, decode)

        job_tag = f"{sys_name}__{_safe_tag(model_name)}__{decode.name}"

        for cond_name, ds_dir in condition_dirs.items():
            exs = load_dataset_dir(ds_dir)
            audio_paths = [e.audio_path for e in exs]
            refs = [e.text for e in exs]
            ids = [e.id for e in exs]

            audio_paths_abs = [_resolve_audio_path(ds_dir, p) for p in audio_paths]

            whisper_items = [_AudioItem(p) for p in audio_paths_abs]
            ns_rows = load_nonspeech_refs(ds_dir)
            ns_paths_abs = [_resolve_audio_path(ds_dir, Path(r["audio_path"])) for r in ns_rows]
            ns_whisper_items = [_AudioItem(p) for p in ns_paths_abs] if ns_paths_abs else []

            out_sub = results_root / cond_name / job_tag
            out_sub.mkdir(parents=True, exist_ok=True)

            inputs = whisper_items if sys_name == "whisper" else audio_paths_abs
            hyps = _transcribe_in_batches(asr, inputs, batch_size=int(cfg.batch_size))

            assert len(hyps) == len(refs)

            def _norm_hyp(s: str) -> str:
                s = str(s).replace("▁", " ")
                s = re.sub(r"\s+", " ", s).strip()
                return s

            hyps_norm = [_norm_hyp(h) for h in hyps]

            agg = aggregate_scores(refs, hyps_norm)

            # Tail/GAP Non-Empty Rate where applicable (ds contains nonspeech_refs.jsonl)
            ns_rate = None
            if ns_whisper_items:
                ns_inputs = ns_whisper_items if sys_name == "whisper" else ns_paths_abs
                ns_hyps = _transcribe_in_batches(asr, ns_inputs, batch_size=int(cfg.batch_size))
                ns_rate = float(nonempty_rate(ns_hyps))
                _write_preds(
                    out_sub / "nonspeech_preds.jsonl",
                    [r["id"] for r in ns_rows],
                    [""] * len(ns_hyps),
                    ns_hyps,
                )

            _write_preds(out_sub / "preds.jsonl", ids, refs, hyps_norm)

            summary = {
                "split": split_name,
                "condition": cond_name,
                "system": sys_name,
                "model": model_name,
                "decode": decode.name,
                # "decode_params": decode.__dict__,
                "decode_params": _jsonify(decode),
                "wer": agg.wer,
                "cer": agg.cer,
                "ins_rate": agg.ins_rate,
                "n_examples": agg.n_examples,
                "total_ref_words": agg.total_ref_words,
                "nonempty_rate": ns_rate,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            (out_sub / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        _release_model(asr)

    # write run manifest
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(cfg.seed),
        "split": split_name,
        "max_core_dur_sec": float(cfg.max_core_dur_sec),
        "whisper_window_max_sec": float(cfg.whisper_max_total_sec),
        "concat_n_pairs": int(cfg.concat_n_pairs),
        "concat_gap_secs": list(cfg.concat_gap_secs),
        "only_conditions": patterns,
        "systems": list(systems),
        # "jobs": [{"system": j["system"], "model": j["model"], "decode": j["decode"].__dict__} for j in jobs],
        "jobs": [{"system": j["system"], "model": j["model"], "decode": _jsonify(j["decode"])} for j in jobs],
        "whisper": {
            "models": whisper_models,
            "compute_type": cfg.whisper_compute_type,
            # "decodes": [d.__dict__ for d in whisper_decodes],
            "decodes": [_jsonify(d) for d in whisper_decodes],
        },
        "conformer": {
            "models": conformer_models,
            # "decodes": [d.__dict__ for d in conformer_decodes],
            "decodes": [_jsonify(d) for d in conformer_decodes],
            "external_lm": {
                "lm_path": (str(cfg.conformer_lm_path) if cfg.conformer_lm_path is not None else None),
                "beam_alpha": float(cfg.conformer_beam_alpha),
                "beam_beta": float(cfg.conformer_beam_beta),
            },
        },
        "derived_root": str(derived_root),
        "results_root": str(results_root),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def _jsonify(x: Any) -> Any:
    """Convert common non-JSON-serializable objects (e.g., Path, dataclasses) into JSON-safe types."""
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


def _write_preds(path: Path, ids: List[str], refs: List[str], hyps: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, (rid, r, h) in enumerate(zip(ids, refs, hyps)):
            row = {"id": rid, "ref": r, "hyp": h}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
