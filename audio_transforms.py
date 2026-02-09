from __future__ import annotations

"""
Waveform utilities and deterministic stressor transforms used by experiments.

Design goals:
- Deterministic given (global_seed, example_id) so that re-runs reproduce exactly.
- Pure waveform-level transforms (noise, appends, concatenation gaps), then each model
  applies its own standard feature extraction (Whisper / NeMo), matching methodology.
- Minimal dependencies: numpy + soundfile (and optional librosa for resampling if needed).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# -----------------------------
# Audio I/O
# -----------------------------

def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Read a mono wav file into float32 in [-1, 1]."""
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        # downmix to mono
        x = np.mean(x, axis=1, dtype=np.float32)
    return x.astype(np.float32), int(sr)


def write_wav(path: Path, x: np.ndarray, sr: int = 16000) -> None:
    """Write a mono wav file (float32)."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.astype(np.float32), int(sr))



def ensure_sr(x: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Resample to target_sr if necessary (best-effort)."""
    if sr == target_sr:
        return x, sr
    try:
        import librosa  # optional
        y = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=target_sr).astype(np.float32)
        return y, target_sr
    except Exception:
        # fallback: naive index mapping (keeps things running, but not ideal quality)
        ratio = target_sr / float(sr)
        idx = (np.arange(int(len(x) * ratio)) / ratio).astype(np.int64)
        idx = np.clip(idx, 0, len(x) - 1)
        y = x[idx].astype(np.float32)
        return y, target_sr


# -----------------------------
# DSP helpers
# -----------------------------

def rms(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps))


def db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def linear_to_db(x: float, eps: float = 1e-12) -> float:
    return float(20.0 * np.log10(max(x, eps)))


def peak_normalize_if_needed(x: np.ndarray, peak: float = 0.999) -> np.ndarray:
    """Only scale down if clipping would occur."""
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m <= 1.0:
        return x
    return (x / m * peak).astype(np.float32)


def apply_fade_out(x: np.ndarray, sr: int = 16000, fade_ms: float = 10.0) -> np.ndarray:
    """Linear fade-out on the last fade_ms milliseconds."""
    n = int(round(sr * (fade_ms / 1000.0)))
    if n <= 0 or n >= len(x):
        return x
    w = np.linspace(1.0, 0.0, n, dtype=np.float32)
    y = x.copy()
    y[-n:] *= w
    return y


# -----------------------------
# Deterministic noise
# -----------------------------

def _rng_for(seed: int, key: str) -> np.random.Generator:
    # stable int from key
    h = 2166136261
    for ch in key.encode("utf-8"):
        h = (h ^ ch) * 16777619
        h &= 0xFFFFFFFF
    return np.random.default_rng(int((seed + h) % (2**32 - 1)))


def awgn_like(x: np.ndarray, seed: int, key: str) -> np.ndarray:
    """Generate zero-mean Gaussian noise of same length as x (deterministic)."""
    rng = _rng_for(seed, key)
    n = rng.standard_normal(size=len(x), dtype=np.float32)
    n -= float(np.mean(n, dtype=np.float64))
    return n.astype(np.float32)


def scale_noise_to_snr(noise: np.ndarray, signal_rms: float, snr_db: float, eps: float = 1e-12) -> np.ndarray:
    """
    Scale noise to achieve target SNR (dB) relative to signal RMS:
      SNR_dB = 20 log10( RMS(signal) / RMS(noise) )
    """
    target_noise_rms = signal_rms / db_to_linear(snr_db)
    n_rms = rms(noise, eps=eps)
    if n_rms <= 0:
        return noise
    return (noise * (target_noise_rms / n_rms)).astype(np.float32)


def add_awgn_snr(x: np.ndarray, sr: int, snr_db: float, seed: int, key: str) -> np.ndarray:
    """Add AWGN at target SNR relative to x RMS. Deterministic given seed+key."""
    sig_rms = rms(x)
    n = awgn_like(x, seed=seed, key=key)
    n = scale_noise_to_snr(n, signal_rms=sig_rms, snr_db=snr_db)
    y = x + n
    return peak_normalize_if_needed(y)


def add_awgn_snr_gradual(
    x: np.ndarray,
    sr: int,
    snr_db_end: float,
    seed: int,
    key: str,
) -> np.ndarray:
    """
    Add AWGN where noise amplitude increases linearly from 0 -> full by the end.

    Interpretation:
    - Generate AWGN scaled so that if applied fully, it corresponds to `snr_db_end`
      relative to the utterance RMS.
    - Apply a linear ramp envelope e(t) from 0 to 1 over the utterance.
    - Output: y[t] = x[t] + e(t)*n_scaled[t]

    This yields high acoustic evidence at the start and increasingly noisy audio
    toward the end, reaching the intended noise level by the end.
    Deterministic given seed+key.
    """
    sig_rms = rms(x)
    n = awgn_like(x, seed=seed, key=key)
    n = scale_noise_to_snr(n, signal_rms=sig_rms, snr_db=float(snr_db_end))

    if len(x) <= 1:
        return peak_normalize_if_needed(x + n)

    env = np.linspace(0.0, 1.0, num=len(x), dtype=np.float32)
    y = x + (n * env).astype(np.float32)
    return peak_normalize_if_needed(y)


def apply_volume_ramp(
    x: np.ndarray,
    end_gain: float,
    floor_db: float = -80.0,
    curve_power_nonzero: float = 0.6,  # < 1.0 => stronger earlier attenuation
) -> np.ndarray:
    """
    Perceptual (dB-linear) volume ramp with an optional curve for non-zero end_gain.

    Goal:
      - end_gain == 0.0: keep the SAME behavior you currently like
      - end_gain > 0.0: make the fade stronger earlier (so last-second speech actually attenuates)

    curve_power_nonzero:
      - 1.0 => your original linear-in-dB behavior
      - 0.8 => slightly stronger earlier
      - 0.6 => noticeably stronger earlier (recommended)
      - 0.5 => stronger still
    """
    g2 = float(end_gain)
    if g2 < 0.0:
        g2 = 0.0

    n = len(x)
    if n <= 1:
        return (x * np.float32(g2)).astype(np.float32)

    # time base
    t = np.linspace(0.0, 1.0, num=n, dtype=np.float32)

    if g2 <= 0.0:
        # KEEP EXACT SAME "0" BEHAVIOR SHAPE: ramp to floor_db then force final sample to 0
        end_db = float(floor_db)
        db_env = (end_db * t).astype(np.float32)
        env = (10.0 ** (db_env / 20.0)).astype(np.float32)
        y = (x.astype(np.float32) * env).astype(np.float32)
        y[-1] = 0.0
        return y

    # Non-zero: honor the true end_gain (NO CAP), but apply a curve for stronger earlier drop
    end_db = float(20.0 * np.log10(g2))  # no clamping/capping

    if curve_power_nonzero != 1.0:
        # concave curve => t becomes larger earlier => more negative dB earlier
        t = np.power(t, np.float32(curve_power_nonzero)).astype(np.float32)

    db_env = (end_db * t).astype(np.float32)
    env = (10.0 ** (db_env / 20.0)).astype(np.float32)

    y = (x.astype(np.float32) * env).astype(np.float32)
    return y


# -----------------------------
# Speaker-change proxy augmentations (deterministic)
# -----------------------------

@dataclass(frozen=True)
class SpeakerAugParams:
    """Parameters for speaker-like augmentation."""
    pitch_semitones: float
    rate: float  # time-stretch rate (>1 faster, <1 slower)


def make_speaker_aug_params(
    seed: int,
    key: str,
    pitch_max_semitones: float = 3.0,
    rate_min: float = 0.9,
    rate_max: float = 1.1,
) -> SpeakerAugParams:
    """Deterministically sample augmentation params from (seed, key)."""
    rng = _rng_for(int(seed), f"{key}__spkaug")
    pitch = float(rng.uniform(-float(pitch_max_semitones), float(pitch_max_semitones)))
    rmin = float(min(rate_min, rate_max))
    rmax = float(max(rate_min, rate_max))
    rate = float(rng.uniform(rmin, rmax))
    return SpeakerAugParams(pitch_semitones=pitch, rate=rate)


def apply_speaker_aug(x: np.ndarray, sr: int, params: SpeakerAugParams) -> np.ndarray:
    """
    Apply speaker-like augmentation: time-stretch (rate) + pitch shift (semitones).

    Requires librosa. If librosa is not installed, this raises a RuntimeError
    (the rest of the codebase remains usable; only speaker-proxy conditions need it).
    """
    try:
        import librosa  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Speaker-proxy augmentation requires librosa. Install it (pip install librosa) to use spkproxy conditions.\n"
            f"Original error: {e}"
        )

    y = x.astype(np.float32)

    # Time-stretch changes duration (expected). Upstream enforces max_total_sec on the final concatenation.
    if params.rate and abs(float(params.rate) - 1.0) > 1e-6:
        y = librosa.effects.time_stretch(y, rate=float(params.rate)).astype(np.float32)

    # Pitch shift (tries to preserve duration)
    if params.pitch_semitones and abs(float(params.pitch_semitones)) > 1e-6:
        y = librosa.effects.pitch_shift(y, sr=int(sr), n_steps=float(params.pitch_semitones)).astype(np.float32)

    return peak_normalize_if_needed(y)



# -----------------------------
# Speaker-like augmentation (pitch + tempo proxy)
# -----------------------------

@dataclass(frozen=True)
class SpeakerAugParams:
    """Deterministic augmentation parameters for the speaker-change proxy."""
    pitch_semitones: float
    rate: float  # tempo / speaking-rate proxy


def _fix_length(x: np.ndarray, n: int) -> np.ndarray:
    """Pad with zeros or truncate so that len(x) == n."""
    if len(x) == n:
        return x.astype(np.float32)
    if len(x) < n:
        y = np.zeros(n, dtype=np.float32)
        y[: len(x)] = x.astype(np.float32)
        return y
    return x[:n].astype(np.float32)


def sample_speaker_aug_params(
    seed: int,
    key: str,
    pitch_range: Tuple[float, float] = (-4.0, 4.0),
    rate_range: Tuple[float, float] = (0.9, 1.1),
) -> SpeakerAugParams:
    """Sample deterministic augmentation parameters from (seed, key)."""
    rng = _rng_for(int(seed), f"{key}__spkaug")
    p = float(rng.uniform(float(pitch_range[0]), float(pitch_range[1])))
    r = float(rng.uniform(float(rate_range[0]), float(rate_range[1])))
    return SpeakerAugParams(pitch_semitones=p, rate=r)


def apply_speaker_aug(
    x: np.ndarray,
    sr: int,
    params: SpeakerAugParams,
    keep_len: bool = True,
) -> np.ndarray:
    """Apply a simple speaker-like augmentation (tempo + pitch). Deterministic via params.

    - Tempo (rate): changes speaking-rate characteristics.
    - Pitch: shifts fundamental frequency characteristics.
    - keep_len=True: ensures output has the same number of samples as input (recommended
      for matched A/B/C/D comparisons that should not change overall duration).
    """
    y = x.astype(np.float32)

    # Best-effort high-quality implementation via librosa (optional dependency).
    try:
        import librosa  # type: ignore

        # Tempo (phase vocoder) changes length; optionally fix length back to input.
        if float(params.rate) != 1.0:
            y2 = librosa.effects.time_stretch(y, rate=float(params.rate)).astype(np.float32)
        else:
            y2 = y

        if keep_len:
            y2 = librosa.util.fix_length(y2, size=len(y)).astype(np.float32)

        # Pitch shift keeps length.
        if float(params.pitch_semitones) != 0.0:
            y3 = librosa.effects.pitch_shift(y2, sr=int(sr), n_steps=float(params.pitch_semitones)).astype(np.float32)
        else:
            y3 = y2

        if keep_len:
            y3 = librosa.util.fix_length(y3, size=len(y)).astype(np.float32)

        return peak_normalize_if_needed(y3)

    except Exception:
        # Fallback (no librosa): approximate by resampling (note: couples tempo+pitch).
        # 1) Resample to emulate tempo change.
        if float(params.rate) != 1.0:
            ratio = float(params.rate)
            idx = (np.arange(int(len(y) / ratio)) * ratio).astype(np.int64)
            idx = np.clip(idx, 0, len(y) - 1)
            y = y[idx].astype(np.float32)

        # 2) Resample to emulate pitch shift (approx; will also affect tempo).
        if float(params.pitch_semitones) != 0.0:
            pitch_ratio = float(2.0 ** (float(params.pitch_semitones) / 12.0))
            idx = (np.arange(int(len(y) / pitch_ratio)) * pitch_ratio).astype(np.int64)
            idx = np.clip(idx, 0, len(y) - 1)
            y = y[idx].astype(np.float32)

        if keep_len:
            y = _fix_length(y, len(x))

        return peak_normalize_if_needed(y)




def make_silence(duration_sec: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(round(duration_sec * sr)), dtype=np.float32)


def make_noise(duration_sec: float, sr: int, seed: int, key: str, ref_rms: float, snr_db: float) -> np.ndarray:
    n = awgn_like(np.zeros(int(round(duration_sec * sr)), dtype=np.float32), seed=seed, key=key)
    n = scale_noise_to_snr(n, signal_rms=ref_rms, snr_db=snr_db)
    return n.astype(np.float32)


# -----------------------------
# Composite transforms
# -----------------------------

@dataclass(frozen=True)
class TailSpec:
    kind: str  # "silence" or "noise"
    duration_sec: float
    noise_snr_db: Optional[float] = None  # for kind="noise"
    fade_ms: float = 10.0


def append_tail(x: np.ndarray, sr: int, spec: TailSpec, seed: int, key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Append a tail segment and return (full_waveform, tail_segment_only).
    Tail is either zeros or AWGN scaled to fixed SNR relative to x RMS (as in methodology).
    A fade-out is applied to the end of x to avoid clicks at the boundary.
    """
    x2 = apply_fade_out(x, sr=sr, fade_ms=spec.fade_ms)
    if spec.kind == "silence":
        tail = make_silence(spec.duration_sec, sr=sr)
    elif spec.kind == "noise":
        if spec.noise_snr_db is None:
            raise ValueError("noise_snr_db must be set for noise tail")
        tail = make_noise(
            duration_sec=spec.duration_sec,
            sr=sr,
            seed=seed,
            key=f"{key}__tail_noise",
            ref_rms=rms(x),
            snr_db=float(spec.noise_snr_db),
        )
    else:
        raise ValueError(f"Unknown tail kind: {spec.kind}")
    y = np.concatenate([x2, tail]).astype(np.float32)
    y = peak_normalize_if_needed(y)
    return y, tail


@dataclass(frozen=True)
class GapSpec:
    kind: str  # "silence" or "noise"
    duration_sec: float
    noise_snr_db: Optional[float] = None  # for kind="noise"


def concat_with_gap(
    x1: np.ndarray,
    x2: np.ndarray,
    sr: int,
    gap: GapSpec,
    seed: int,
    key: str,
    ref_rms_for_noise: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Concatenate x1 + gap + x2 and return (full_waveform, gap_segment_only_or_None).
    For noise gaps, scale to a fixed SNR relative to ref_rms_for_noise (defaults to RMS(x1)).
    """
    if gap.duration_sec <= 0:
        y = np.concatenate([x1, x2]).astype(np.float32)
        y = peak_normalize_if_needed(y)
        return y, None

    if gap.kind == "silence":
        g = make_silence(gap.duration_sec, sr=sr)
    elif gap.kind == "noise":
        if gap.noise_snr_db is None:
            raise ValueError("noise_snr_db must be set for noise gap")
        ref_rms = rms(x1) if ref_rms_for_noise is None else float(ref_rms_for_noise)
        g = make_noise(
            duration_sec=gap.duration_sec,
            sr=sr,
            seed=seed,
            key=f"{key}__gap_noise",
            ref_rms=ref_rms,
            snr_db=float(gap.noise_snr_db),
        )
    else:
        raise ValueError(f"Unknown gap kind: {gap.kind}")

    y = np.concatenate([x1, g, x2]).astype(np.float32)
    y = peak_normalize_if_needed(y)
    return y, g


def duration_sec(x: np.ndarray, sr: int) -> float:
    return float(len(x) / float(sr))
