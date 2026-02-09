from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from .jsonl_dataset import write_jsonl


def _split_to_hf_config_and_split(name: str) -> Tuple[str, str]:
    """
    Supported names:
      dev-clean, test-clean, test-other,
      train-clean-100, train-clean-360, train-other-500
    """
    name = name.strip().lower()
    if name == "dev-clean":
        return "clean", "validation"
    if name == "test-clean":
        return "clean", "test"
    if name == "test-other":
        return "other", "test"
    if name == "train-clean-100":
        return "clean", "train.100"
    if name == "train-clean-360":
        return "clean", "train.360"
    if name == "train-other-500":
        return "other", "train.500"
    raise ValueError(f"Unsupported split: {name}")


def cache_librispeech(
    out_root: Path,
    split_name: str,
    limit: int = 0,
    seed: int = 0,
) -> Path:
    """
    Downloads LibriSpeech via HuggingFace datasets and writes:
      out_root/librispeech/<split_name>/{audio/*.wav, refs.jsonl, meta.json}

    Returns the dataset directory path.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "Missing dependency `datasets`. Install with: pip install datasets\n"
            f"Original error: {e}"
        )

    try:
        import numpy as np
        import soundfile as sf
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for audio IO (`soundfile` and `numpy`).\n"
            "Install with: pip install soundfile numpy\n"
            f"Original error: {e}"
        )

    cfg, hf_split = _split_to_hf_config_and_split(split_name)
    ds = load_dataset("librispeech_asr", cfg, split=hf_split)

    ds_dir = out_root / "librispeech" / split_name
    audio_dir = ds_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    n = len(ds)
    take = n if limit <= 0 else min(limit, n)

    rows: List[Dict] = []
    for i in range(take):
        ex = ds[i]
        utt_id = str(ex.get("id", f"{split_name}_{i:06d}"))

        audio = ex["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])

        target_sr = 16000
        if sr != target_sr:
            try:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr).astype(np.float32)
                sr = target_sr
            except Exception:
                ratio = target_sr / sr
                idx = (np.arange(int(len(arr) * ratio)) / ratio).astype(np.int64)
                idx = np.clip(idx, 0, len(arr) - 1)
                arr = arr[idx].astype(np.float32)
                sr = target_sr

        dur = float(len(arr) / sr)
        wav_path = audio_dir / f"{utt_id}.wav"
        sf.write(str(wav_path), arr, sr)

        text = str(ex.get("text", ""))
        speaker = utt_id.split("-")[0] if "-" in utt_id else None

        rows.append(
            {
                "id": utt_id,
                "audio_path": f"audio/{utt_id}.wav",
                "text": text,
                "duration_sec": dur,
                "speaker": speaker,
            }
        )

    write_jsonl(ds_dir / "refs.jsonl", rows)
    (ds_dir / "meta.json").write_text(
        json.dumps(
            {
                "source": "librispeech_asr (HF datasets)",
                "config": cfg,
                "split": hf_split,
                "split_name": split_name,
                "limit": int(limit),
                "seed": int(seed),
                "num_examples": int(take),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ds_dir
