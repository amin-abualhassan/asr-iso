from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import os

import whisper

from ..utils.device import pick_device
from .base import AudioTextExample


@dataclass
class WhisperConfig:
    name: str
    device: str
    compute_type: str
    language: str
    task: str
    beam_size: int
    temperature: float
    condition_on_previous_text: bool


class WhisperASR:
    def __init__(self, cfg: WhisperConfig, cache_dir: Path):
        self.cfg = cfg
        self.name = f"whisper::{cfg.name}"
        self._device_spec = pick_device(cfg.device)

        cache_dir.mkdir(parents=True, exist_ok=True)

        # Offline guard: Whisper downloads its weights if missing.
        # If you want strict offline runs, set WHISPER_OFFLINE=1 and warm caches first.
        offline = os.environ.get("WHISPER_OFFLINE") == "1"
        expected = cache_dir / f"{cfg.name}.pt"
        if offline and not expected.exists() and not any(cache_dir.glob("*.pt")):
            raise FileNotFoundError(
                f"Whisper offline requested but weights not found in {cache_dir}. "
                f"Run: python -m asr_mvp.cache_warmup"
            )

        fp16 = self._device_spec.is_cuda and (cfg.compute_type == "float16")
        self._model = whisper.load_model(cfg.name, device=self._device_spec.device, download_root=str(cache_dir))
        self._fp16 = fp16

    def transcribe(self, examples: List[AudioTextExample]) -> List[str]:
        outs: List[str] = []
        for ex in examples:
            result = self._model.transcribe(
                str(ex.audio_path),
                language=self.cfg.language,
                task=self.cfg.task,
                temperature=self.cfg.temperature,
                beam_size=self.cfg.beam_size,
                condition_on_previous_text=self.cfg.condition_on_previous_text,
                fp16=self._fp16,
                verbose=False,
            )
            outs.append((result.get("text") or "").strip())
        return outs
