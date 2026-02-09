from __future__ import annotations

"""
NeMo Conformer+CTC runner used by experiments.
...
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import os
import inspect



@dataclass(frozen=True)
class ConformerRunnerConfig:
    pretrained_name: str = "stt_en_conformer_ctc_large"
    cache_dir: Path = Path("cache/nemo")
    device: str = "cuda"  # "cuda" or "cpu"
    decode_mode: str = "greedy"  # "greedy" | "beam"
    beam_size: int = 32
    token_prune_topk: int = 20

    nemo_local_path: Optional[Path] = None

    # External LM (KenLM) shallow fusion via pyctcdecode.
    lm_path: Optional[Path] = None
    beam_alpha: float = 0.0
    beam_beta: float = 0.0

    # NEW: for KenLM *binary* files, pyctcdecode can't infer unigrams automatically.
    # Provide a one-token-per-line file (typically extracted from the ARPA 1-grams section).
    lm_unigrams_path: Optional[Path] = None


class ConformerCTCRunner:
    def __init__(self, cfg: ConformerRunnerConfig):
        self.cfg = cfg
        self.model = self._load_model()
        self._configure_decoding()

    def _load_model(self):
        try:
            import torch
            from nemo.collections.asr.models import EncDecCTCModelBPE
        except Exception as e:
            raise RuntimeError(
                "Conformer+CTC runner requires NVIDIA NeMo (nemo_toolkit[asr]) and torch.\n"
                f"Original error: {e}"
            )

        try:
            model = EncDecCTCModelBPE.from_pretrained(model_name=self.cfg.pretrained_name)
        except TypeError:
            model = EncDecCTCModelBPE.from_pretrained(self.cfg.pretrained_name)

        # Persist locally for future runs (best-effort).
        try:
            p = Path(self.cfg.nemo_local_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            model.save_to(str(p))
        except Exception:
            pass

        device = torch.device(self.cfg.device if (self.cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def _load_unigrams_file(p: Path) -> List[str]:
        # One token per line. For word-LM, these are WORDS.
        # Filter common specials (safe).
        specials = {"<unk>", "<s>", "</s>"}
        out: List[str] = []
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                t = line.strip()
                if not t or t in specials:
                    continue
                out.append(t)
        return out


    def _inject_pyctcdecode_decoder(self) -> None:
        """
        Force pyctcdecode to use our unigrams list even when kenlm_path is a binary .bin/.trie.bin,
        by pre-building the decoder and attaching it to NeMo's beam decoder object.

        This avoids the warning:
        "Unigrams not provided and cannot be automatically determined from LM file (only arpa format)."
        """
        if not (self.cfg.lm_path and self.cfg.lm_unigrams_path):
            return

        try:
            import inspect
            import pyctcdecode
        except Exception:
            return

        kenlm_path = str(self.cfg.lm_path)
        unigrams = self._load_unigrams_file(Path(self.cfg.lm_unigrams_path))

        # Find NeMo's beam-decoder object
        ctc_decoding = getattr(self.model, "decoding", None)
        if ctc_decoding is None:
            return

        beam_decoder = getattr(ctc_decoding, "decoding", None)
        if beam_decoder is None:
            return

        # If NeMo already built one, don't override.
        if getattr(beam_decoder, "pyctcdecode_beam_scorer", None) is not None:
            return

        # Try to get the exact labels list NeMo uses (must match logit indices)
        labels = None
        for attr in ("labels", "vocabulary", "vocab_list", "_labels", "_vocabulary"):
            v = getattr(beam_decoder, attr, None)
            if isinstance(v, (list, tuple)) and len(v) > 0:
                labels = list(v)
                break

        # Fallbacks (best-effort)
        if labels is None:
            v = getattr(getattr(self.model, "decoder", None), "vocabulary", None)
            if isinstance(v, (list, tuple)) and len(v) > 0:
                labels = list(v)

        if not labels:
            return

        # Build kwargs robustly (pyctcdecode signatures can vary)
        sig = inspect.signature(pyctcdecode.build_ctcdecoder)
        params = sig.parameters

        kwargs = {}
        if "kenlm_model_path" in params:
            kwargs["kenlm_model_path"] = kenlm_path
        elif "kenlm_path" in params:
            kwargs["kenlm_path"] = kenlm_path

        if "unigrams" in params:
            kwargs["unigrams"] = unigrams

        if "alpha" in params:
            kwargs["alpha"] = float(self.cfg.beam_alpha)
        if "beta" in params:
            kwargs["beta"] = float(self.cfg.beam_beta)

        # These help a lot for SentencePiece/BPE token sets if supported by your pyctcdecode version
        if "is_bpe" in params:
            kwargs["is_bpe"] = True
        if "bpe_separator" in params:
            kwargs["bpe_separator"] = "▁"

        # Word delimiter (pyctcdecode uses this for word-level LM scoring)
        if "word_delimiter_token" in params:
            kwargs["word_delimiter_token"] = " "

        # Pass pruning params if supported
        if "beam_prune_logp" in params:
            kwargs["beam_prune_logp"] = -10.0
        if "token_min_logp" in params:
            kwargs["token_min_logp"] = -5.0
        if "prune_history" in params:
            kwargs["prune_history"] = False
        if "hotwords" in params:
            kwargs["hotwords"] = None
        if "hotword_weight" in params:
            kwargs["hotword_weight"] = 10.0

        decoder = pyctcdecode.build_ctcdecoder(labels, **kwargs)
        beam_decoder.pyctcdecode_beam_scorer = decoder



    def _configure_decoding(self) -> None:
        mode = str(self.cfg.decode_mode).lower()
        if mode not in {"greedy", "beam"}:
            raise ValueError(f"decode_mode must be greedy|beam, got: {self.cfg.decode_mode}")

        try:
            from omegaconf import OmegaConf
        except Exception:
            OmegaConf = None  # type: ignore

        if mode == "greedy":
            dec = {"strategy": "greedy_batch"}
        else:
            # Beam decoding via pyctcdecode (LM optional)
            try:
                import pyctcdecode  # noqa: F401
            except Exception as e:
                raise RuntimeError(
                    "Beam decoding via pyctcdecode requires pyctcdecode:\n"
                    "  pip install -U pyctcdecode\n"
                    f"Import error: {e}"
                )

            kenlm_path = str(self.cfg.lm_path) if self.cfg.lm_path else None
            alpha = float(self.cfg.beam_alpha) if self.cfg.lm_path else 0.0
            beta = float(self.cfg.beam_beta) if self.cfg.lm_path else 0.0

            # --- IMPORTANT: force NeMo's internal pyctcdecode.build_ctcdecoder() to receive unigrams ---
            unigrams: Optional[List[str]] = None
            orig_build = None

            if self.cfg.lm_path and self.cfg.lm_unigrams_path:
                try:
                    unigrams = self._load_unigrams_file(Path(self.cfg.lm_unigrams_path))
                except Exception:
                    unigrams = None

            if unigrams:
                try:
                    import pyctcdecode
                    orig_build = pyctcdecode.build_ctcdecoder
                    sig = inspect.signature(orig_build)
                    param_names = list(sig.parameters.keys())

                    # Find where "unigrams" would be in positional args (after "labels")
                    unigrams_pos = None
                    if "unigrams" in param_names:
                        # position in full signature, including labels
                        unigrams_pos = param_names.index("unigrams") - 1  # minus 1 because *args excludes labels

                    def _wrapped_build_ctcdecoder(labels, *args, **kwargs):
                        # Detect whether caller already provided unigrams (kw or positional)
                        provided = ("unigrams" in kwargs and kwargs["unigrams"] is not None)

                        if (not provided) and (unigrams_pos is not None) and (len(args) > unigrams_pos):
                            if args[unigrams_pos] is not None:
                                provided = True

                        if not provided:
                            kwargs["unigrams"] = unigrams

                        # Keep BPE flags explicit if supported (harmless if ignored)
                        try:
                            ps = inspect.signature(orig_build).parameters
                            if "is_bpe" in ps and kwargs.get("is_bpe", None) is None:
                                kwargs["is_bpe"] = True
                            if "bpe_separator" in ps and kwargs.get("bpe_separator", None) is None:
                                kwargs["bpe_separator"] = "▁"
                        except Exception:
                            pass

                        return orig_build(labels, *args, **kwargs)

                    pyctcdecode.build_ctcdecoder = _wrapped_build_ctcdecoder

                    import sys
                    for m in list(sys.modules.values()):
                        if m is None:
                            continue
                        try:
                            if getattr(m, "build_ctcdecoder", None) is orig_build:
                                setattr(m, "build_ctcdecoder", _wrapped_build_ctcdecoder)
                        except Exception:
                            pass


                except Exception:
                    orig_build = None


            # IMPORTANT:
            # For SentencePiece-based models like stt_en_conformer_ctc_large,
            # keep NeMo's word separator as ▁ (space is not in vocab in your earlier check).
            dec = {
                "strategy": "pyctcdecode",
                "word_seperator": "▁",

                "beam": {
                    "beam_size": int(self.cfg.beam_size),
                    "search_type": "pyctcdecode",

                    "beam_prune_topk": int(self.cfg.token_prune_topk),
                    "token_prune_topk": int(self.cfg.token_prune_topk),

                    "kenlm_path": kenlm_path,
                    "beam_alpha": alpha,
                    "beam_beta": beta,

                    "pyctcdecode_cfg": {
                        "beam_prune_logp": -10.0,
                        "token_min_logp": -5.0,
                        "prune_history": False,
                        "hotwords": None,
                        "hotword_weight": 10.0,
                    },
                },
            }

        cfg_obj = OmegaConf.create(dec) if OmegaConf is not None else dec

        if hasattr(self.model, "decoding"):
            self.model.decoding = None

        applied = False


        if hasattr(self.model, "change_decoding_strategy"):
            try:
                self.model.change_decoding_strategy(cfg_obj)  # type: ignore
                applied = True
            except Exception:
                applied = False

        try:
            if hasattr(self.model, "cfg") and hasattr(self.model.cfg, "decoding"):
                self.model.cfg.decoding = cfg_obj  # type: ignore
                applied = True
        except Exception:
            pass

        if hasattr(self.model, "_setup_decoding"):
            try:
                self.model._setup_decoding()
            except Exception:
                pass

        # optional: keep this as a fallback (won't hurt)
        self._inject_pyctcdecode_decoder()

        if not applied:
            return


    def transcribe(self, audio_paths: List[Path], batch_size: int = 16) -> List[str]:

        # Fallback in case NeMo rebuilds decoding lazily at first transcribe
        try:
            self._inject_pyctcdecode_decoder()
        except Exception:
            pass


        paths = [str(p) for p in audio_paths]
        bs = int(batch_size)

        try:
            out = self.model.transcribe(audio=paths, batch_size=bs)
        except TypeError:
            try:
                out = self.model.transcribe(paths, batch_size=bs)
            except TypeError:
                out = self.model.transcribe(paths2audio_files=paths, batch_size=bs)

        def _to_text(x) -> str:
            if isinstance(x, str):
                return x
            if hasattr(x, "text"):
                return str(getattr(x, "text"))
            return str(x)

        return [_to_text(s) for s in out]



# from __future__ import annotations

# """
# NeMo Conformer+CTC runner used by experiments.

# Methodology (main experiments):
# - model: stt_en_conformer_ctc_large
# - decoding modes:
#   (i) greedy (best-path)
#   (ii) CTC beam decoding with beam_size=32, token_prune_topk=20
# - no external LM (lm_path=None)

# This runner is intentionally self-contained so experiments do not depend on extra wrappers.
# """

# from dataclasses import dataclass
# from pathlib import Path
# import re
# from typing import List, Optional


# @dataclass(frozen=True)
# class ConformerRunnerConfig:
#     pretrained_name: str = "stt_en_conformer_ctc_large"
#     cache_dir: Path = Path("cache/nemo")
#     nemo_local_path: Optional[Path] = None
#     device: str = "cuda"  # "cuda" or "cpu"
#     decode_mode: str = "greedy"  # "greedy" | "beam"
#     beam_size: int = 32
#     token_prune_topk: int = 20

#     # External LM (KenLM) for shallow fusion via pyctcdecode. If lm_path is None, decoding is LM-free.
#     lm_path: Optional[Path] = None
#     beam_alpha: float = 0.0  # LM weight (only used if lm_path is set)
#     beam_beta: float = 0.0   # word insertion bonus (only used if lm_path is set)


# class ConformerCTCRunner:
#     def __init__(self, cfg: ConformerRunnerConfig):
#         self.cfg = cfg
#         self.model = self._load_model()
#         self._configure_decoding()

#     def _load_model(self):
#         try:
#             import torch
#             from nemo.collections.asr.models import EncDecCTCModelBPE
#         except Exception as e:
#             raise RuntimeError(
#                 "Conformer+CTC runner requires NVIDIA NeMo (nemo_toolkit[asr]) and torch.\n"
#                 f"Original error: {e}"
#             )

#         # NeMo API varies slightly across versions; try common signatures.
#         try:
#             model = EncDecCTCModelBPE.from_pretrained(model_name=self.cfg.pretrained_name)
#         except TypeError:
#             model = EncDecCTCModelBPE.from_pretrained(self.cfg.pretrained_name)

#         # Persist locally for future runs (best-effort).
#         try:
#             p = self.cfg.nemo_local_path or (Path(self.cfg.cache_dir) / f"{self.cfg.pretrained_name}.nemo")
#             p.parent.mkdir(parents=True, exist_ok=True)
#             model.save_to(str(p))
#         except Exception:
#             pass

#         device = torch.device(self.cfg.device if (self.cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
#         model = model.to(device)
#         model.eval()
#         return model

#     def _configure_decoding(self) -> None:
#         mode = str(self.cfg.decode_mode).lower()
#         if mode not in {"greedy", "beam"}:
#             raise ValueError(f"decode_mode must be greedy|beam, got: {self.cfg.decode_mode}")

#         try:
#             from omegaconf import OmegaConf
#         except Exception:
#             OmegaConf = None  # type: ignore

#         # Greedy: use the faster implementation NeMo itself recommends.
#         if mode == "greedy":
#             dec = {"strategy": "greedy_batch"}  # same output interface, faster than greedy
#         else:
#             # Beam decoding via pyctcdecode (with optional KenLM)
#             try:
#                 import pyctcdecode  # noqa: F401
#             except Exception as e:
#                 raise RuntimeError(
#                     "Beam decoding via pyctcdecode requires pyctcdecode:\n"
#                     "  pip install -U pyctcdecode\n"
#                     f"Import error: {e}"
#                 )

#             use_lm = bool(self.cfg.lm_path) and (float(self.cfg.beam_alpha) != 0.0 or float(self.cfg.beam_beta) != 0.0)

#             kenlm_path = str(self.cfg.lm_path) if use_lm else None
#             alpha = float(self.cfg.beam_alpha) if use_lm else 0.0
#             beta = float(self.cfg.beam_beta) if use_lm else 0.0


#             dec = {
#                 "strategy": "pyctcdecode",
#                 "word_seperator": " ",
#                 "beam": {
#                     "beam_size": int(self.cfg.beam_size),
#                     "search_type": "pyctcdecode",
#                     "beam_prune_topk": int(self.cfg.token_prune_topk),
#                     "token_prune_topk": int(self.cfg.token_prune_topk),

#                     "kenlm_path": kenlm_path,
#                     "beam_alpha": alpha,
#                     "beam_beta": beta,

#                     "pyctcdecode_cfg": {
#                         "beam_prune_logp": -10.0,
#                         "token_min_logp": -5.0,
#                         "prune_history": False,
#                         "hotwords": None,
#                         "hotword_weight": 10.0,
#                     },
#                 },
#             }


#         # Apply config and FORCE rebuild of decoding modules.
#         cfg_obj = OmegaConf.create(dec) if OmegaConf is not None else dec

#         if hasattr(self.model, "decoding"):
#             self.model.decoding = None

#         applied = False
#         if hasattr(self.model, "change_decoding_strategy"):
#             try:
#                 self.model.change_decoding_strategy(cfg_obj)  # type: ignore
#                 applied = True
#             except Exception:
#                 applied = False

#         # Even if change_decoding_strategy "worked", NeMo 2.2.1 can keep the old decoder object.
#         # So we also patch cfg directly and rebuild decoding.
#         try:
#             if hasattr(self.model, "cfg") and hasattr(self.model.cfg, "decoding"):
#                 self.model.cfg.decoding = cfg_obj  # type: ignore
#                 applied = True
#         except Exception:
#             pass

#         # Critical: rebuild the decoding module so search_type actually takes effect.
#         if hasattr(self.model, "_setup_decoding"):
#             try:
#                 self.model._setup_decoding()
#             except Exception:
#                 pass

#         if not applied:
#             # last resort: do nothing, model will run with its defaults
#             return

#     def transcribe(self, audio_paths: List[Path], batch_size: int = 16) -> List[str]:
#         paths = [str(p) for p in audio_paths]
#         bs = int(batch_size)

#         # Try the common NeMo 2.x signature first
#         try:
#             out = self.model.transcribe(audio=paths, batch_size=bs)
#         except TypeError:
#             # Some versions accept positional audio list
#             try:
#                 out = self.model.transcribe(paths, batch_size=bs)
#             except TypeError:
#                 # Older signature name
#                 out = self.model.transcribe(paths2audio_files=paths, batch_size=bs)

#         # NeMo can return either List[str] or List[Hypothesis]. Normalize to plain text.
#         def _to_text(x) -> str:
#             if isinstance(x, str):
#                 return x
#             if hasattr(x, "text"):
#                 return str(getattr(x, "text"))
#             return str(x)

#         def _normalize_text(s: str) -> str:
#             # SentencePiece uses ▁ to indicate word boundary; make it normal text for scoring.
#             s = s.replace("▁", " ")
#             # collapse whitespace
#             s = re.sub(r"\s+", " ", s).strip()
#             return s

#         return [_normalize_text(_to_text(s)) for s in out]









# from __future__ import annotations

# """
# NeMo Conformer+CTC runner used by experiments.

# Methodology (main experiments):
# - model: stt_en_conformer_ctc_large
# - decoding modes:
#   (i) greedy (best-path)
#   (ii) CTC beam decoding with beam_size=32, token_prune_topk=20
# - no external LM (lm_path=None)

# This runner is intentionally self-contained so experiments do not depend on extra wrappers.
# """

# from dataclasses import dataclass
# from pathlib import Path
# from typing import List, Optional


# @dataclass(frozen=True)
# class ConformerRunnerConfig:
#     pretrained_name: str = "stt_en_conformer_ctc_large"
#     cache_dir: Path = Path("cache/nemo")
#     device: str = "cuda"  # "cuda" or "cpu"
#     decode_mode: str = "greedy"  # "greedy" | "beam"
#     beam_size: int = 32
#     token_prune_topk: int = 20

#     # External LM (KenLM) for shallow fusion via pyctcdecode. If lm_path is None, decoding is LM-free.
#     lm_path: Optional[Path] = None
#     beam_alpha: float = 0.0  # LM weight (only used if lm_path is set)
#     beam_beta: float = 0.0   # word insertion bonus (only used if lm_path is set)



# class ConformerCTCRunner:
#     def __init__(self, cfg: ConformerRunnerConfig):
#         self.cfg = cfg
#         self.model = self._load_model()
#         self._configure_decoding()

#     def _load_model(self):
#         try:
#             import torch
#             from nemo.collections.asr.models import EncDecCTCModelBPE
#         except Exception as e:
#             raise RuntimeError(
#                 "Conformer+CTC runner requires NVIDIA NeMo (nemo_toolkit[asr]) and torch.\n"
#                 f"Original error: {e}"
#             )

#         # NeMo API varies slightly across versions; try common signatures.
#         try:
#             model = EncDecCTCModelBPE.from_pretrained(model_name=self.cfg.pretrained_name)
#         except TypeError:
#             model = EncDecCTCModelBPE.from_pretrained(self.cfg.pretrained_name)

#         # Persist locally for future runs (best-effort).
#         try:
#             p = Path(self.cfg.nemo_local_path)
#             p.parent.mkdir(parents=True, exist_ok=True)
#             model.save_to(str(p))
#         except Exception:
#             pass
#         device = torch.device(self.cfg.device if (self.cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
#         model = model.to(device)
#         model.eval()
#         return model



#     def _configure_decoding(self) -> None:
#         mode = str(self.cfg.decode_mode).lower()
#         if mode not in {"greedy", "beam"}:
#             raise ValueError(f"decode_mode must be greedy|beam, got: {self.cfg.decode_mode}")

#         try:
#             from omegaconf import OmegaConf
#         except Exception:
#             OmegaConf = None  # type: ignore

#         # Greedy: use the faster implementation NeMo itself recommends.
#         if mode == "greedy":
#             dec = {"strategy": "greedy_batch"}  # same output interface, faster than greedy
#         else:
#             # Beam WITHOUT KenLM: use NeMo's pyctcdecode strategy (not "beam")
#             try:
#                 import pyctcdecode  # noqa: F401
#             except Exception as e:
#                 raise RuntimeError(
#                     "Beam decoding via pyctcdecode requires pyctcdecode:\n"
#                     "  pip install -U pyctcdecode\n"
#                     f"Import error: {e}"
#                 )

#             # dec = {
#             #     # ✅ THIS is the key fix: strategy must be "pyctcdecode"
#             #     "strategy": "pyctcdecode",
#             #     "beam": {
#             #         "beam_size": int(self.cfg.beam_size),
#             #         "search_type": "pyctcdecode",

#             #         # pruning knobs (optional)
#             #         "beam_prune_topk": int(self.cfg.token_prune_topk),
#             #         "token_prune_topk": int(self.cfg.token_prune_topk),

#             #         # ✅ Explicitly no LM
#             #         "kenlm_path": None,
#             #         "beam_alpha": 0.0,
#             #         "beam_beta": 0.0,

#             #         "pyctcdecode_cfg": {
#             #             "beam_prune_logp": -10.0,
#             #             "token_min_logp": -5.0,
#             #             "prune_history": False,
#             #             "hotwords": None,
#             #             "hotword_weight": 10.0,
#             #         },
#             #     },
#             # }

#             kenlm_path = str(self.cfg.lm_path) if self.cfg.lm_path else None
#             alpha = float(self.cfg.beam_alpha) if self.cfg.lm_path else 0.0
#             beta = float(self.cfg.beam_beta) if self.cfg.lm_path else 0.0

#             dec = {
#                 # ✅ THIS is the key fix: strategy must be "pyctcdecode"
#                 "strategy": "pyctcdecode",
#                 "beam": {
#                     "beam_size": int(self.cfg.beam_size),
#                     "search_type": "pyctcdecode",

#                     # pruning knobs (optional)
#                     "beam_prune_topk": int(self.cfg.token_prune_topk),
#                     "token_prune_topk": int(self.cfg.token_prune_topk),

#                     # External LM via KenLM (optional)
#                     "kenlm_path": kenlm_path,
#                     "beam_alpha": alpha,
#                     "beam_beta": beta,

#                     "pyctcdecode_cfg": {
#                         "beam_prune_logp": -10.0,
#                         "token_min_logp": -5.0,
#                         "prune_history": False,
#                         "hotwords": None,
#                         "hotword_weight": 10.0,
#                     },
#                 },
#             }


#         # Apply config and FORCE rebuild of decoding modules.
#         cfg_obj = OmegaConf.create(dec) if OmegaConf is not None else dec

#         if hasattr(self.model, "decoding"):
#             self.model.decoding = None
            
#         applied = False
#         if hasattr(self.model, "change_decoding_strategy"):
#             try:
#                 self.model.change_decoding_strategy(cfg_obj)  # type: ignore
#                 applied = True
#             except Exception:
#                 applied = False

#         # Even if change_decoding_strategy "worked", NeMo 2.2.1 can keep the old decoder object.
#         # So we also patch cfg directly and rebuild decoding.
#         try:
#             if hasattr(self.model, "cfg") and hasattr(self.model.cfg, "decoding"):
#                 self.model.cfg.decoding = cfg_obj  # type: ignore
#                 applied = True
#         except Exception:
#             pass

#         # Critical: rebuild the decoding module so search_type actually takes effect.
#         if hasattr(self.model, "_setup_decoding"):
#             try:
#                 self.model._setup_decoding()
#             except Exception:
#                 pass

#         if not applied:
#             # last resort: do nothing, model will run with its defaults
#             return

#     def transcribe(self, audio_paths: List[Path], batch_size: int = 16) -> List[str]:
#         paths = [str(p) for p in audio_paths]
#         bs = int(batch_size)

#         # Try the common NeMo 2.x signature first
#         try:
#             out = self.model.transcribe(audio=paths, batch_size=bs)
#         except TypeError:
#             # Some versions accept positional audio list
#             try:
#                 out = self.model.transcribe(paths, batch_size=bs)
#             except TypeError:
#                 # Older signature name
#                 out = self.model.transcribe(paths2audio_files=paths, batch_size=bs)

#         # NeMo can return either List[str] or List[Hypothesis]. Normalize to plain text.
#         def _to_text(x) -> str:
#             if isinstance(x, str):
#                 return x
#             if hasattr(x, "text"):
#                 return str(getattr(x, "text"))
#             return str(x)

#         return [_to_text(s) for s in out]
