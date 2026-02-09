from __future__ import annotations

"""
CLI entrypoint for running the ISO experiments.

Typical usage (after installing the package):
  python3 -m asr_mvp.experiments.run --split test-clean

Notes:
- If you get "No module named asr_mvp", install your project in editable mode:
    pip install -e .
  or (temporary) export:
    export PYTHONPATH=src
- This script writes derived datasets and results under ./results by default.
"""

import argparse
from pathlib import Path

from .eval_runner import EvalConfig, run_full_evaluation


def _try_load_paths(config_path: Path) -> tuple[dict, Path, Path]:
    """
    Best-effort integration with the repo's Paths helper. If unavailable, fall back to
    ./cache and ./results.
    """
    try:
        from asr_mvp.utils.paths import load_config, Paths  # type: ignore
        cfg = load_config(config_path)
        paths = Paths.from_config(cfg, project_root=Path.cwd())
        cache_root = getattr(paths, "cache_root", Path("cache"))
        results_root = getattr(paths, "results_root", Path("results"))
        return cfg, Path(cache_root), Path(results_root)
    except Exception:
        return {}, Path("cache"), Path("results")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run ISO experiments (conditions 1-6) for Whisper vs Conformer+CTC.")
    p.add_argument("--config", type=str, default="conf/config.yaml", help="Path to config.yaml (optional).")
    p.add_argument("--split", type=str, default="test-clean", choices=["dev-clean", "test-clean"], help="LibriSpeech split.")
    p.add_argument("--out", type=str, default="", help="Output directory (defaults to results/<split>_<timestamp>).")
    p.add_argument("--device", type=str, default="cuda", help="Device string for models (e.g., cuda or cpu).")
    p.add_argument("--whisper-compute-type", type=str, default="float16", help="Whisper compute type (wrapper-dependent).")
    p.add_argument("--limit-core", type=int, default=0, help="Limit core subset size for quick runs (0 = full).")
    p.add_argument("--concat-n", type=int, default=1000, help="Number of concatenated examples per concat condition.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for inference (best-effort).")

    # NEW: select subsets of conditions/systems/models/decodes
    p.add_argument(
        "--only-condition",
        action="append",
        dest="only_conditions",
        default=None,
        help="Run only conditions matching this fnmatch pattern (repeatable). Example: --only-condition baseline --only-condition tail_*",
    )
    p.add_argument(
        "--systems",
        nargs="+",
        choices=["whisper", "conformer"],
        default=None,
        help="Which systems to run (default: whisper + conformer).",
    )

    # Whisper selection
    p.add_argument(
        "--whisper-model",
        action="append",
        default=None,
        help="Whisper model name (repeatable). Examples: small.en, large-v3",
    )
    p.add_argument(
        "--whisper-decode",
        action="append",
        choices=["deterministic", "sampling_t0p8"],
        default=None,
        help="Whisper decode preset (repeatable).",
    )

    # Conformer selection
    p.add_argument(
        "--conformer-model",
        action="append",
        default=None,
        help="NeMo Conformer+CTC pretrained name (repeatable). Example: stt_en_conformer_ctc_large",
    ) 
    p.add_argument(
        "--conformer-decode",
        action="append",
        choices=["greedy", "beam", "beam_lm"],
        default=None,
        help="Conformer decode preset (repeatable).",
    )
    p.add_argument("--conformer-beam-size", type=int, default=32, help="CTC beam size for beam/beam_lm.")
    p.add_argument("--conformer-token-prune-topk", type=int, default=20, help="CTC token_prune_topk for beam/beam_lm.")

    # External LM for CTC shallow fusion (KenLM / pyctcdecode)
    p.add_argument("--conformer-lm-path", type=str, default="", help="Path to KenLM .arpa/.bin for beam_lm.")
    p.add_argument("--conformer-beam-alpha", type=float, default=0.0, help="KenLM weight alpha for beam_lm.")
    p.add_argument("--conformer-beam-beta", type=float, default=0.0, help="KenLM word bonus beta for beam_lm.")

    # NEW: unigrams list for binary kenlm
    p.add_argument(
        "--conformer-lm-unigrams-path",
        type=str,
        default="",
        help="Path to 1-token-per-line unigrams file (recommended for KenLM binary .bin/.trie.bin).",
    )

    return p



def main() -> None:
    args = build_argparser().parse_args()
    config_path = Path(args.config)

    _cfg_dict, cache_root, results_root = _try_load_paths(config_path)

    out_dir = Path(args.out) if args.out else (results_root / f"{args.split}_{_now_tag()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Basic validation for LM decode
    if args.conformer_decode and ("beam_lm" in set(args.conformer_decode)) and (not args.conformer_lm_path):
        raise SystemExit("You selected --conformer-decode beam_lm but did not provide --conformer-lm-path")

    whisper_default = args.whisper_model[0] if args.whisper_model else "small.en"
    conformer_default = args.conformer_model[0] if args.conformer_model else "stt_en_conformer_ctc_large"
    systems = tuple(args.systems) if args.systems else ("whisper", "conformer")

    cfg = EvalConfig(
        # existing core config
        seed=70072,
        max_core_dur_sec=10.0,
        whisper_max_total_sec=30.0,
        concat_n_pairs=int(args.concat_n),
        concat_gap_secs=(0.0, 5.0),
        whisper_model=str(whisper_default),
        conformer_model=str(conformer_default),
        device=str(args.device),
        whisper_compute_type=str(args.whisper_compute_type),
        batch_size=int(args.batch_size),
        limit_core=int(args.limit_core),

        # NEW selection
        only_conditions=args.only_conditions,
        systems=systems,
        whisper_models=args.whisper_model,
        whisper_decode_names=args.whisper_decode,
        conformer_models=args.conformer_model,
        conformer_decode_names=args.conformer_decode,
        conformer_beam_size=int(args.conformer_beam_size),
        conformer_token_prune_topk=int(args.conformer_token_prune_topk),
        conformer_lm_path=(Path(args.conformer_lm_path) if args.conformer_lm_path else None),
        conformer_beam_alpha=float(args.conformer_beam_alpha),
        conformer_beam_beta=float(args.conformer_beam_beta),
        conformer_lm_unigrams_path=(Path(args.conformer_lm_unigrams_path) if args.conformer_lm_unigrams_path else None),
    )

    run_full_evaluation(split_name=str(args.split), cache_root=cache_root, out_dir=out_dir, cfg=cfg)
    print(f"[OK] Wrote results to: {out_dir}")


def _now_tag() -> str:
    import time
    return time.strftime("%Y-%m-%d_%H%M%S", time.localtime())


if __name__ == "__main__":
    main()
