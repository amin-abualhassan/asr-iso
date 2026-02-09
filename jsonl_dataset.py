
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


@dataclass(frozen=True)
class JsonlExample:
    id: str
    audio_path: Path
    text: str
    duration_sec: Optional[float] = None
    speaker: Optional[str] = None


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_dataset_dir(ds_dir: Path) -> List[JsonlExample]:
    """
    Dataset format:
      ds_dir/
        refs.jsonl  lines: {"id":..., "audio_path":"audio/<id>.wav", "text":..., "duration_sec":..., "speaker":...}
        audio/*.wav

    IMPORTANT:
      We resolve ds_dir to an absolute path and return absolute audio paths.
      This prevents accidental double-prefixing when downstream code also
      tries to resolve relative paths.
    """
    ds_dir = ds_dir.resolve()

    refs = ds_dir / "refs.jsonl"
    if not refs.exists():
        raise FileNotFoundError(f"Missing refs.jsonl: {refs}")

    out: List[JsonlExample] = []
    for r in read_jsonl(refs):
        ap = (ds_dir / r["audio_path"]).resolve()
        out.append(
            JsonlExample(
                id=str(r["id"]),
                audio_path=ap,
                text=str(r.get("text", "")),
                duration_sec=(float(r["duration_sec"]) if "duration_sec" in r and r["duration_sec"] is not None else None),
                speaker=(str(r["speaker"]) if "speaker" in r and r["speaker"] is not None else None),
            )
        )
    return out


# from __future__ import annotations

# import json
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Iterable, Iterator, List, Optional


# @dataclass(frozen=True)
# class JsonlExample:
#     id: str
#     audio_path: Path
#     text: str
#     duration_sec: Optional[float] = None
#     speaker: Optional[str] = None


# def read_jsonl(path: Path) -> Iterator[dict]:
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             yield json.loads(line)


# def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with path.open("w", encoding="utf-8") as f:
#         for r in rows:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")


# def load_dataset_dir(ds_dir: Path) -> List[JsonlExample]:
#     """
#     Dataset format:
#       ds_dir/
#         refs.jsonl  lines: {"id":..., "audio_path":"audio/<id>.wav", "text":..., "duration_sec":..., "speaker":...}
#         audio/*.wav

#     IMPORTANT:
#       We resolve ds_dir to an absolute path and return absolute audio paths.
#       This prevents accidental double-prefixing when downstream code also
#       tries to resolve relative paths.
#     """
#     ds_dir = ds_dir.resolve()

#     refs = ds_dir / "refs.jsonl"
#     if not refs.exists():
#         raise FileNotFoundError(f"Missing refs.jsonl: {refs}")

#     out: List[JsonlExample] = []
#     for r in read_jsonl(refs):
#         ap = (ds_dir / r["audio_path"]).resolve()
#         out.append(
#             JsonlExample(
#                 id=str(r["id"]),
#                 audio_path=ap,
#                 text=str(r.get("text", "")),
#                 duration_sec=(float(r["duration_sec"]) if "duration_sec" in r and r["duration_sec"] is not None else None),
#                 speaker=(str(r["speaker"]) if "speaker" in r and r["speaker"] is not None else None),
#             )
#         )
#     return out
