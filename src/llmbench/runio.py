"""Run directory I/O: run-id directories, manifest.json, and per-rep JSONL records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llmbench.manifest import make_run_id


def create_run_dir(root: Path | str, fingerprint: str) -> Path:
    """Create and return ``root/<run_id>/`` where run_id = fingerprint[:12] + UTC stamp.

    Parents are created as needed; an existing directory is never reused. The
    run-id timestamp has 1-second resolution, so a same-second collision for
    one fingerprint gets a ``-2``, ``-3``, ... suffix instead of crashing the
    colliding run.
    """
    base_id = make_run_id(fingerprint, datetime.now(UTC))
    root_path = Path(root)
    for attempt in range(1, 1000):
        run_id = base_id if attempt == 1 else f"{base_id}-{attempt}"
        run_dir = root_path / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return run_dir
    raise FileExistsError(f"could not create a unique run dir under {root_path} for {base_id}")


def write_manifest(run_dir: Path | str, manifest: Mapping[str, object]) -> Path:
    """Write ``manifest.json`` (pretty-printed, key order preserved) into run_dir."""
    path = Path(run_dir) / "manifest.json"
    path.write_text(json.dumps(dict(manifest), indent=2) + "\n", encoding="utf-8")
    return path


def append_records_jsonl(path: Path | str, records: Iterable[Mapping[str, object]]) -> None:
    """Append records to a JSONL file, one compact JSON object per line."""
    lines = "".join(json.dumps(dict(r), separators=(",", ":")) + "\n" for r in records)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(lines)


def read_records_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSONL record file back into a list of dicts, skipping blank lines.

    Raises ValueError naming the file and line on a corrupt record (e.g. a
    final line truncated by a killed writer), so callers can report the state
    of a partially-written run instead of surfacing a raw traceback.
    """
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {lineno}: invalid JSON record: {exc}") from exc
    return records
