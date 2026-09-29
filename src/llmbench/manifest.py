"""Run manifests: reproducibility metadata with a deterministic config fingerprint.

A manifest records everything needed to reproduce a benchmark run. The
fingerprint covers only the configuration (model, engine, quantization,
hardware, workload, loadgen) so that runs with identical configs are
identifiable across time, while ``created_at`` and ``notes`` never affect it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

SCHEMA_VERSION = 1

# Keys excluded from fingerprinting: they vary between otherwise-identical runs.
_FINGERPRINT_EXCLUDED = frozenset({"fingerprint", "created_at", "notes"})


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Identity of the model under test."""

    id: str
    revision: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the manifest ``model`` sub-dict."""
        return {"id": self.id, "revision": self.revision}


@dataclass(frozen=True, slots=True)
class EngineInfo:
    """Serving engine identity and launch configuration."""

    name: str
    version: str | None = None
    image: str | None = None
    image_digest: str | None = None
    serving_args: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return the manifest ``engine`` sub-dict."""
        return {
            "name": self.name,
            "version": self.version,
            "image": self.image,
            "image_digest": self.image_digest,
            "serving_args": list(self.serving_args),
        }


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    """Hardware the engine ran on; all fields optional (unknown off-GPU)."""

    gpu_name: str | None = None
    gpu_count: int | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    vram_mib: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the manifest ``hardware`` sub-dict."""
        return {
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "vram_mib": self.vram_mib,
        }


@dataclass(frozen=True, slots=True)
class LoadgenInfo:
    """Load-generator settings for the run."""

    harness_version: str
    concurrency: int
    num_requests: int
    warmup_requests: int
    repeats: int
    timeout_s: float

    def to_dict(self) -> dict[str, object]:
        """Return the manifest ``loadgen`` sub-dict."""
        return {
            "harness_version": self.harness_version,
            "concurrency": self.concurrency,
            "num_requests": self.num_requests,
            "warmup_requests": self.warmup_requests,
            "repeats": self.repeats,
            "timeout_s": self.timeout_s,
        }


def fingerprint(manifest: Mapping[str, object]) -> str:
    """Return the sha256 hex fingerprint of a manifest dict.

    Excludes ``fingerprint``, ``created_at``, and ``notes``; canonicalizes the
    rest as compact, key-sorted, ASCII-only JSON so identical configs always
    hash identically.
    """
    core = {k: v for k, v in manifest.items() if k not in _FINGERPRINT_EXCLUDED}
    payload = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def make_run_id(fingerprint_hex: str, created_at: datetime) -> str:
    """Return ``<fingerprint[:12]>-<UTC YYYYMMDDTHHMMSSZ>`` for a run directory.

    ``created_at`` must be timezone-aware; it is converted to UTC. Passing time
    explicitly keeps fingerprinting and id construction free of clock reads.
    """
    if created_at.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")
    stamp = created_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{fingerprint_hex[:12]}-{stamp}"


def build_manifest(
    *,
    model: ModelInfo,
    engine: EngineInfo,
    quantization: str,
    hardware: HardwareInfo,
    workload: Mapping[str, object],
    loadgen: LoadgenInfo,
    created_at: datetime,
    notes: str | None = None,
) -> dict[str, object]:
    """Assemble the full manifest dict, including its computed fingerprint.

    ``workload`` is the dict produced by ``WorkloadSpec`` serialization (all
    spec fields); it is copied, not aliased. ``created_at`` must be
    timezone-aware and is recorded as UTC ISO 8601.
    """
    if created_at.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at.astimezone(UTC).isoformat(),
        "model": model.to_dict(),
        "engine": engine.to_dict(),
        "quantization": quantization,
        "hardware": hardware.to_dict(),
        "workload": dict(workload),
        "loadgen": loadgen.to_dict(),
        "notes": notes,
    }
    manifest["fingerprint"] = fingerprint(manifest)
    return manifest
