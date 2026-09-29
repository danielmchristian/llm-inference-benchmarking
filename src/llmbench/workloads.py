"""Workload definitions: named, reproducible prompt sets for benchmark runs.

A workload pins every knob that affects request content (prompt corpus, output
length, sampling temperature, seed) so that two runs against different servers
send byte-identical request bodies.
"""

import hashlib
import random
from dataclasses import asdict, dataclass
from importlib import resources

__all__ = ["WORKLOADS", "WorkloadSpec", "build_messages", "to_manifest_dict"]


@dataclass(frozen=True)
class WorkloadSpec:
    """Immutable description of a benchmark workload."""

    name: str
    description: str
    prompt_source: str
    max_output_tokens: int
    temperature: float
    ignore_eos: bool
    seed: int


def to_manifest_dict(spec: WorkloadSpec) -> dict[str, object]:
    """Return every field of the spec, plus ``prompt_source_sha256``, for run manifests.

    The corpus content hash pins the exact prompt bytes into the manifest (and
    therefore the fingerprint); the filename alone would let two runs with
    different prompt content share an identical fingerprint.
    """
    result: dict[str, object] = asdict(spec)
    result["prompt_source_sha256"] = _corpus_sha256(spec.prompt_source)
    return result


def _corpus_sha256(prompt_source: str) -> str:
    """Return the sha256 hex digest of the bundled corpus file's raw bytes."""
    data = (resources.files("llmbench.data") / prompt_source).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _load_corpus(prompt_source: str) -> list[str]:
    """Load the bundled prompt corpus (one prompt per line) via package resources."""
    text = (resources.files("llmbench.data") / prompt_source).read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def build_messages(spec: WorkloadSpec, n: int) -> list[list[dict[str, str]]]:
    """Build `n` deterministic single-turn message lists for the workload.

    The corpus is shuffled with random.Random(spec.seed) and cycled when `n`
    exceeds the corpus size, so the same (spec, n) always yields byte-identical
    output.
    """
    corpus = _load_corpus(spec.prompt_source)
    random.Random(spec.seed).shuffle(corpus)
    return [[{"role": "user", "content": corpus[i % len(corpus)]}] for i in range(n)]


WORKLOADS: dict[str, WorkloadSpec] = {
    "chat-short": WorkloadSpec(
        name="chat-short",
        description=(
            "Short single-turn chat questions; output length is FIXED at "
            "max_output_tokens via ignore_eos (vLLM extension) so perf runs "
            "are comparable across configs."
        ),
        prompt_source="chat_prompts.txt",
        max_output_tokens=128,
        temperature=0.0,
        ignore_eos=True,
        seed=1337,
    ),
}
