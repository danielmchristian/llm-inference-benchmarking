"""Tests for llmbench.workloads: determinism, cycling, shape, and corpus integrity."""

import contextlib
import dataclasses
import hashlib
from importlib import resources
from pathlib import Path

from llmbench.workloads import WORKLOADS, WorkloadSpec, build_messages, to_manifest_dict

CHAT_SHORT = WORKLOADS["chat-short"]


def _corpus_lines() -> list[str]:
    text = (resources.files("llmbench.data") / "chat_prompts.txt").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def test_registry_chat_short_spec() -> None:
    assert CHAT_SHORT.name == "chat-short"
    assert CHAT_SHORT.prompt_source == "chat_prompts.txt"
    assert CHAT_SHORT.max_output_tokens == 128
    assert CHAT_SHORT.temperature == 0.0
    assert CHAT_SHORT.ignore_eos is True
    assert CHAT_SHORT.seed == 1337
    assert "ignore_eos" in CHAT_SHORT.description


def test_build_messages_deterministic() -> None:
    a = build_messages(CHAT_SHORT, 50)
    b = build_messages(CHAT_SHORT, 50)
    assert a == b


def test_build_messages_seed_changes_output() -> None:
    other = dataclasses.replace(CHAT_SHORT, seed=CHAT_SHORT.seed + 1)
    assert build_messages(CHAT_SHORT, 50) != build_messages(other, 50)


def test_build_messages_cycles_beyond_corpus() -> None:
    corpus_size = len(_corpus_lines())
    msgs = build_messages(CHAT_SHORT, corpus_size + 10)
    assert len(msgs) == corpus_size + 10
    # Cycling repeats the shuffled order exactly.
    assert msgs[corpus_size] == msgs[0]
    assert msgs[corpus_size + 9] == msgs[9]


def test_build_messages_shape() -> None:
    msgs = build_messages(CHAT_SHORT, 5)
    assert len(msgs) == 5
    for msg_list in msgs:
        assert len(msg_list) == 1
        (msg,) = msg_list
        assert set(msg) == {"role", "content"}
        assert msg["role"] == "user"
        assert isinstance(msg["content"], str)
        assert msg["content"].strip()


def test_corpus_loads_from_package_resources(tmp_path: Path) -> None:
    # Run from an unrelated cwd to prove no filesystem-relative path is used.
    with contextlib.chdir(tmp_path):
        assert (resources.files("llmbench.data") / "chat_prompts.txt").is_file()
        assert build_messages(CHAT_SHORT, 3)


def test_corpus_has_64_nonempty_unique_lines() -> None:
    lines = _corpus_lines()
    assert len(lines) >= 64
    assert len(set(lines)) == len(lines)


def test_to_manifest_dict_contains_every_field() -> None:
    manifest = to_manifest_dict(CHAT_SHORT)
    field_names = {f.name for f in dataclasses.fields(WorkloadSpec)}
    assert set(manifest) == field_names | {"prompt_source_sha256"}
    for name in field_names:
        assert manifest[name] == getattr(CHAT_SHORT, name)


def test_to_manifest_dict_pins_corpus_content_hash() -> None:
    """The manifest must hash the corpus BYTES: two corpora sharing a filename
    but differing in content must not produce identical manifests/fingerprints."""
    manifest = to_manifest_dict(CHAT_SHORT)
    raw = (resources.files("llmbench.data") / CHAT_SHORT.prompt_source).read_bytes()
    assert manifest["prompt_source_sha256"] == hashlib.sha256(raw).hexdigest()
