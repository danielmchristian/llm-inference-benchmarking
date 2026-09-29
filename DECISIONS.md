# Decision log

Append-only and numbered, newest at the bottom. Each entry records the
decision, the rationale, and what was rejected — so every benchmark choice can
be defended later. Entries are never edited after the fact; a reversal gets a
new entry that supersedes the old one.

---

## 0001 — Benchmark GPU: RTX 4090 24GB (community cloud)

- **Date:** 2026-09-28 · **Status:** accepted
- **Decision:** All single-GPU benchmarks run on a rented RTX 4090 24GB
  (RunPod/Vast community cloud, ~$0.35–0.70/hr).
- **Rationale:** Cheapest Ada Lovelace card. FP8 is a hardware feature, not
  just a file format: only Ada (SM89) and Hopper (SM90) have FP8 tensor cores,
  and Phase 3 compares FP16/FP8/AWQ/GPTQ — on older silicon the FP8 column
  would run fallback kernels and systematically undersell FP8. 24GB fits an 8B
  model in FP16 with modest KV-cache headroom; hitting that ceiling under load
  is itself a realistic, publishable finding.
- **Rejected:** A100 (Ampere — no native FP8 compute), H100 (burns the budget
  on rental hours instead of coverage), RTX 3090 (no FP8), L40S (~2× the cost
  for headroom not needed yet).

## 0002 — Benchmark model: meta-llama/Llama-3.1-8B-Instruct

- **Date:** 2026-09-28 · **Status:** accepted
- **Decision:** The benchmarked model is `meta-llama/Llama-3.1-8B-Instruct`,
  pinned to an exact revision hash in every run manifest (first pinned in
  Phase 1).
- **Rationale:** The de facto standard in inference benchmarks — vLLM's and
  SGLang's own published numbers use it, so harness results can be
  sanity-checked against known baselines before we trust them. Official FP8
  and well-maintained AWQ/GPTQ quantized variants exist.
- **Caveats:** Distributed under the Llama 3.1 Community License (not
  OSI-permissive) and gated on Hugging Face. An `HF_TOKEN` is needed at
  GPU-run time only; it lives in a gitignored `.env` and never appears in CI
  or the repo.
- **Rejected:** Qwen3-8B (Apache-2.0 and ungated, but fewer third-party
  baselines to cross-check, and its hybrid thinking mode must be pinned off
  for comparable output lengths), Mistral-7B-Instruct-v0.3 (2023-generation;
  results would feel dated).

## 0003 — Baseline serving engine: vLLM

- **Date:** 2026-09-28 · **Status:** accepted
- **Decision:** Phase 1 benchmarks vLLM only. SGLang joins in Phase 2; the
  third engine is an open question (see below).
- **Rationale:** Largest community, best docs, cleanest OpenAI-compatible
  server, and the most published reference numbers — in Phase 1, when nothing
  exists to compare against, that is how we tell a harness bug from a real
  result.
- **Rejected:** SGLang first (fewer public baselines), starting multi-engine
  (scope creep before the harness is proven).

## 0004 — Total GPU budget ceiling: under $50

- **Date:** 2026-09-28 · **Status:** accepted
- **Decision:** The whole project spends less than $50 on GPU rental
  (~70–140 hours of 4090 time).
- **Rationale:** Forces the discipline that makes the results trustworthy:
  every phase is debugged GPU-free against a local stub server first, every
  phase gets a GPU-hour estimate before any instance is rented, and workloads
  are sized to answer the question with the fewest tokens that still give
  honest percentiles.

## 0005 — License: Apache-2.0

- **Date:** 2026-09-28 · **Status:** accepted
- **Decision:** The code is licensed Apache-2.0.
- **Rationale:** Permissive, includes an explicit patent grant, and is the
  standard for infrastructure tooling — no adoption friction.
- **Rejected:** MIT (no patent grant), AGPL (chills adoption of a tool meant
  to be reused and cited).

## 0006 — Dependency management: uv with a committed lockfile

- **Date:** 2026-09-28 · **Status:** accepted
- **Decision:** uv manages the environment; `uv.lock` is committed; CI and the
  quickstart use `uv sync --locked`, so a stale lockfile fails the build.
- **Rationale:** Locked installs are part of the reproducibility story — a
  reviewer cloning the repo gets byte-identical dependency versions.
- **Rejected:** poetry (slower, a second config dialect), pip +
  requirements.txt (no cross-version lock or hash checking by default).

## 0007 — Task runner: Makefile

- **Date:** 2026-09-28 · **Status:** accepted
- **Decision:** A Makefile holds the dev commands (`setup`, `lint`, `format`,
  `typecheck`, `test`, `check`). `check` mirrors CI exactly; there is
  deliberately no separate `ci` target — two names for the same thing drift.
- **Rationale:** make is universally installed and the targets are trivial;
  `just` is not installed on the dev machine.
- **Rejected:** justfile (extra install), bare `scripts/` (no discoverability).

## 0008 — Layout and naming: src layout, `llm-bench` / `llmbench`

- **Date:** 2026-09-28 · **Status:** accepted
- **Decision:** `src/` layout. Distribution name `llm-bench`, import package
  `llmbench`, future CLI `llm-bench` (entry point lands in Phase 1 with the
  first real module). A `py.typed` marker ships in the wheel.
- **Rationale:** src layout forces tests to run against the installed package
  rather than the working directory — packaging bugs surface in CI, not in a
  user's clone. The dist/import name split matches the brand and costs two
  lines of hatch config.
- **Rejected:** flat layout (masks packaging bugs), adding the CLI entry point
  now (it would point at a module that doesn't exist).

## 0009 — Lint and types: ruff (lint + format) + mypy strict

- **Date:** 2026-09-28 · **Status:** accepted
- **Decision:** ruff for linting and formatting; mypy `strict = true` from day
  one, checking against Python 3.11 (the support floor). Ruff rule sets:
  E/W/F/I/UP/B/C4/SIM/PT/RUF.
- **Rationale:** Strict typing is free on an empty package and expensive to
  retrofit. Excluded rule sets, with reasons: D (docstring ceremony at this
  stage), ANN (mypy strict already enforces annotations), S/bandit
  (client-side benchmark tool, low security surface — revisit if subprocess or
  Docker control lands), PL (noisy, low marginal value). Coverage measurement
  deferred to Phase 1 when there is logic worth measuring.
- **Rejected:** black (redundant with ruff format), pyright (mypy is the
  boring default for OSS Python).

## 0010 — Architecture: the harness is a client

- **Date:** 2026-09-28 · **Status:** accepted
- **Decision:** The harness drives OpenAI-compatible HTTP endpoints and never
  links against serving-engine internals. Engines run in Docker on the GPU
  box, pinned by image digest. Reserved module map (each created only in the
  phase that implements it): `llmbench.loadgen`, `.manifest`, `.metrics`,
  `.workloads`, `.stubserver`, `.engines/` (vllm, sglang, …), `.analysis`,
  `.cli`.
- **Rationale:** Engine-agnostic by construction (adding an engine is adding a
  Docker invocation, not a dependency), harness deps stay light, and the whole
  harness is testable GPU-free against a local stub server.

## 0011 — Results convention: `runs/` ignored, `results/` committed

- **Date:** 2026-09-28 · **Status:** accepted
- **Decision:** `/runs/` holds raw per-run output (large, machine-specific)
  and is gitignored. `/results/` holds small curated JSONL plus their
  SHA-256-fingerprinted manifests and is committed as the reproducibility
  artifact. Both directories are created in Phase 1 when content exists.
- **Rationale:** Published numbers must be re-analyzable without re-running,
  which means the data behind every table and plot lives in the repo — but not
  every debugging run does.

## 0012 — CI: GitHub Actions, no GPU ever

- **Date:** 2026-09-28 · **Status:** accepted
- **Decision:** Three jobs — lint and typecheck once on Python 3.12, pytest on
  a 3.11/3.12 matrix. All jobs install with `uv sync --locked`. CI never
  touches a GPU or Hugging Face; benchmark numbers come from documented manual
  runs published under `results/`.
- **Rationale:** ruff and mypy don't execute the code (ruff targets the
  version in `requires-python`; mypy pins `python_version = "3.11"` in
  config), so running them per-version adds minutes and no information. GPU
  runs in CI would be expensive, unreproducible on shared runners, and a
  secrets risk.

## 0013 — Phase 1 runtime dependencies and CLI shape

- **Date:** 2026-09-29 · **Status:** accepted
- **Decision:** Runtime deps: httpx (async client), starlette + uvicorn (the
  stub server ships in the package as a first-class feature), numpy
  (statistics), matplotlib (plots, imported lazily so `run` never pays its
  startup cost). CLI is argparse with subcommands `run`, `stub`, `analyze`,
  `envinfo`, entry point `llm-bench`.
- **Rationale:** The stub is not a test helper — it is how anyone (including
  reviewers) verifies the harness measures correctly, so it ships. argparse is
  zero-dependency and boring.
- **Rejected:** click/typer (a dependency for no capability we need),
  splitting deps into optional extras (premature for five small wheels).

## 0014 — mypy checks under a 3.12 target, not the 3.11 floor

- **Date:** 2026-09-29 · **Status:** accepted (amends 0009)
- **Decision:** `[tool.mypy] python_version = "3.12"` while
  `requires-python = ">=3.11"` stands. Runtime 3.11 compatibility is enforced
  by the CI test matrix, which actually executes the code on 3.11.
- **Rationale:** numpy >= 2.5 ships PEP 695 `type` aliases in its stubs, which
  mypy refuses to parse under a 3.11 target — any module importing numpy
  failed typechecking.
- **Rejected:** `Any`-typed shadow stubs or dynamic `import_module` imports
  (defeats strict typing exactly where the math lives), pinning numpy < 2.5
  (holds back a runtime dep for the sake of tooling).

## 0015 — GPU rental provider: RunPod community cloud

- **Date:** 2026-09-29 · **Status:** accepted
- **Decision:** GPU runs rent RunPod community-cloud RTX 4090s, deployed from
  the official `vllm/vllm-openai` image pinned by digest, all traffic over
  SSH, nothing exposed publicly.
- **Rationale:** Reliable 4090 availability, per-second billing, clean
  Docker-image pods; the official vLLM image keeps the engine install out of
  our hands entirely (one less thing a skeptic has to trust).
- **Rejected for now:** Vast.ai (often cheaper, less consistent; the fallback
  if RunPod pricing or availability turns bad).

---

## Open questions

- **Third engine for Phase 2:** TGI vs TensorRT-LLM vs llama.cpp-server.
  Decide when Phase 2 starts, based on what runs credibly on a single 4090.
- **Committing curated results JSONL:** leaning yes (it is the reproducibility
  artifact); confirm once real file sizes are known.
- **GitHub repo name and visibility timing:** "llm-bench" likely collides with
  existing projects on GitHub/PyPI — check before pushing; the push itself is
  deferred until the user is ready.
