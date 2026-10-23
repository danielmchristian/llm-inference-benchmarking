# Phase 1 runbook: single GPU run on RunPod

The exact procedure for the one Phase-1 GPU measurement: Llama-3.1-8B-Instruct
under vLLM on a rented RTX 4090. Everything else in Phase 1 is provable
GPU-free against the stub; this run produces the one real dataset. Follow the
steps in order; the termination checklist at the end is not optional.

Security posture: **nothing is exposed publicly**. Do not add HTTP ports to
the pod. All traffic — the OpenAI endpoint, file transfer — goes over SSH.

## 1. Create the pod

- RunPod → Community Cloud → **RTX 4090** (24 GB), 1x GPU.
- Container image: **resolve the digest first, pin at creation.** `latest` is
  a moving tag; handing RunPod a digest-pinned reference means the pod provably
  runs the exact image the manifest records — nothing to read back after the
  fact. From your workstation (no docker daemon or image pull needed):

  ```bash
  curl -s "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/latest" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['digest'])"
  ```

  Note the digest and the release tag that shares it (check the repo's tags
  page — e.g. on 2026-10-23, `latest` == `v0.32.0`). Set the pod's container
  image to:

  ```
  vllm/vllm-openai@sha256:<DIGEST>
  ```

  The same digest goes into `--image-digest` below, and the release tag into
  `--image`. If RunPod's image field rejects the `@sha256:` form, fall back to
  the release tag (`vllm/vllm-openai:v0.32.0`) and confirm the resolved digest
  in the pod details page before trusting the run.
- **Override the container start command** (RunPod pod config → "Container
  Start Command") with `bash -c "sleep infinity"`. The image's default
  entrypoint starts an API server immediately with no model configured; the
  override keeps the container idle so step 2 launches vLLM under your
  control, in a terminal where you can read its startup log.
- Environment variable: `HF_TOKEN=<your Hugging Face token>` (the Llama 3.1
  repo is gated; accept the license on the HF model page first).
- Disk: **60 GB** container/volume disk (model weights ~16 GB in fp16, plus
  HF cache and image layers; the default 20 GB will not fit comfortably).
- Enable SSH access (RunPod injects your public key). No exposed HTTP ports.

## 2. Start vLLM (tmux pane 1)

SSH in, start tmux (`tmux new -s bench` — if the image lacks tmux, install it
first: `apt-get update && apt-get install -y tmux`; plain multiple SSH
sessions also work), and launch the server:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096
```

Phase-1 policy: **engine defaults except the flags listed here**, so results
reflect vLLM's out-of-the-box behavior rather than accidental tuning. Every
non-default flag and its justification:

| Flag | Value | Why it is set |
|---|---|---|
| `--max-model-len` | `4096` | Llama 3.1's native 128k context makes vLLM budget KV-cache for 128k-token sequences; on a 24 GB card that either aborts at startup or strangles batch capacity. The chat-short workload needs well under 4096 tokens, so 4096 bounds KV-cache allocation without constraining the workload. |

Everything else (port 8000, dtype auto → bf16, scheduling, batching) is left
at engine defaults, and `serving_args` in the manifest records exactly the
flags above — nothing more, so the manifest is a complete description of the
launch.

**Record the model revision.** vLLM resolves `meta-llama/Llama-3.1-8B-Instruct`
to a specific HF commit. Get the hash from the download cache once the server
is loading (the snapshot directory name is the commit hash):

```bash
ls ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/
```

Write it down; it goes into `--model-revision` below. Wait for
`Application startup complete` in the vLLM log, then confirm:

```bash
curl -s http://127.0.0.1:8000/v1/models | head -c 300
```

## 3. Install the harness (tmux pane 2)

From your workstation, copy the repo to the pod (adjust host/port to the SSH
command RunPod shows):

```bash
rsync -az --exclude .venv --exclude runs \
    -e "ssh -p <POD_SSH_PORT>" \
    . root@<POD_IP>:/root/llm-bench
```

On the pod:

```bash
cd /root/llm-bench
pip install .
llm-bench --help          # sanity check
llm-bench envinfo > hw.json
cat hw.json               # expect RTX 4090, driver, CUDA version, ~24564 MiB
```

## 4. GPU sampling (tmux pane 3)

Start a 1 Hz utilization/memory log **before** the run and leave it running:

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1 > gpu.csv
```

This is the evidence that the GPU was actually saturated during measurement
(and idle before it). Stop it with Ctrl-C after the run.

## 5. The benchmark run (tmux pane 2)

```bash
cd /root/llm-bench
llm-bench run \
  --base-url http://127.0.0.1:8000 \
  --workload chat-short \
  --concurrency 8 \
  --num-requests 256 \
  --warmup 16 \
  --repeats 3 \
  --timeout-s 120 \
  --out runs \
  --model-revision <HF_COMMIT_HASH_FROM_STEP_2> \
  --engine-name vllm \
  --quantization fp16 \
  --image vllm/vllm-openai:<RELEASE_TAG> \
  --image-digest <DIGEST_FROM_STEP_1> \
  --serving-arg=--max-model-len \
  --serving-arg=4096 \
  --hardware-json hw.json
```

Notes:

- Model id and engine version are auto-detected from `/v1/models` and
  `/version`; `--serving-arg` values starting with `-` must use the
  `--serving-arg=<value>` form, one per token, mirroring the `vllm serve`
  command exactly.
- The command prints the fingerprint, the run directory, one summary line per
  rep, and the cross-rep table. Expect roughly 1–2 minutes per rep at this
  workload on a 4090.
- Sanity-check the output before leaving the pod: `error_rate` must be 0 and
  every rep line should show `ok=256`. Then:

```bash
llm-bench analyze runs/<RUN_ID> --plot runs/<RUN_ID>/latency.png
```

## 6. Copy results home

From your workstation:

```bash
scp -P <POD_SSH_PORT> -r root@<POD_IP>:/root/llm-bench/runs ./runs-phase1
scp -P <POD_SSH_PORT> root@<POD_IP>:/root/llm-bench/gpu.csv ./runs-phase1/
scp -P <POD_SSH_PORT> root@<POD_IP>:/root/llm-bench/hw.json ./runs-phase1/
```

Verify locally before touching the pod again:

```bash
ls runs-phase1/*/manifest.json runs-phase1/*/rep*.jsonl
llm-bench analyze runs-phase1/<RUN_ID>
```

## 7. TERMINATE THE POD — checklist

Community-cloud pods bill until terminated, not until idle. In order:

- [ ] `runs/`, `gpu.csv`, and `hw.json` copied home and verified readable
      (`llm-bench analyze` works locally).
- [ ] Image digest, model revision hash, and the exact `vllm serve` command
      recorded (they are in the manifest — confirm by reading it).
- [ ] **Terminate** (not stop) the pod in the RunPod UI. A stopped pod keeps
      billing for storage.
- [ ] RunPod dashboard shows zero running pods and zero storage volumes.
- [ ] If the HF token was ever pasted anywhere other than the pod env var
      (shell history, a file in the repo), rotate it at
      huggingface.co/settings/tokens.

## Expected cost

RTX 4090 community-cloud rates are roughly $0.40–0.70/hr (check the current
listing).

| Step | Time | Cost at $0.40–0.70/hr |
|---|---|---|
| Pod boot + image pull | ~5 min | $0.03–0.06 |
| Model download + vLLM startup | ~5–10 min | $0.03–0.12 |
| Harness install + envinfo | ~2 min | $0.01–0.02 |
| Benchmark run (3 reps) + analyze | ~5 min | $0.03–0.06 |
| Copy results + terminate | ~3 min | $0.02–0.04 |
| **Total** | **~20–25 min** | **~$0.15–0.30** |

Even with a false start or one repeat of the whole procedure, the run stays
well under $1.
