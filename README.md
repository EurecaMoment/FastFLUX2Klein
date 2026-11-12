# FastFLUX2Klein

OpenAI Images-compatible inference service for Black Forest Labs' FLUX.2 Klein models. This repository is a fork of [black-forest-labs/flux2](https://github.com/black-forest-labs/flux2); the upstream source, documentation, assets and license remain available here as `UPSTREAM_README.md`, `src/`, `scripts/`, `docs/`, and the original license files.

The `service/` directory adds a batched FastAPI server, OpenAI SDK client, single- and multi-reference image editing, exact and approximate diffusion caches, performance benchmarks and tests. No model weights, API keys or generated outputs are included.

## Features

- `POST /v1/images/generations` and `POST /v1/images/edits` with the OpenAI Python SDK.
- One or multiple reference images, with bounded image size and memory-aware batching.
- BF16 inference, regional Transformer compilation, native SDPA, batched prompt encoding, prompt/reference caches and parallel output encoding.
- `diffusion_cache="none"|"exact"|"taylorseer"`. `exact` is pixel-identical to `none` in tested runs. TaylorSeer predicts late residual blocks and changes image details; it is opt-in and reports whether predictions were used.
- PNG, JPEG and WEBP output; square and non-square resolutions.

The service package and setup below were validated on Python 3.10, PyTorch 2.6.0+cu124 and an NVIDIA A100 80GB with a 40GB process budget. Other CUDA GPUs may need a matching PyTorch wheel and different batch/memory settings.

## Environment

Install a current NVIDIA driver compatible with CUDA 12.4, Git, Conda, and an NVIDIA GPU with enough free memory for the 9B model. The reference deployment used an A100 80GB shared with another workload and limited this service to 40GB. At least 40GB of free disk is recommended for model weights, compilation cache and generated files.

```bash
git clone https://github.com/EurecaMoment/FastFLUX2Klein.git
cd FastFLUX2Klein
cp .env.example .env
# Edit .env: set FLUX2_MODEL_PATH and FLUX2KLEIN_API_KEY.
bash service/run.sh install
```

`service/run.sh` creates the `flux2klein` Conda environment with Python 3.10 when it does not exist, then installs the pinned CUDA 12.4 PyTorch wheel and inference stack. Set `FLUX2KLEIN_CONDA_ENV` or `FLUX2KLEIN_CONDA_PREFIX` in `.env` to select another environment. The launcher calls `conda run`, so every command uses the selected environment without requiring shell-specific activation. For another CUDA version, edit the PyTorch wheel/index entries to match the installed driver and GPU before installing.

### Download the model

The 9B weights use the FLUX.2 model license and may require accepting its terms on Hugging Face before download. This repository does not redistribute weights.

```bash
bash service/run.sh hf auth login
mkdir -p "$HOME/models/FLUX.2-klein-9B"
bash service/run.sh hf download black-forest-labs/FLUX.2-klein-9B \
  --local-dir "$HOME/models/FLUX.2-klein-9B"
```

Set up local service configuration before installation:

```bash
chmod 600 .env
```

Edit `.env` and set an absolute `FLUX2_MODEL_PATH`, a long random `FLUX2KLEIN_API_KEY`, and `CUDA_VISIBLE_DEVICES` for the GPU assigned to this process. Do not commit `.env`. For example, use `openssl rand -hex 32` to create a key, then store the result in `.env`.

## Start

Start the server in the foreground with Bash; use Ctrl+C to stop it:

```bash
./service/run.sh serve
```

The same launcher works from a login shell, a regular terminal, or a user-created Bash process. It only needs Bash, a working Conda installation, and the selected Conda environment.

The first startup compiles and warms the supported batch shapes before marking `/health` ready. Subsequent starts reuse the local TorchInductor cache. Default address: `http://127.0.0.1:8036/v1`.

To make the service reachable from other machines, set `--host 0.0.0.0` and place it behind a trusted network or TLS reverse proxy. Keep the API key private; this service does not configure TLS itself.

## OpenAI Python SDK

```python
import base64
import os
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8036/v1",
    api_key=os.environ["FLUX2KLEIN_API_KEY"],
    timeout=600,
)

# Text-to-image
result = client.images.generate(
    model="flux2klein",
    prompt="A red ceramic teapot on a wooden table in soft daylight",
    size="512x512",
    response_format="b64_json",
    extra_body={"seed": 42, "diffusion_cache": "exact"},
)
Path("generated.png").write_bytes(base64.b64decode(result.data[0].b64_json))

# Single-reference edit
with open("input.png", "rb") as image:
    result = client.images.edit(
        model="flux2klein",
        image=image,
        prompt="Change the teapot to cobalt blue. Preserve its shape and the table.",
        size="512x512",
        extra_body={"seed": 60, "diffusion_cache": "exact"},
    )
Path("edited.png").write_bytes(base64.b64decode(result.data[0].b64_json))

# Multi-reference edit
with open("teapot.png", "rb") as first, open("robot.png", "rb") as second:
    result = client.images.edit(
        model="flux2klein",
        image=[first, second],
        prompt="Place the robot from image 2 next to the teapot from image 1.",
        size="512x512",
        extra_body={"seed": 62, "diffusion_cache": "exact"},
    )
Path("combined.png").write_bytes(base64.b64decode(result.data[0].b64_json))
```

## Cache controls

| Option | Default | Behavior |
|---|---:|---|
| `cache_prompt` | `true` | Reuses text features in a bounded CPU pinned-memory LRU. |
| `cache_reference` | `true` | Reuses deterministic VAE latents for identical reference images. |
| `diffusion_cache` | `exact` | `none` performs the regular calculation; `exact` memoizes invariant projections/position embeddings within a request; `taylorseer` additionally predicts selected late-block residuals. |
| `cache_blocks` | `12` | Number of trailing single-stream blocks selected for Taylor prediction, 1–24. |
| `cache_warmup_steps` | `2` | Full steps before prediction history is used. |
| `cache_interval` | `2` | Refresh interval for approximate prediction. The final denoising step is always full. |

Approximate example:

```python
extra_body={
    "seed": 42,
    "num_inference_steps": 4,
    "diffusion_cache": "taylorseer",
    "cache_blocks": 12,
}
```

With 4 steps and defaults, the service predicts selected blocks only during an intermediate step. `timings.predicted_blocks` and `timings.approximate` report actual use. A mode name alone does not guarantee that a prediction occurred.

## Service API and limits

- `POST /v1/images/generations`: OpenAI-shaped text-to-image request.
- `POST /v1/images/edits`: multipart or JSON Base64/data-URL reference-image editing.
- `GET /v1/models`: authenticated model list.
- `GET /v1/files/{id}.{png|jpeg|webp}`: authenticated download for URL responses.
- `GET /health`: readiness, active limits, caches, memory and startup-equivalence checks.

Requests support `n=1..8`, `num_inference_steps=1..50`, 16-pixel-aligned dimensions from 256 to 1536, and up to 1,048,576 output pixels. A request can include up to 8 PNG/JPEG/WEBP references, each up to 16 MiB, with a combined resized area up to 2MP. Masks/inpainting and dynamic LoRA management are not implemented; unsupported fields return an error instead of being ignored.

Default model-memory budget is 40 decimal GB. Initial batch limits are 16 for outputs up to 512x512 and 4 for larger outputs; available memory and reference context can reduce effective batch sizes. The GPU worker processes one compatible batch at a time; request queue capacity is not GPU parallelism.

## CLI, benchmark and tests

```bash
./service/run.sh generate --size 512x512 --prompt 'A red teapot' --output /tmp/teapot.png
./service/run.sh edit --image /tmp/teapot.png --prompt 'Make the teapot blue' --size 512x512 --output /tmp/blue.png
./service/run.sh edit --image /tmp/teapot.png --image /tmp/robot.png --prompt 'Place both objects together' --size 512x512 --output /tmp/together.png
./service/run.sh benchmark --concurrency 64 --requests 64 --size 512x512 --unique-prompts --output /tmp/bench.json
./service/run.sh test
./service/run.sh verify
```

The CPU contract suite tests the OpenAI-style HTTP schema without loading the model. The GPU integration suite requires a running local server and generates its own reference images. Benchmark prompts receive a per-run random namespace so cold-prompt results cannot silently hit an earlier run's cache.

## Reference deployment result

On an A100 80GB with a 40GB process budget, BF16 and 4 denoising steps, the validated deployment measured 2.38 images/s for 64 concurrent cached 512x512 requests and 2.14 images/s for 64 unique prompts. Eight concurrent 1024x1024 requests measured 0.69 images/s. These are one deployment's observations, not a hardware-wide guarantee. See `service/README.md` for the full report and measured tradeoffs.

## Licensing and attribution

This repository retains upstream FLUX.2 source and license files. The included 9B model weights are not redistributed and are subject to the separate FLUX.2 model license linked from the upstream model card. Check and accept the model terms before downloading or using the weights. See `LICENSE.md`, `model_licenses/`, and the [official FLUX.2 Klein 9B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B).
