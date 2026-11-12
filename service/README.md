# Service Package

The files in this directory provide a standalone OpenAI Images-compatible API around Hugging Face Diffusers `Flux2KleinPipeline`.

## Files

- `server.py`: authenticated FastAPI routes, OpenAI-shaped errors, request validation, uploads, adaptive batching and health metrics.
- `engine.py`: resident BF16 model loading, dynamic GPU batching, text/reference caching, generation/edit execution and memory-aware retry.
- `optimizations.py`: exact Qwen feature extraction, unused-weight parking, bounded host caches, exact invariant memoization and opt-in Taylor residual prediction.
- `native_attention.py`: the native PyTorch SDPA adapter used by the FLUX.2 Transformer.
- `client_flux2klein.py`: OpenAI SDK client for generation and one or several image references.
- `benchmark_v2.py`: HTTP throughput and latency benchmark with unique prompt namespaces.
- `test_contract.py`: CPU API contract tests using a deterministic engine stub.
- `verify_live.py`: real GPU generation, cache, and reference-edit integration checks.
- `run.sh`: portable local environment and command entry point.
- `requirements.txt`: the versions validated by the reference deployment.

## Setup

Use the setup and model access instructions in the repository root README. Create `.env` from `.env.example`, set the model directory and a private random API key, then install `service/requirements.txt` in the repository `.venv`.

Start in the foreground:

```bash
./service/run.sh serve
```

The server requires a local model directory and `FLUX2KLEIN_API_KEY`. It does not download weights at runtime. Compilation and warm-up happen before readiness. Stop the foreground server with Ctrl+C.

## Cache semantics

`none` runs the regular Transformer calculation. `exact` caches context and positional projections within a single inference; the tested outputs are bitwise equal to `none`. `taylorseer` predicts a configured suffix of single-stream residual blocks after warm-up and between refreshes. It changes image details, remains opt-in, and is reflected in `predicted_blocks` and `approximate` output metrics. Per-request prediction history is cleared at the end of the call.

Text features are cached in bounded pinned CPU memory by prompt. Reference image latents are keyed by normalized decoded image content. No generated image or response cache is used. The weights for unused Qwen output layers may be parked in CPU memory after startup equivalence checks; all parameters used for inference remain on the GPU and there is no per-request weight offload.

## Reference measurements

Validated on an NVIDIA A100 80GB shared with another model, a 40 decimal GB process budget, PyTorch 2.6.0+cu124, Diffusers 0.37.0, Transformers 4.57.3, BF16 and 4 steps:

| Load | Requests | Throughput | P50 | P95 | Prompt-cache hits | OOM splits |
|---|---:|---:|---:|---:|---:|---:|
| 512x512, cached prompt | 64 at concurrency 64 | 2.381 images/s | 16.02 s | 26.83 s | 64/64 | 0 |
| 512x512, unique prompts | 64 at concurrency 64 | 2.135 images/s | 15.57 s | 29.93 s | 0/64 | 0 |
| 1024x1024, cached prompt | 8 at concurrency 8 | 0.690 images/s | 5.93 s | 11.58 s | 8/8 | 0 |

Measurements are HTTP completion times from the server host, including queueing, output encoding and response transfer. They are observations for this shared GPU configuration, not performance guarantees. High image throughput can coincide with longer individual request latency.

At startup the encoder's optimized hidden features were bitwise equal to the full Qwen encoder. Parking unused text weights released 4.718GB of GPU parameter allocations in the reference deployment. CUDA Graph output was bitwise equal but only 2.15% faster, below the configured 3% acceptance threshold, so it was not retained. QKV fusion is disabled by default because it introduced numerical changes without a material measured gain.

The approximate TaylorSeer quality comparison in the reference run had pixel MAE 11.22/255 and PSNR 23.04dB against exact mode. Evaluate output quality on your prompts before using approximate mode.

## Limits

The API supports up to 8 reference images and a total resized reference area up to 2MP. Output dimensions are multiples of 16, from 256 to 1536 per side, up to 1MP total. The initial batch limits are 16 for outputs up to 512x512 and 4 for larger outputs; reference context and available VRAM can reduce them. The queue admits up to 256 images by default, but one GPU worker executes one compatible batch at a time.

Mask-based inpainting and dynamic LoRA management are not implemented. Unknown request parameters are rejected.

