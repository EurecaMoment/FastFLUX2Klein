"""GPU integration checks against a running FastFLUX2Klein service."""
import base64
import datetime
import io
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np
from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "verification"
OUT.mkdir(exist_ok=True)
key = os.environ["FLUX2KLEIN_API_KEY"]
base = os.environ.get("FLUX2KLEIN_BASE_URL", "http://127.0.0.1:8036").rstrip("/")
client = OpenAI(base_url=base + "/v1", api_key=key, timeout=900, max_retries=0)
session = httpx.Client(trust_env=False, timeout=10)
last_phase = None
for _ in range(180):
    try:
        health = session.get(base + "/health").json()
        if health.get("phase") != last_phase:
            last_phase = health.get("phase")
            print("PHASE", last_phase, flush=True)
        if health.get("error"):
            raise RuntimeError(health["error"])
        if health.get("ready"):
            break
    except (httpx.TransportError, ValueError):
        pass
    time.sleep(5)
else:
    raise RuntimeError("Service did not become ready")

report = {"tested_at": datetime.datetime.now().isoformat(), "health_before": health,
          "checks": [], "calls": []}


def check(name, condition, **details):
    row = {"check": name, "passed": bool(condition), **details}
    report["checks"].append(row)
    print(json.dumps(row), flush=True)
    if not condition:
        raise AssertionError(name)


def invoke(name, prompt, size="512x512", mode="exact", seed=42, refs=None, n=1):
    params = {"model": "flux2klein", "prompt": prompt, "size": size, "n": n,
              "response_format": "b64_json",
              "extra_body": {"seed": seed, "num_inference_steps": 4, "diffusion_cache": mode}}
    started = time.perf_counter()
    if refs is not None:
        params["image"] = [(f"ref-{i}.png", data, "image/png") for i, data in enumerate(refs)]
        response = client.images.edit(**params)
    else:
        response = client.images.generate(**params)
    arrays = []
    rows = []
    for index, item in enumerate(response.data):
        data = base64.b64decode(item.b64_json)
        (OUT / f"{name}-{index + 1}.png").write_bytes(data)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        arrays.append(np.asarray(image))
        rows.append({"dimensions": image.size, "timings": item.model_extra.get("timings"),
                     "bytes": len(data)})
    call = {"name": name, "seconds": time.perf_counter() - started, "rows": rows}
    report["calls"].append(call)
    print(json.dumps(call), flush=True)
    (OUT / "verification.json").write_text(json.dumps(report, indent=2))
    return arrays, rows


prompt = "A red ceramic teapot on a wooden table in soft daylight"
check("startup_text_features_identical", health["startup_checks"]["early_text_features_bitwise_equal"])
none, _ = invoke("none", prompt, mode="none")
exact, _ = invoke("exact", prompt, mode="exact")
check("exact_matches_none", bool(np.array_equal(none[0], exact[0])))
approximate, rows = invoke("taylorseer", prompt, mode="taylorseer")
check("prediction_exercised", rows[0]["timings"]["predicted_blocks"] > 0)
delta = np.abs(exact[0].astype(np.float32) - approximate[0].astype(np.float32))
report["taylorseer_comparison"] = {
    "pixel_mae": float(delta.mean()), "pixel_max": float(delta.max()),
    "psnr": float(10 * np.log10(255**2 / (np.mean(delta**2) + 1e-12))),
}
check("approximate_image_nonempty", float(approximate[0].std()) > 10,
      **report["taylorseer_comparison"])
again, _ = invoke("none_after_approx", prompt, mode="none")
check("cache_state_does_not_leak", bool(np.array_equal(none[0], again[0])))

source = (OUT / "exact-1.png").read_bytes()
old = exact[0]
edit_prompt = "Change the red ceramic teapot to cobalt blue. Preserve its shape and the wooden table."
single, rows = invoke("single_edit", edit_prompt, seed=60, refs=[source])
check("single_edit_real_reference",
      rows[0]["timings"]["reference_images"] == 1 and single[0].shape == (512, 512, 3))
check("single_edit_changes_source", not np.array_equal(single[0], old))
_, rows = invoke("single_edit_cached", edit_prompt, seed=61, refs=[source])
check("reference_cache_hit", rows[0]["timings"]["reference_cache_hits"] == 1)
invoke("robot_reference", "A small friendly robot sitting and reading a book in a cozy library.", seed=43)
second = (OUT / "robot_reference-1.png").read_bytes()
multiple, rows = invoke(
    "multi_edit",
    "Place the robot from image 2 next to the ceramic teapot from image 1 on the same wooden table. Preserve both objects.",
    seed=62, refs=[source, second])
check("multi_edit_real_references",
      rows[0]["timings"]["reference_images"] == 2 and multiple[0].shape == (512, 512, 3))
outputs, rows = invoke("single_edit_n2", edit_prompt, seed=63, refs=[source], n=2)
check("edit_two_distinct_outputs", len(outputs) == 2 and not np.array_equal(outputs[0], outputs[1]))
_, rows = invoke("multi_edit_taylorseer", "Make a charming illustration featuring the teapot and robot together.",
                 seed=64, refs=[source, second], mode="taylorseer")
check("multi_edit_prediction", rows[0]["timings"]["predicted_blocks"] > 0)
invoke("generation_1024", "A mountain lake at sunrise with pine trees and reflections.",
       size="1024x1024", seed=44)

report["health_after"] = session.get(base + "/health").json()
check("no_oom_split", report["health_after"]["oom_splits"] == 0)
check("service_healthy_after_tests",
      report["health_after"]["ready"] and not report["health_after"]["error"])
(OUT / "verification.json").write_text(json.dumps(report, indent=2))
print("VERIFICATION_COMPLETE", sum(row["passed"] for row in report["checks"]), flush=True)
