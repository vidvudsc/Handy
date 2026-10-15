"""Handy MLX transcription sidecar.

Persistent process wrapping parakeet-mlx with 8-bit quantized weights.

Protocol (all little-endian, over stdio):
  request:  u64 byte length, then that many bytes of f32 PCM at 16 kHz mono
  response: one JSON line: {"text": str, "ms": float} or {"error": str}
  startup:  one JSON line {"ready": true} once the model is loaded

Logs go to stderr.
"""

import json
import math
import struct
import sys
import time

# Round request lengths up to this many seconds to bound MLX kernel shapes.
PAD_STEP_S = 2
# Common dictations are short. Warm every bucket through this duration so a
# first request at, say, 14 or 24 seconds never has to compile Metal kernels.
WARM_THROUGH_S = 30
# Longer dictations remain rare enough that only these coarse shapes are worth
# keeping resident at launch.
LONG_WARM_BUCKETS_S = (40, 60)
# Keep only a modest pool of reusable Metal buffers between dictations. The
# model's live weights are unaffected by this limit.
CACHE_LIMIT_BYTES = 1024 * 1024 * 1024


def log(msg: str) -> None:
    print(f"[mlx-sidecar] {msg}", file=sys.stderr, flush=True)


def respond(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def transcribe_payload(payload, np, mx, model, get_logmel) -> str:
    """Keep temporary MLX arrays scoped to one request for cache cleanup."""
    samples = np.frombuffer(payload, dtype=np.float32)
    pad_step = 16000 * PAD_STEP_S
    target = max(pad_step, math.ceil(len(samples) / pad_step) * pad_step)
    if target > len(samples):
        samples = np.pad(samples, (0, target - len(samples)))
    mel = get_logmel(mx.array(samples), model.preprocessor_config)
    results = model.generate(mel)
    return results[0].text.strip() if results else ""


def main() -> None:
    t0 = time.time()
    log("loading parakeet-mlx model...")
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.audio import get_logmel

    model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    nn.quantize(
        model,
        bits=8,
        group_size=64,
        class_predicate=lambda p, m: isinstance(m, nn.Linear)
        and m.weight.shape[-1] % 64 == 0,
    )
    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    # MLX JIT-compiles Metal kernels per input shape, so real dictations (each
    # a unique length) would recompile every time. We round every request up to
    # the next PAD_STEP_S seconds, bounding the set of shapes while keeping the
    # padding waste small (coarse buckets over-pad and cost real compute). The
    # common short lengths are pre-warmed here so the first dictations are
    # instant. Keep this exactly aligned with the rounding rule below: gaps in
    # the warm list cause otherwise ordinary requests to pay a compile stall.
    warm_buckets = tuple(range(PAD_STEP_S, WARM_THROUGH_S + 1, PAD_STEP_S))
    for seconds in (*warm_buckets, *LONG_WARM_BUCKETS_S):
        warm = mx.zeros(16000 * seconds)
        model.generate(get_logmel(warm, model.preprocessor_config))
    del warm
    mx.clear_cache()
    log(f"model ready in {time.time() - t0:.1f}s")
    respond({"ready": True})

    stdin = sys.stdin.buffer
    while True:
        header = stdin.read(8)
        if len(header) < 8:
            log("stdin closed, exiting")
            return
        (nbytes,) = struct.unpack("<Q", header)
        payload = stdin.read(nbytes)
        if len(payload) < nbytes:
            log("truncated payload, exiting")
            return
        try:
            t0 = time.time()
            text = transcribe_payload(payload, np, mx, model, get_logmel)
            respond({"text": text, "ms": (time.time() - t0) * 1000})
        except Exception as e:  # noqa: BLE001 - report and keep serving
            log(f"transcription error: {e}")
            respond({"error": str(e)})
        finally:
            mx.clear_cache()


if __name__ == "__main__":
    main()
