#!/usr/bin/env python3
"""Memory plan for a single-node local language model deployment.

Reports the device memory a serving configuration needs, separated into three
terms: weights, key/value cache, and the runtime overhead outside both of them.

The third term is supplied by the operator and must exclude the weights and the
cache, because this tool adds those separately. A total resident-memory reading
taken after the model has loaded is NOT the reserve: it already contains the
weights, and on a runtime that preallocates its cache it contains that too.
Using it double-counts and badly over-sizes the hardware.

The tool performs arithmetic on declared values. It does not estimate the
reserve, detect hardware, read a model repository or predict throughput.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Dict, Optional, Tuple

# Bytes per stored key/value element. Values are the element widths the
# serving runtimes expose as cache types, not a claim about accuracy.
CACHE_DTYPE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "q8_0": 1.0625,
    "q4_0": 0.5625,
}

# Attention families whose cache size this formula describes. Multi-head
# latent attention, state-space and hybrid recurrent layers store something
# other than one key and one value per head per token.
SUPPORTED_ATTENTION = {"mha", "gqa", "mqa"}

REQUIRED_SPEC_FIELDS = (
    "name",
    "attention",
    "num_hidden_layers",
    "num_key_value_heads",
    "head_dim",
)

GIB = 1024 ** 3


class SpecError(ValueError):
    """The supplied model specification cannot be used for planning."""


def _positive_int(spec: Dict[str, Any], field: str) -> int:
    value = spec.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError("%s must be an integer" % field)
    if value <= 0:
        raise SpecError("%s must be greater than zero" % field)
    return value


def weight_bytes(spec: Dict[str, Any]) -> Tuple[int, bool]:
    """Return the weight footprint and whether it was estimated.

    The measured total of the weight files on disk is the better input, because
    a quantised file mixes element widths between tensors and a parameter count
    multiplied by a nominal bits-per-weight figure is an estimate, reported as
    one. File size is still a planning proxy rather than a measurement of
    device-resident weights: padding, alignment, any tensor the runtime keeps in
    host memory and any partial offload all move the real figure.
    """
    measured = spec.get("weight_file_bytes")
    if measured is not None:
        if isinstance(measured, bool) or not isinstance(measured, int):
            raise SpecError("weight_file_bytes must be an integer")
        if measured <= 0:
            raise SpecError("weight_file_bytes must be greater than zero")
        return measured, False

    params = spec.get("parameter_count")
    bits = spec.get("bits_per_weight")
    if params is None or bits is None:
        raise SpecError(
            "supply weight_file_bytes, or both parameter_count and bits_per_weight"
        )
    if isinstance(params, bool) or not isinstance(params, int) or params <= 0:
        raise SpecError("parameter_count must be a positive integer")
    if isinstance(bits, bool) or not isinstance(bits, (int, float)):
        raise SpecError("bits_per_weight must be a positive number")
    if not math.isfinite(float(bits)) or bits <= 0:
        raise SpecError("bits_per_weight must be a finite positive number")
    return int(math.ceil(params * float(bits) / 8.0)), True


def kv_bytes_per_token(spec: Dict[str, Any], cache_dtype: str) -> int:
    """Bytes of key/value cache one sequence adds for one token."""
    attention = spec.get("attention")
    if not isinstance(attention, str) or attention.lower() not in SUPPORTED_ATTENTION:
        raise SpecError(
            "attention must be one of %s; other families store a different "
            "cache shape and need their own measurement"
            % ", ".join(sorted(SUPPORTED_ATTENTION))
        )
    if cache_dtype not in CACHE_DTYPE_BYTES:
        raise SpecError("unknown cache dtype: %s" % cache_dtype)

    layers = _positive_int(spec, "num_hidden_layers")
    kv_heads = _positive_int(spec, "num_key_value_heads")
    head_dim = _positive_int(spec, "head_dim")
    element = CACHE_DTYPE_BYTES[cache_dtype]
    # One key and one value entry per key/value head, per layer, per token.
    return int(math.ceil(2 * layers * kv_heads * head_dim * element))


def validate_spec(spec: Dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        raise SpecError("model specification must be a JSON object")
    missing = [f for f in REQUIRED_SPEC_FIELDS if f not in spec]
    if missing:
        raise SpecError("missing required field(s): %s" % ", ".join(missing))
    if not isinstance(spec["name"], str) or not spec["name"].strip():
        raise SpecError("name must be a non-empty string")


def plan(
    spec: Dict[str, Any],
    device_bytes: int,
    reserve_bytes: int,
    context_tokens: int,
    concurrency: int,
    cache_dtype: str = "f16",
) -> Dict[str, Any]:
    """Build a memory plan for one serving configuration."""
    validate_spec(spec)
    if device_bytes <= 0:
        raise SpecError("device_bytes must be greater than zero")
    if reserve_bytes < 0:
        raise SpecError("reserve_bytes cannot be negative")
    if context_tokens <= 0:
        raise SpecError("context_tokens must be greater than zero")
    if concurrency <= 0:
        raise SpecError("concurrency must be greater than zero")

    weights, estimated = weight_bytes(spec)
    per_token = kv_bytes_per_token(spec, cache_dtype)
    per_sequence = per_token * context_tokens
    cache_total = per_sequence * concurrency
    required = weights + cache_total + reserve_bytes
    headroom = device_bytes - required

    remaining_for_cache = device_bytes - weights - reserve_bytes
    if remaining_for_cache < 0:
        remaining_for_cache = 0
    max_concurrency = remaining_for_cache // per_sequence if per_sequence else 0
    max_context = (
        remaining_for_cache // (per_token * concurrency) if per_token else 0
    )

    return {
        "model": spec["name"],
        "cache_dtype": cache_dtype,
        "requested": {
            "context_tokens": context_tokens,
            "concurrency": concurrency,
        },
        "bytes": {
            "weights": weights,
            "kv_cache_per_token_per_sequence": per_token,
            "kv_cache_per_sequence": per_sequence,
            "kv_cache_total": cache_total,
            "operator_reserve": reserve_bytes,
            "required": required,
            "device": device_bytes,
            "headroom": headroom,
        },
        "limits_at_this_reserve": {
            "max_concurrency_at_requested_context": int(max_concurrency),
            "max_context_at_requested_concurrency": int(max_context),
        },
        "fits": headroom >= 0,
        "weights_estimated": estimated,
        "notes": _notes(estimated, headroom, device_bytes),
    }


def _notes(estimated: bool, headroom: int, device_bytes: int) -> list:
    notes = []
    if estimated:
        notes.append(
            "Weight size was estimated from a parameter count. Replace it with "
            "the measured total of the weight files before committing hardware."
        )
    if headroom < 0:
        notes.append("This configuration does not fit on the declared device.")
    elif headroom < device_bytes * 0.05:
        notes.append(
            "Headroom is under five per cent of device memory. Fragmentation "
            "and a longer prompt batch can still exhaust it."
        )
    notes.append(
        "The reserve must cover only what this arithmetic does not model: "
        "activations, the CUDA context, graph and compute buffers, and "
        "allocator fragmentation. It must exclude the weights and the key/value "
        "cache, which are added separately above. Do not pass a total resident "
        "device-memory reading: that already includes the weights, and on a "
        "runtime that preallocates its cache it includes that too."
    )
    notes.append(
        "Obtain it from the runtime's own load-time buffer diagnostics rather "
        "than a single idle reading, then validate it under the intended prompt "
        "lengths, batch sizes and concurrency, because activation memory moves "
        "with the workload."
    )
    return notes


def _parse_size(text: str) -> int:
    """Accept a byte count, or a value suffixed with GiB, GB, MiB or MB."""
    value = text.strip().lower()
    multipliers = (
        ("gib", 1024 ** 3),
        ("mib", 1024 ** 2),
        ("kib", 1024),
        ("gb", 1000 ** 3),
        ("mb", 1000 ** 2),
        ("kb", 1000),
        ("g", 1024 ** 3),
        ("m", 1024 ** 2),
    )
    for suffix, factor in multipliers:
        if value.endswith(suffix):
            number = float(value[: -len(suffix)].strip())
            # "inf" and "nan" parse as floats and would otherwise reach int()
            # and raise OverflowError or ValueError outside the documented
            # input-error contract.
            if not math.isfinite(number) or number < 0:
                raise ValueError("size must be a finite non-negative number: %s" % text)
            return int(round(number * factor))
    return int(value)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan device memory for one local serving configuration."
    )
    parser.add_argument("--spec", required=True, help="path to a model spec JSON file")
    parser.add_argument(
        "--device",
        required=True,
        help="usable device memory, for example 24GiB or 25769803776",
    )
    parser.add_argument(
        "--reserve",
        required=True,
        help="runtime overhead EXCLUDING weights and cache, for example 2GiB; "
             "not a total resident-memory reading",
    )
    parser.add_argument("--context", type=int, required=True, help="context tokens per sequence")
    parser.add_argument(
        "--concurrency", type=int, required=True, help="sequences resident at once"
    )
    parser.add_argument(
        "--cache-dtype",
        default="f16",
        choices=sorted(CACHE_DTYPE_BYTES),
        help="key/value cache element type (default: f16)",
    )
    args = parser.parse_args(argv)

    try:
        with open(args.spec, "r", encoding="utf-8") as handle:
            spec = json.load(handle)
        result = plan(
            spec,
            device_bytes=_parse_size(args.device),
            reserve_bytes=_parse_size(args.reserve),
            context_tokens=args.context,
            concurrency=args.concurrency,
            cache_dtype=args.cache_dtype,
        )
    except (OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["fits"] else 1


if __name__ == "__main__":
    sys.exit(main())
