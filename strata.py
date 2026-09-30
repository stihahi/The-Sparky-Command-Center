"""Readers for a Strata server's JSON GET /metrics (github.com/Niko1221/Strata).

Pure functions: server.py fetches the body, these turn it into the same shapes the
Prometheus engines (vLLM, llama.cpp, SGLang) and the SSH-polled discrete hosts produce.
"""
import json

MIB = 1048576


def parse_metrics(body):
    """The decoded payload when `body` is Strata's /metrics, otherwise None."""
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    return payload if _has_strata_shape(payload) else None


def _has_strata_shape(payload):
    return isinstance(payload, dict) and "engine" in payload and "live" in payload


# ----------------------------------------------------------------------------
# Model card
# ----------------------------------------------------------------------------
def model_reading(metrics, previous_progress, now):
    """(model-card fields, prompt progress to pass back on the next poll)."""
    live = metrics["live"]
    progress = _prompt_progress(live, now)
    fields = {"engine": "Strata", "model": metrics["engine"].get("model"),
              "decode_tps": _decode_speed(live),
              "prefill_tps": _prefill_speed(previous_progress, progress),
              "ttft_ms": _latest_prompt_ms(metrics.get("requests")),
              "running": int(live.get("state") != "idle"), "waiting": int(live.get("queued") or 0)}
    return fields, progress


def _decode_speed(live):
    return float(live.get("tok_s") or 0.0) if live.get("state") == "generating" else 0.0


def _prompt_progress(live, now):
    if live.get("state") != "reading" or live.get("prompt_read") is None:
        return None
    return {"time": now, "read": live["prompt_read"], "total": live.get("prompt_total")}


def _prefill_speed(previous, current):
    if not (previous and current) or previous["total"] != current["total"]:
        return 0.0
    elapsed = current["time"] - previous["time"]
    read = current["read"] - previous["read"]
    return round(read / elapsed, 1) if elapsed > 0 and read >= 0 else 0.0


def _latest_prompt_ms(requests):
    return requests[0].get("prompt_ms") if requests else None


# ----------------------------------------------------------------------------
# Host card (the same shape poll_discrete builds from nvidia-smi over SSH)
# ----------------------------------------------------------------------------
def node_reading(metrics):
    hardware = metrics.get("hardware") or {}
    ram_used, ram_total = _mib(hardware.get("ram_used")), _mib(hardware.get("ram_total"))
    model = metrics["engine"].get("model")
    return {"gpus": [_gpu(hardware, metrics.get("hardware_static") or {})],
            "cpu_temp": None, "cpu_temps": [],
            "mem_used_mb": ram_used, "mem_total_mb": ram_total, "mem_pct": _percent(ram_used, ram_total),
            "models": [{"name": model, "label": model, "up": True}]}


def _gpu(hardware, static):
    used, total = _mib(hardware.get("gpu_mem_used")), _mib(hardware.get("gpu_mem_total"))
    return {"index": 0, "name": (static.get("gpu_name") or "GPU").replace("NVIDIA GeForce ", ""),
            "temp": hardware.get("gpu_temp"), "power": _one_decimal(hardware.get("gpu_power")),
            "power_limit": hardware.get("gpu_power_limit"), "util": hardware.get("gpu_util"),
            "mem_used_mb": used, "mem_total_mb": total, "mem_pct": _percent(used, total),
            "fan": None, "gr_clock": None, "mem_clock": None}


def _mib(value_bytes):
    return None if value_bytes is None else round(value_bytes / MIB)


def _percent(used, total):
    return round(used / total * 100.0, 1) if used is not None and total else None


def _one_decimal(value):
    return None if value is None else round(value, 1)


# ----------------------------------------------------------------------------
# Token bank
# ----------------------------------------------------------------------------
def token_counts(metrics):
    """(prompt tokens, generated tokens) since the Strata server started."""
    totals = metrics.get("totals") or {}
    return float(totals.get("prompt_tokens") or 0), float(totals.get("output_tokens") or 0)
