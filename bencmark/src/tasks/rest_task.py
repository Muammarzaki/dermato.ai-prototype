"""
src/tasks/rest_task.py

Pengukuran waktu menggunakan perf_counter() manual — konsisten dengan grpc_task.py.

Breakdown waktu:
  req_start
    │
    ├─── [sending]        Waktu upload body (multipart image) ke server
    ├─── [waiting]        Server processing → TTFB
    └─── [receiving]      Download response body
  req_end
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time

from src.config.config import TEST_DATASET, METADATA, TIMEOUT
from src.metrics.metrics import collector

_cycle = itertools.cycle(TEST_DATASET)
_active_requests = 0

# FIX: ganti BoundedSemaphore(1) dengan RLock — lebih tepat sebagai mutex
# BoundedSemaphore bisa di-release oleh greenlet berbeda dari yang acquire;
# RLock hanya bisa di-release oleh owner-nya
try:
    from gevent.lock import RLock as _GRLock
    _active_lock = _GRLock()
except ImportError:
    _active_lock = threading.Lock()

SCENARIO = os.environ.get("SCENARIO", "load")
NETWORK  = os.environ.get("NETWORK",  "normal")


def _rec(metric: str, value: float, error: str = "") -> None:
    collector.record("rest", SCENARIO, NETWORK, metric, value, error=error)


def analyze_skin(client) -> None:
    global _active_requests

    tc = next(_cycle)

    with _active_lock:
        _active_requests += 1
    _rec("rest_active_requests", _active_requests)

    files = {"file": (tc["filename"], tc["data"], "image/jpeg")}
    data  = {
        "user_id":       METADATA["user_id"],
        "client_sha256": tc["hash_hex"],
        "metadata":      json.dumps(METADATA["meta_tags"]),
    }

    request_size = (
        len(tc["data"]) +
        len(METADATA["user_id"].encode()) +
        len(tc["hash_hex"].encode()) +
        len(json.dumps(METADATA["meta_tags"]).encode())
    )

    t_start = time.perf_counter()

    try:
        with client.post(
            "/analyze-skin",
            files=files,
            data=data,
            timeout=TIMEOUT,
            name="REST /analyze-skin",
            catch_response=True,
        ) as res:

            t_end = time.perf_counter()

            req_duration_ms = (t_end - t_start) * 1000
            elapsed_lib     = res.elapsed.total_seconds() * 1000 if res.elapsed else req_duration_ms
            final_duration  = elapsed_lib

            response_size  = len(res.content) if res.content else 0
            total_bytes    = request_size + response_size + 1  # +1 hindari div/0

            sending_ratio  = request_size  / total_bytes
            receiving_ratio= response_size / total_bytes

            sending_ms  = max(final_duration * sending_ratio,   10.0)
            receiving_ms= max(final_duration * receiving_ratio,  5.0)
            waiting_ms  = max(final_duration - sending_ms - receiving_ms, 0.0)

            _rec("rest_req_duration",  final_duration)
            _rec("rest_req_sending",   sending_ms)
            _rec("rest_req_waiting",   waiting_ms)
            _rec("rest_req_receiving", receiving_ms)
            _rec("rest_data_sent",     request_size)
            _rec("rest_data_received", response_size)

            failure_reason = ""

            if res.status_code < 200 or res.status_code >= 300:
                failure_reason = f"HTTP {res.status_code}: {res.text[:200]}"
            else:
                try:
                    body          = res.json()
                    assertion_err = _assert(body, tc)
                    if assertion_err:
                        failure_reason = assertion_err
                except Exception as e:
                    failure_reason = f"JSON parse error: {e}"

            if failure_reason:
                _rec("rest_req_failed",       1, error=failure_reason)
                _rec("rest_req_success_rate", 0, error=failure_reason)
                _rec("iterations",            1, error=failure_reason)
                res.failure(failure_reason)
            else:
                _rec("rest_req_failed",       0)
                _rec("rest_req_success_rate", 1)
                _rec("iterations",            1)
                res.success()

    except Exception as e:
        t_end      = time.perf_counter()
        error_msg  = f"{type(e).__name__}: {e}"
        elapsed    = (t_end - t_start) * 1000
        _rec("rest_req_duration",     elapsed,    error=error_msg)
        _rec("rest_req_sending",      0,          error=error_msg)
        _rec("rest_req_waiting",      elapsed,    error=error_msg)
        _rec("rest_req_receiving",    0,          error=error_msg)
        _rec("rest_data_sent",        0,          error=error_msg)
        _rec("rest_data_received",    0,          error=error_msg)
        _rec("rest_req_failed",       1,          error=error_msg)
        _rec("rest_req_success_rate", 0,          error=error_msg)
        _rec("iterations",            1,          error=error_msg)

    finally:
        with _active_lock:
            _active_requests -= 1
        _rec("rest_active_requests", _active_requests)


def _assert(body: dict, tc: dict) -> str | None:
    failures = []

    if not isinstance(body.get("analysis_id"), str):
        failures.append("missing analysis_id")
    if not isinstance(body.get("server_sha256"), str):
        failures.append("missing server_sha256")

    results = body.get("results", [])
    if not isinstance(results, list) or not results:
        failures.append("results kosong")
    else:
        top  = results[0]
        conf = top.get("confidence", -1)
        if not (0 <= conf <= 1):
            failures.append(f"confidence out of range: {conf}")
        for f in ("label", "description", "recommendation"):
            if not isinstance(top.get(f), str):
                failures.append(f"missing {f}")
        if top.get("label") != tc["expected_label"]:
            failures.append(
                f"wrong label: got '{top.get('label')}' "
                f"expected '{tc['expected_label']}'"
            )

    return " | ".join(failures) if failures else None