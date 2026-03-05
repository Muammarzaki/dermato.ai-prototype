"""
src/tasks/rest_task.py
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time

from src.config.config import TEST_DATASET, METADATA, TIMEOUT
from src.metrics.metrics import collector

_cycle           = itertools.cycle(TEST_DATASET)
_active_requests = 0
_active_lock     = threading.Lock()

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
        len(METADATA["user_id"]) +
        len(tc["hash_hex"]) +
        len(json.dumps(METADATA["meta_tags"]))
    )

    error_msg = ""

    try:
        with client.post(
            "/analyze-skin",
            files=files,
            data=data,
            timeout=TIMEOUT,
            name="REST /analyze-skin",
            catch_response=True,
        ) as res:

            t   = getattr(res, "elapsed", None)
            dur = t.total_seconds() * 1000 if t else 0

            _rec("rest_req_duration",  dur)
            _rec("rest_data_sent",     request_size)
            _rec("rest_data_received", len(res.content) if res.content else 0)

            raw = getattr(res, "_locust_request_meta", {})
            if raw:
                _rec("rest_req_blocked",        raw.get("blocked",         0) or 0)
                _rec("rest_req_connecting",      raw.get("connecting",      0) or 0)
                _rec("rest_req_tls_handshaking", raw.get("tls_handshaking", 0) or 0)
                _rec("rest_req_sending",         raw.get("sending",         0) or 0)
                _rec("rest_req_waiting",         raw.get("waiting",         0) or 0)
                _rec("rest_req_receiving",       raw.get("receiving",       0) or 0)
            else:
                _rec("rest_req_waiting",  dur * 0.8)
                _rec("rest_req_receiving",dur * 0.2)

            # ── Status HTTP ───────────────────────────────────────────────────
            if res.status_code < 200 or res.status_code >= 300:
                error_msg = f"HTTP {res.status_code}: {res.text[:300]}"
                _rec("rest_req_failed",       1, error=error_msg)
                _rec("rest_req_success_rate", 0, error=error_msg)
                _rec("iterations",            1, error=error_msg)
                res.failure(error_msg)
                return

            # ── Parse body ────────────────────────────────────────────────────
            try:
                body = res.json()
            except Exception as e:
                error_msg = f"JSON parse error: {e}"
                _rec("rest_req_failed",       1, error=error_msg)
                _rec("rest_req_success_rate", 0, error=error_msg)
                _rec("iterations",            1, error=error_msg)
                res.failure(error_msg)
                return

            # ── Assertions ────────────────────────────────────────────────────
            assertion_err = _assert(body, tc)
            if assertion_err:
                error_msg = assertion_err
                _rec("rest_req_failed",       1, error=error_msg)
                _rec("rest_req_success_rate", 0, error=error_msg)
                _rec("iterations",            1, error=error_msg)
                res.failure(error_msg)
            else:
                _rec("rest_req_failed",       0)
                _rec("rest_req_success_rate", 1)
                _rec("iterations",            1)
                res.success()

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        _rec("rest_req_failed",       1, error=error_msg)
        _rec("rest_req_success_rate", 0, error=error_msg)
        _rec("iterations",            1, error=error_msg)

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