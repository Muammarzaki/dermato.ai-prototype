"""
src/tasks/grpc_task.py
"""

from __future__ import annotations

import itertools
import time
import os
from typing import Iterator

from src.config.config import TEST_DATASET, METADATA, CHUNK_SIZE, TIMEOUT
from src.utils.grpc_client import get_pb2
from src.metrics.metrics import collector

_cycle          = itertools.cycle(TEST_DATASET)
_active_streams = 0
_active_lock    = __import__("threading").Lock()

SCENARIO = os.environ.get("SCENARIO", "load")
NETWORK  = os.environ.get("NETWORK",  "normal")


def _rec(metric: str, value: float, error: str = "") -> None:
    collector.record("grpc", SCENARIO, NETWORK, metric, value, error=error)


def _request_iter(tc: dict, state: dict) -> Iterator:
    pb2 = get_pb2()

    meta_msg = pb2.AnalyzeSkinRequest(
        info=pb2.ImageInfo(
            user_id       = METADATA["user_id"],
            image_type    = METADATA["image_type"],
            client_sha256 = tc["hash_bytes"],
            metadata      = {
                **METADATA["meta_tags"],
                "file_size": str(len(tc["data"])),
            },
        )
    )
    state["bytes_sent"] += len(meta_msg.SerializeToString())
    yield meta_msg

    data   = tc["data"]
    offset = 0
    state["send_start"] = time.perf_counter()

    while offset < len(data):
        end   = min(offset + CHUNK_SIZE, len(data))
        chunk = pb2.AnalyzeSkinRequest(chunk=data[offset:end])
        state["bytes_sent"] += end - offset
        yield chunk
        offset = end

    state["send_end"] = time.perf_counter()


def analyze_skin(stub, environment) -> None:
    global _active_streams

    tc    = next(_cycle)
    state = {"bytes_sent": 0, "send_start": 0.0, "send_end": 0.0}

    with _active_lock:
        _active_streams += 1
    _rec("grpc_active_streams", _active_streams)

    req_start = time.perf_counter()
    exc       = None
    error_msg = ""

    try:
        res = stub.AnalyzeSkin(_request_iter(tc, state), timeout=TIMEOUT)

        resp_time  = time.perf_counter()
        resp_bytes = len(res.SerializeToString()) if res else 0

        req_duration  = (resp_time - req_start)              * 1000
        sending_time  = (state["send_end"] - state["send_start"]) * 1000 if state["send_end"] else 0
        waiting_time  = (resp_time - state["send_end"])       * 1000 if state["send_end"] else 0

        _rec("grpc_req_duration",    req_duration)
        _rec("grpc_stream_duration", req_duration)
        _rec("grpc_req_sending",     sending_time)
        _rec("grpc_req_waiting",     waiting_time)
        _rec("grpc_data_sent",       state["bytes_sent"])
        _rec("grpc_data_received",   resp_bytes)

        assertion_err = _assert(res, tc)
        if assertion_err:
            error_msg = assertion_err
            exc       = AssertionError(assertion_err)
            _rec("grpc_req_failed",       1, error=error_msg)
            _rec("grpc_req_success_rate", 0, error=error_msg)
            _rec("iterations",            1, error=error_msg)
        else:
            _rec("grpc_req_failed",       0)
            _rec("grpc_req_success_rate", 1)
            _rec("iterations",            1)

    except Exception as e:
        import grpc as _grpc
        # DEADLINE_EXCEEDED dan UNAVAILABLE adalah error jaringan yang wajar
        # di kondisi worst/3g — catat sebagai failure biasa, jangan crash worker
        code = getattr(e, "code", lambda: None)()
        if code in (_grpc.StatusCode.DEADLINE_EXCEEDED, _grpc.StatusCode.UNAVAILABLE):
            error_msg = f"TIMEOUT({NETWORK}): {code.name} setelah {TIMEOUT}s"
        else:
            error_msg = f"{type(e).__name__}: {e}"
        exc        = e
        elapsed_ms = (time.perf_counter() - req_start) * 1000
        _rec("grpc_req_duration",    elapsed_ms, error=error_msg)
        _rec("grpc_req_failed",      1,          error=error_msg)
        _rec("grpc_req_success_rate",0,          error=error_msg)
        _rec("iterations",           1,          error=error_msg)

    finally:
        with _active_lock:
            _active_streams -= 1
        _rec("grpc_active_streams", _active_streams)

        elapsed_ms = (time.perf_counter() - req_start) * 1000
        environment.events.request.fire(
            request_type    = "gRPC",
            name            = "SkinAnalysisService/AnalyzeSkin",
            response_time   = elapsed_ms,
            response_length = 0,
            exception       = exc,
            context         = {},
        )


def _assert(res, tc: dict) -> str | None:
    failures = []

    if not isinstance(getattr(res, "analysis_id", None), str):
        failures.append("missing analysisId")
    if not isinstance(getattr(res, "server_sha256", None), bytes):
        failures.append("missing server_sha256")

    results = list(getattr(res, "results", []))
    if not results:
        failures.append("results kosong")
    else:
        top  = results[0]
        conf = getattr(top, "confidence", -1)
        if not (0 <= conf <= 1):
            failures.append(f"confidence out of range: {conf}")
        for f in ("label", "description", "recommendation"):
            if not isinstance(getattr(top, f, None), str):
                failures.append(f"missing {f}")
        if getattr(top, "label", "") != tc["expected_label"]:
            failures.append(
                f"wrong label: got '{getattr(top, 'label', '')}' "
                f"expected '{tc['expected_label']}'"
            )

    return " | ".join(failures) if failures else None