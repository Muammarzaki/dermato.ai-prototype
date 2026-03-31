"""
src/tasks/grpc_task.py
"""

from __future__ import annotations

import itertools
import time
import os
from typing import Iterator

from src.config.config import TEST_DATASET, METADATA, TIMEOUT
from src.utils.grpc_client import get_pb2
from src.metrics.metrics import collector

_cycle = itertools.cycle(TEST_DATASET)
_active_streams = 0

# FIX: ganti BoundedSemaphore(1) dengan RLock — lebih tepat sebagai mutex
# BoundedSemaphore bisa di-release oleh greenlet berbeda, RLock tidak
try:
    from gevent.lock import RLock as _GRLock

    _active_lock = _GRLock()
except ImportError:
    import threading

    _active_lock = threading.Lock()

SCENARIO = os.environ.get("SCENARIO", "load")
NETWORK = os.environ.get("NETWORK", "normal")


def _rec(metric: str, value: float, error: str = "") -> None:
    collector.record("grpc", SCENARIO, NETWORK, metric, value, error=error)


CHUNK_SIZE_BYTES = 256 * 1024


def _get_chunk_size() -> int:
    return CHUNK_SIZE_BYTES


def _request_iter(tc: dict, state: dict) -> Iterator:
    pb2 = get_pb2()
    chunk_size = _get_chunk_size()
    state["chunk_size"] = chunk_size

    meta_msg = pb2.AnalyzeSkinRequest(
        info=pb2.ImageInfo(
            user_id=METADATA["user_id"],
            image_type=METADATA["image_type"],
            client_sha256=tc["hash_bytes"],
            metadata={
                **METADATA["meta_tags"],
                "file_size": str(len(tc["data"])),
            },
        )
    )
    state["bytes_sent"] += len(meta_msg.SerializeToString())
    yield meta_msg

    data = tc["data"]
    offset = 0
    state["send_start"] = time.perf_counter()
    state["chunk_count"] = 0
    state["chunk_times_ms"] = []
    state["chunk_bytes_list"] = []

    data_view = memoryview(data)

    while offset < len(data):
        end = min(offset + chunk_size, len(data))
        chunk_size_actual = end - offset

        chunk = pb2.AnalyzeSkinRequest(chunk=bytes(data_view[offset:end]))

        t_chunk_start = time.perf_counter()
        state["bytes_sent"] += chunk_size_actual
        state["chunk_count"] += 1
        yield chunk
        t_chunk_end = time.perf_counter()

        chunk_ms = (t_chunk_end - t_chunk_start) * 1000
        state["chunk_times_ms"].append(chunk_ms)
        state["chunk_bytes_list"].append(chunk_size_actual)

        offset = end

    state["send_end"] = time.perf_counter()


def analyze_skin(stub, environment) -> None:
    global _active_streams

    tc = next(_cycle)
    state = {
        "bytes_sent": 0,
        # FIX: gunakan None sebagai sentinel, bukan 0.0
        # 0.0 evaluasi False di 'if state["send_end"]' → timing jadi salah
        # None lebih eksplisit: "belum di-set" vs "di-set ke nilai 0.0"
        "send_start": None,
        "send_end": None,
        "chunk_size": 0,
        "chunk_count": 0,
        "chunk_times_ms": [],
        "chunk_bytes_list": [],
    }

    with _active_lock:
        _active_streams += 1
    _rec("grpc_active_streams", _active_streams)

    req_start = time.perf_counter()
    exc = None
    error_msg = ""

    try:
        res = stub.AnalyzeSkin(_request_iter(tc, state), timeout=TIMEOUT)

        resp_received = time.perf_counter()

        resp_bytes = len(res.SerializeToString()) if res else 0
        resp_end = time.perf_counter()

        req_duration = (resp_end - req_start) * 1000

        # FIX: cek 'is not None' — benar secara semantik, tidak bergantung
        # pada nilai numerik send_end (0.0 akan evaluasi False secara salah)
        sending_time = (state["send_end"] - state["send_start"]) * 1000 \
            if state["send_end"] is not None else 0.0
        waiting_time = (resp_received - state["send_end"]) * 1000 \
            if state["send_end"] is not None else 0.0
        receiving_time = (resp_end - resp_received) * 1000

        sending_time = max(sending_time, 0.0)
        waiting_time = max(waiting_time, 0.0)
        receiving_time = max(receiving_time, 0.0)

        _rec("grpc_req_duration", req_duration)
        _rec("grpc_req_sending", sending_time)
        _rec("grpc_req_waiting", waiting_time)
        _rec("grpc_req_receiving", receiving_time)
        _rec("grpc_data_sent", state["bytes_sent"])
        _rec("grpc_data_received", resp_bytes)
        _rec("grpc_chunk_count", state["chunk_count"])
        _rec("grpc_chunk_size_kb", state["chunk_size"] / 1024)

        chunk_times = state.get("chunk_times_ms", [])
        if chunk_times:
            _rec("grpc_chunk_time_avg_ms", sum(chunk_times) / len(chunk_times))
            _rec("grpc_chunk_time_min_ms", min(chunk_times))
            _rec("grpc_chunk_time_max_ms", max(chunk_times))
            _rec("grpc_chunk_time_jitter_ms", max(chunk_times) - min(chunk_times))
        else:
            _rec("grpc_chunk_time_avg_ms", 0.0)
            _rec("grpc_chunk_time_min_ms", 0.0)
            _rec("grpc_chunk_time_max_ms", 0.0)
            _rec("grpc_chunk_time_jitter_ms", 0.0)

        assertion_err = _assert(res, tc)
        if assertion_err:
            error_msg = assertion_err
            exc = AssertionError(assertion_err)
            _rec("grpc_req_failed", 1, error=error_msg)
            _rec("grpc_req_success_rate", 0, error=error_msg)
            _rec("iterations", 1, error=error_msg)
        else:
            _rec("grpc_req_failed", 0)
            _rec("grpc_req_success_rate", 1)
            _rec("iterations", 1)

            # ... existing code ...

    except Exception as e:
        import grpc as _grpc
        code = getattr(e, "code", lambda: None)()

        if code == _grpc.StatusCode.DEADLINE_EXCEEDED:
            error_msg = f"TIMEOUT({NETWORK}): {code.name} setelah {TIMEOUT}s"
        elif code == _grpc.StatusCode.UNAVAILABLE:
            error_msg = f"TRANSPORT({NETWORK}): {code.name}"
        else:
            error_msg = f"{type(e).__name__}: {e}"

        exc = e
        elapsed_ms = (time.perf_counter() - req_start) * 1000
        _rec("grpc_req_duration", elapsed_ms, error=error_msg)
        _rec("grpc_req_sending", 0, error=error_msg)
        _rec("grpc_req_waiting", elapsed_ms, error=error_msg)
        _rec("grpc_req_receiving", 0, error=error_msg)
        _rec("grpc_data_sent", 0, error=error_msg)
        _rec("grpc_data_received", 0, error=error_msg)
        _rec("grpc_chunk_count", 0, error=error_msg)
        _rec("grpc_chunk_size_kb", 0, error=error_msg)
        _rec("grpc_chunk_time_avg_ms", 0, error=error_msg)
        _rec("grpc_chunk_time_min_ms", 0, error=error_msg)
        _rec("grpc_chunk_time_max_ms", 0, error=error_msg)
        _rec("grpc_chunk_time_jitter_ms", 0, error=error_msg)
        _rec("grpc_req_failed", 1, error=error_msg)
        _rec("grpc_req_success_rate", 0, error=error_msg)
        _rec("iterations", 1, error=error_msg)

    finally:
        with _active_lock:
            _active_streams -= 1
        _rec("grpc_active_streams", _active_streams)

        elapsed_ms = (time.perf_counter() - req_start) * 1000
        environment.events.request.fire(
            request_type="gRPC",
            name="SkinAnalysisService/AnalyzeSkin",
            response_time=elapsed_ms,
            response_length=0,
            exception=exc,
            context={},
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
        top = results[0]
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
