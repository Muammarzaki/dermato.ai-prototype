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

_cycle          = itertools.cycle(TEST_DATASET)
_active_streams = 0
_active_lock    = __import__("threading").Lock()

SCENARIO = os.environ.get("SCENARIO", "load")
NETWORK  = os.environ.get("NETWORK",  "normal")


def _rec(metric: str, value: float, error: str = "") -> None:
    collector.record("grpc", SCENARIO, NETWORK, metric, value, error=error)


# ─── Dynamic chunk size ───────────────────────────────────────────────────────
#
# Strategi: chunk size ditentukan oleh DUA faktor — ukuran file + kondisi jaringan.
#
# Ukuran file menentukan batas atas chunk:
#   ≤1MB  → max 256KB  (sedikit chunk, file kecil aman)
#   ≤2MB  → max 512KB
#   ≤3MB  → max 768KB
#   >3MB  → max 1024KB (1MB)
#
# Kondisi jaringan men-scale down chunk size:
#   normal/4g → 100% (pakai max)
#   poor      → 75%  (sedikit lebih kecil, toleransi loss 1%)
#   3g        → 50%  (bandwidth 384kbit, chunk besar membuang bandwidth kalau loss)
#   worst     → 40%  (loss 3%, chunk kecil agar retransmit tidak mahal)
#
# Kenapa ini lebih baik dari flat 64KB:
#   - 4MB di normal  : 64 chunk → 4 chunk (16x lebih sedikit round-trip)
#   - 4MB di worst   : 64 chunk → 10 chunk (toleran loss tapi tetap jauh lebih baik)
#   - REST tidak bisa streaming, jadi kirim sekaligus — gRPC dengan chunk besar
#     mendekati efisiensi REST tapi tetap bisa mulai diproses lebih awal (streaming)

_NETWORK_SCALE = {
    "normal": 1.00,
    "4g":     1.00,
    "poor":   0.75,
    "3g":     0.50,
    "worst":  0.40,
}

def _get_chunk_size(file_size_bytes: int) -> int:
    """Hitung chunk size optimal berdasarkan ukuran file dan kondisi jaringan."""
    size_mb = file_size_bytes / (1024 * 1024)

    # Batas atas berdasarkan ukuran file
    if size_mb <= 1.0:
        base_kb = 256
    elif size_mb <= 2.0:
        base_kb = 512
    elif size_mb <= 3.0:
        base_kb = 768
    else:
        base_kb = 1024

    # Scale down berdasarkan network
    scale  = _NETWORK_SCALE.get(NETWORK, 0.75)
    final  = int(base_kb * scale) * 1024  # konversi ke bytes

    # Floor: minimal 64KB supaya tidak terlalu banyak chunk di kondisi terburuk
    return max(final, 64 * 1024)


def _request_iter(tc: dict, state: dict) -> Iterator:
    pb2        = get_pb2()
    chunk_size = _get_chunk_size(len(tc["data"]))
    state["chunk_size"] = chunk_size  # untuk logging

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
    state["chunk_count"] = 0

    while offset < len(data):
        end   = min(offset + chunk_size, len(data))
        chunk = pb2.AnalyzeSkinRequest(chunk=data[offset:end])
        state["bytes_sent"]  += end - offset
        state["chunk_count"] += 1
        yield chunk
        offset = end

    state["send_end"] = time.perf_counter()


def analyze_skin(stub, environment) -> None:
    global _active_streams

    tc    = next(_cycle)
    state = {
        "bytes_sent":  0,
        "send_start":  0.0,
        "send_end":    0.0,
        "chunk_size":  0,
        "chunk_count": 0,
    }

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

        req_duration = (resp_time - req_start)                       * 1000
        sending_time = (state["send_end"] - state["send_start"])     * 1000 if state["send_end"] else 0
        waiting_time = (resp_time - state["send_end"])               * 1000 if state["send_end"] else 0

        _rec("grpc_req_duration",    req_duration)
        _rec("grpc_stream_duration", req_duration)
        _rec("grpc_req_sending",     sending_time)
        _rec("grpc_req_waiting",     waiting_time)
        _rec("grpc_data_sent",       state["bytes_sent"])
        _rec("grpc_data_received",   resp_bytes)
        # chunk_count berguna untuk analisis efisiensi di notebook
        _rec("grpc_chunk_count",     state["chunk_count"])
        _rec("grpc_chunk_size_kb",   state["chunk_size"] / 1024)

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