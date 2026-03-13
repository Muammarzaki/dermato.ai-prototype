"""
src/tasks/grpc_task.py

Fase timing (identik dengan rest_task.py):

  req_start
     │
     ├─── [sending]   → stream semua chunk ke server
     │                   ≈ rest: upload multipart selesai
     │
     ├─── [waiting]   → server processing (selesai kirim → response diterima)
     │                   ≈ rest: TTFB (send_end → resp_start)
     │
     ├─── [receiving] → deserialize response protobuf
     │                   ≈ rest: baca body response
     │
  req_end
     │
     └─── req_duration = req_end - req_start  (identik dengan REST)
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
_active_lock = __import__("gevent.lock", fromlist=["BoundedSemaphore"]).BoundedSemaphore(1)

SCENARIO = os.environ.get("SCENARIO", "load")
NETWORK = os.environ.get("NETWORK", "normal")


def _rec(metric: str, value: float, error: str = "") -> None:
    collector.record("grpc", SCENARIO, NETWORK, metric, value, error=error)


# ─── Dynamic chunk size ─────────────────────────────────────────────────────
#
# Chunk size ditentukan oleh kondisi jaringan, bukan ukuran file.
#
# normal  → 256KB: loopback/LAN, BDP sangat kecil, chunk besar OK
# 4g      → 128KB: 1280KB/s, delay 80ms, loss 0.5%
# poor    → 128KB: no rate limit, delay 100ms, loss 1%
# 3g      →  64KB: 48KB/s — chunk >64KB butuh >1s per chunk, rentan timeout
#                   BDP = 48 × 0.4s = 19KB → 64KB sudah cukup
# worst   → 128KB: TIDAK ada rate limit! hanya delay 300ms + loss 3%
#                   Sebelumnya worst=40% < 3g=50% — TERBALIK karena
#                   worst tidak punya rate limit tapi chunk-nya lebih kecil.
#                   Fix: worst = 128KB (lebih besar dari 3g=64KB)

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
    state["chunk_times_ms"] = []  # waktu per chunk (ms)
    state["chunk_bytes_list"] = []  # ukuran aktual per chunk (bytes)

    # Gunakan memoryview untuk hindari copy bytes per chunk.
    # memoryview(data)[offset:end] tidak mengalokasikan memory baru —
    # hanya pointer ke slice dari bytes asli.
    # bytes() dipanggil hanya saat protobuf benar-benar serialize ke wire.
    data_view = memoryview(data)

    while offset < len(data):
        end = min(offset + chunk_size, len(data))
        chunk_size_actual = end - offset

        # Zero-copy slice — tidak alokasi memory baru sampai protobuf serialize
        chunk = pb2.AnalyzeSkinRequest(chunk=bytes(data_view[offset:end]))

        t_chunk_start = time.perf_counter()
        state["bytes_sent"] += chunk_size_actual
        state["chunk_count"] += 1
        yield chunk  # gRPC kirim chunk di sini
        t_chunk_end = time.perf_counter()  # kontrol kembali = chunk ter-ack / flow control

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
        "send_start": 0.0,
        "send_end": 0.0,
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

        # Setelah response diterima dari network
        resp_received = time.perf_counter()

        # receiving = waktu deserialize response (analog REST baca body)
        resp_bytes = len(res.SerializeToString()) if res else 0
        resp_end = time.perf_counter()

        # ── Hitung fase (ms) — definisi identik dengan REST ──────────────────
        req_duration = (resp_end - req_start) * 1000
        sending_time = (state["send_end"] - state["send_start"]) * 1000 if state["send_end"] else 0
        waiting_time = (resp_received - state["send_end"]) * 1000 if state["send_end"] else 0
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

        # ── Per-chunk timing stats ────────────────────────────────────────────
        # Berguna untuk analisis: apakah bottleneck di chunk awal atau akhir?
        # Apakah jaringan buruk menyebabkan chunk tertentu jauh lebih lambat?
        chunk_times = state.get("chunk_times_ms", [])
        if chunk_times:
            _rec("grpc_chunk_time_avg_ms", sum(chunk_times) / len(chunk_times))
            _rec("grpc_chunk_time_min_ms", min(chunk_times))
            _rec("grpc_chunk_time_max_ms", max(chunk_times))
            # Selisih max-min: indikator jitter/variasi antar chunk
            # Nilai tinggi = ada chunk yang jauh lebih lambat (retransmit, loss)
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

    except Exception as e:
        import grpc as _grpc
        code = getattr(e, "code", lambda: None)()
        if code in (_grpc.StatusCode.DEADLINE_EXCEEDED, _grpc.StatusCode.UNAVAILABLE):
            error_msg = f"TIMEOUT({NETWORK}): {code.name} setelah {TIMEOUT}s"
        else:
            error_msg = f"{type(e).__name__}: {e}"
        exc = e
        elapsed_ms = (time.perf_counter() - req_start) * 1000
        # ERROR path — konsisten dengan REST error path:
        # sending/receiving = 0 (tidak sempat/selesai transfer)
        # waiting = elapsed (semua waktu dianggap menunggu/timeout)
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
