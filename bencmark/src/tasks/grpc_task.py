"""
src/tasks/grpc_task.py

Perubahan dari versi sebelumnya:
  1. tc = random.choice(TEST_DATASET) menggantikan itertools.cycle
     → distribusi gambar acak per-call, tidak deterministik per-VU
     → menghilangkan pola round-robin yang tidak realistis

  2. grpc_data_sent menggunakan tc["proto_frame_size"] dari config.py
     → ukuran protobuf serialized seluruh stream (meta + semua chunk)
     → dihitung sekali saat load, nol overhead di measurement window
     → setara dengan rest_data_sent yang juga pre-computed (wire_size)
     → membuat perbandingan payload efficiency (rumusan 2) adil dan simetris

  3. bytes_sent di _request_iter dihapus — tidak dipakai lagi karena
     proto_frame_size sudah pre-computed. Iterator hanya tracking
     chunk_count, chunk_times, dan send timing.

Metrik utama per rumusan masalah:
  Rumusan 1 → grpc_req_duration (completion time), grpc_req_success (transmisi)
  Rumusan 2 → grpc_data_sent (= proto_frame_size), grpc_chunk_count, grpc_chunk_size_kb
  Rumusan 4 → grpc_active_streams, grpc_req_duration distribusi saat spike

Catatan metodologis (dokumentasikan di laporan):
  grpc_data_sent = protobuf application-layer frame size.
  HTTP/2 transport headers (HEADERS frame + HPACK) tidak terhitung —
  limitasi yang SIMETRIS dengan REST (HTTP/1.1 request line + headers
  tidak terhitung). Keduanya comparable sebagai application payload size.

  grpc_req_sending diukur dari perf_counter() di dalam iterator.
  Ada window kecil antara stub.AnalyzeSkin() dipanggil dan iterator
  mulai di-consume oleh gRPC runtime — tidak terhitung di sending_time.
  Efeknya diabaikan di jaringan terdegradasi (dominan oleh latency jaringan).
"""

from __future__ import annotations

import random
import time
import os
from typing import Iterator

from src.config.config import TEST_DATASET, METADATA, TIMEOUT, CHUNK_SIZE_BYTES
from src.utils.grpc_client import get_pb2
from src.metrics.metrics import collector

_active_streams = 0

try:
    from gevent.lock import RLock as _GRLock
    _active_lock = _GRLock()
except ImportError:
    import threading
    _active_lock = threading.Lock()

SCENARIO = os.environ.get("SCENARIO", "load")
NETWORK  = os.environ.get("NETWORK",  "normal")


def _rec(metric: str, value: float, error: str = "") -> None:
    collector.record("grpc", SCENARIO, NETWORK, metric, value, error=error)


def _request_iter(tc: dict, state: dict) -> Iterator:
    """
    Generator stream AnalyzeSkinRequest.

    Tracking di sini:
      - send_start / send_end : untuk grpc_req_sending dan grpc_req_waiting
      - chunk_count           : jumlah chunk yang dikirim
      - chunk_times_ms        : latency per-chunk (untuk jitter analysis)

    bytes_sent TIDAK di-track di sini — sudah pre-computed di config.py
    sebagai tc["proto_frame_size"], dipakai langsung di analyze_skin().
    """
    pb2        = get_pb2()
    chunk_size = CHUNK_SIZE_BYTES  # konsisten dengan nilai yang dipakai config.py

    state["chunk_size"] = chunk_size

    # Pesan pertama: metadata ImageInfo
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
    yield meta_msg

    # Pesan berikutnya: chunk-chunk data
    data      = tc["data"]
    data_view = memoryview(data)
    offset    = 0

    state["send_start"]     = time.perf_counter()
    state["chunk_count"]    = 0
    state["chunk_times_ms"] = []

    while offset < len(data):
        end   = min(offset + chunk_size, len(data))
        chunk = pb2.AnalyzeSkinRequest(chunk=bytes(data_view[offset:end]))

        t0 = time.perf_counter()
        state["chunk_count"] += 1
        yield chunk
        t1 = time.perf_counter()

        state["chunk_times_ms"].append((t1 - t0) * 1000)
        offset = end

    state["send_end"] = time.perf_counter()


def analyze_skin(stub, environment) -> None:
    global _active_streams

    # random.choice: distribusi gambar acak per-call, tidak deterministik per-VU.
    # Menghilangkan pola round-robin yang bisa bias hasil per-gambar.
    tc = random.choice(TEST_DATASET)

    state = {
        "send_start":     None,
        "send_end":       None,
        "chunk_size":     0,
        "chunk_count":    0,
        "chunk_times_ms": [],
    }

    with _active_lock:
        _active_streams += 1
    _rec("grpc_active_streams", _active_streams)

    req_start = time.perf_counter()
    exc       = None
    error_msg = ""

    try:
        res = stub.AnalyzeSkin(_request_iter(tc, state), timeout=TIMEOUT)

        resp_received = time.perf_counter()
        resp_bytes    = len(res.SerializeToString()) if res else 0
        resp_end      = time.perf_counter()

        req_duration = (resp_end - req_start) * 1000

        sending_time = (
            (state["send_end"] - state["send_start"]) * 1000
            if state["send_end"] is not None else 0.0
        )
        waiting_time = (
            (resp_received - state["send_end"]) * 1000
            if state["send_end"] is not None else 0.0
        )
        receiving_time = (resp_end - resp_received) * 1000

        sending_time   = max(sending_time,   0.0)
        waiting_time   = max(waiting_time,   0.0)
        receiving_time = max(receiving_time, 0.0)

        # ── Rumusan 1: completion time & breakdown ────────────────────────
        _rec("grpc_req_duration",  req_duration)   # = "completion time" rumusan 1
        _rec("grpc_req_sending",   sending_time)
        _rec("grpc_req_waiting",   waiting_time)
        _rec("grpc_req_receiving", receiving_time)

        # ── Rumusan 2: payload size — pre-computed, nol overhead benchmark ─
        # grpc_data_sent = proto_frame_size: total serialized bytes (meta + semua chunk)
        # Setara dengan rest_data_sent (multipart wire size) → perbandingan adil
        _rec("grpc_data_sent",     tc["proto_frame_size"])
        _rec("grpc_data_received", resp_bytes)
        _rec("grpc_chunk_count",   state["chunk_count"])
        _rec("grpc_chunk_size_kb", state["chunk_size"] / 1024)

        chunk_times = state["chunk_times_ms"]
        if chunk_times:
            _rec("grpc_chunk_time_avg_ms",    sum(chunk_times) / len(chunk_times))
            _rec("grpc_chunk_time_jitter_ms", max(chunk_times) - min(chunk_times))
        else:
            _rec("grpc_chunk_time_avg_ms",    0.0)
            _rec("grpc_chunk_time_jitter_ms", 0.0)

        # ── Label warning — terpisah dari success rate ────────────────────
        warning = _assert_label_warning(res, tc)
        if warning:
            print(f"[grpc_task] LABEL WARNING: {warning}")
            _rec("grpc_label_warning", 1, error=warning)

        # ── Rumusan 1: success = keberhasilan transmisi, bukan akurasi AI ─
        struct_err = _assert_structure(res)
        if struct_err:
            error_msg = struct_err
            exc       = AssertionError(struct_err)
            _rec("grpc_req_success", 0, error=error_msg)
            _rec("iterations",       1, error=error_msg)
        else:
            _rec("grpc_req_success", 1)
            _rec("iterations",       1)

    except Exception as e:
        import grpc as _grpc
        code = getattr(e, "code", lambda: None)()

        if code == _grpc.StatusCode.DEADLINE_EXCEEDED:
            error_msg = f"TIMEOUT({NETWORK}): {code.name} setelah {TIMEOUT}s"
        elif code == _grpc.StatusCode.UNAVAILABLE:
            error_msg = f"TRANSPORT({NETWORK}): {code.name}"
        else:
            error_msg = f"{type(e).__name__}: {e}"

        exc        = e
        elapsed_ms = (time.perf_counter() - req_start) * 1000

        _rec("grpc_req_duration",         elapsed_ms, error=error_msg)
        _rec("grpc_req_sending",          0,          error=error_msg)
        _rec("grpc_req_waiting",          elapsed_ms, error=error_msg)
        _rec("grpc_req_receiving",        0,          error=error_msg)
        _rec("grpc_data_sent",            0,          error=error_msg)
        _rec("grpc_data_received",        0,          error=error_msg)
        _rec("grpc_chunk_count",          0,          error=error_msg)
        _rec("grpc_chunk_size_kb",        0,          error=error_msg)
        _rec("grpc_chunk_time_avg_ms",    0,          error=error_msg)
        _rec("grpc_chunk_time_jitter_ms", 0,          error=error_msg)
        _rec("grpc_req_success",          0,          error=error_msg)
        _rec("iterations",                1,          error=error_msg)

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


def _assert_structure(res) -> str | None:
    """
    Validasi struktur response — tanpa cek label AI.
    Failure di sini = server tidak mengembalikan format yang benar.
    Success rate (rumusan 1) murni mencerminkan keberhasilan transmisi.
    """
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
    return " | ".join(failures) if failures else None


def _assert_label_warning(res, tc: dict) -> str | None:
    """
    Cek label vs expected — warning only, tidak mempengaruhi success rate.
    Dipisah agar akurasi model AI server tidak mencemari metrik transmisi.
    """
    results = list(getattr(res, "results", []))
    if not results:
        return None
    got      = getattr(results[0], "label", "")
    expected = tc["expected_label"]
    if got != expected:
        return (
            f"label mismatch: got '{got}' expected '{expected}' "
            f"(file: {tc['filename']})"
        )
    return None