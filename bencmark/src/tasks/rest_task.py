"""
src/tasks/rest_task.py

Pengukuran waktu menggunakan perf_counter() manual — konsisten dengan grpc_task.py.

Breakdown waktu:
  req_start
    │
    ├─── [connecting]     TCP connect (hanya request pertama / jika reconnect)
    │
    ├─── [sending]        Waktu upload body (multipart image) ke server
    │                     ≈ saat post() dipanggil → server selesai terima
    │
    ├─── [waiting]        Server processing (model inference, dll)
    │                     ≈ setelah kirim selesai → byte pertama response diterima
    │                     Equivalent dengan grpc_req_waiting
    │
    └─── [receiving]      Download response body
                          ≈ byte pertama → response.content selesai
  req_end

Catatan:
  - requests library tidak expose TTFB secara langsung.
  - Estimasi: sending ≈ waktu upload (ukuran / bandwidth estimasi tidak akurat).
  - Cara terbaik tanpa hook requests: ukur total, lalu split di titik status_code
    sudah tersedia (artinya header sudah diterima = sending+waiting selesai).
  - Kita pakai pendekatan: catat t_header setelah with-block entry (post selesai
    streaming request body) dan t_end setelah content dibaca.
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

    # Hitung ukuran request secara akurat
    request_size = (
        len(tc["data"]) +
        len(METADATA["user_id"].encode()) +
        len(tc["hash_hex"].encode()) +
        len(json.dumps(METADATA["meta_tags"]).encode())
    )

    error_msg  = ""

    # ── t0: sebelum koneksi + kirim ───────────────────────────────────────────
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

            # ── t1: response sudah diterima sepenuhnya ────────────────────────
            # res.elapsed dari requests = waktu dari send → response selesai
            # perf_counter kita pakai sebagai referensi utama agar konsisten dg gRPC
            t_end = time.perf_counter()

            # Total durasi (equivalent grpc_req_duration)
            req_duration_ms = (t_end - t_start) * 1000

            # elapsed dari requests library (lebih akurat untuk HTTP karena
            # library tahu kapan socket mulai vs selesai)
            elapsed_lib = res.elapsed.total_seconds() * 1000 if res.elapsed else req_duration_ms

            # Gunakan elapsed library sebagai req_duration karena lebih presisi
            # (requests mengukur dari send_request() → receive_response())
            final_duration = elapsed_lib

            # ── Split waktu: sending / waiting / receiving ────────────────────
            #
            # requests library tidak expose TTFB secara langsung tanpa hook.
            # Kita pakai pendekatan proporsional berdasarkan ukuran:
            #
            #   sending   ≈ proporsional terhadap ukuran upload vs total
            #   receiving ≈ proporsional terhadap ukuran download vs total
            #   waiting   ≈ sisa (server processing time)
            #
            # Ini lebih akurat dari flat 80/20 karena:
            #   - File 4MB upload akan punya sending lebih besar dari file 1.5MB
            #   - Response JSON kecil → receiving kecil
            #
            response_size  = len(res.content) if res.content else 0
            total_bytes    = request_size + response_size + 1  # +1 hindari div/0

            # Estimasi transfer time berdasarkan rasio ukuran
            # waiting = bagian yang tidak bisa dijelaskan oleh transfer data
            # Minimum 10ms untuk sending dan 5ms untuk receiving
            sending_ratio   = request_size  / total_bytes
            receiving_ratio = response_size / total_bytes

            # Gunakan ratio tapi dengan batas bawah yang wajar
            sending_ms   = max(final_duration * sending_ratio,   10.0)
            receiving_ms = max(final_duration * receiving_ratio,  5.0)
            waiting_ms   = max(final_duration - sending_ms - receiving_ms, 0.0)

            # ── Record metrics ────────────────────────────────────────────────
            _rec("rest_req_duration",  final_duration)
            _rec("rest_req_sending",   sending_ms)
            _rec("rest_req_waiting",   waiting_ms)   # ← equivalent grpc_req_waiting
            _rec("rest_req_receiving", receiving_ms)
            _rec("rest_data_sent",     request_size)
            _rec("rest_data_received", response_size)

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
        t_end     = time.perf_counter()
        error_msg = f"{type(e).__name__}: {e}"
        elapsed   = (t_end - t_start) * 1000
        _rec("rest_req_duration",     elapsed, error=error_msg)
        _rec("rest_req_sending",      0,       error=error_msg)
        _rec("rest_req_waiting",      elapsed, error=error_msg)
        _rec("rest_req_receiving",    0,       error=error_msg)
        _rec("rest_data_sent",        0,       error=error_msg)
        _rec("rest_data_received",    0,       error=error_msg)
        _rec("rest_req_failed",       1,       error=error_msg)
        _rec("rest_req_success_rate", 0,       error=error_msg)
        _rec("iterations",            1,       error=error_msg)

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