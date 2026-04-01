"""
src/tasks/rest_task.py

Perubahan dari versi sebelumnya:
  1. tc = random.choice(TEST_DATASET) menggantikan itertools.cycle
     → distribusi gambar acak per-call, simetris dengan grpc_task.py

  2. wire_size diambil dari tc["wire_size"] (pre-computed di config.py)
     → PreparedRequest.prepare() TIDAK lagi dipanggil di dalam loop benchmark
     → menghilangkan overhead 5–15ms yang sebelumnya terjadi sebelum t_send_start
     → rest_data_sent kini benar-benar nol-overhead, setara dengan grpc_data_sent

  3. _compute_wire_size() dihapus sepenuhnya — logikanya sudah ada di config.py

Breakdown waktu (limitasi HTTP/1.1 didokumentasikan):
  t_send_start
    │
    ├─── rest_req_ttfb      upload body + server processing → byte pertama response
    │                       (tidak bisa dipisah di HTTP/1.1 tanpa akses TCP layer)
    └─── rest_req_receiving download body setelah byte pertama
  t_end

  rest_req_ttfb vs grpc_req_waiting TIDAK setara — lihat docstring di bawah.
  Hanya rest_req_duration vs grpc_req_duration yang bisa dibandingkan langsung
  sebagai completion time (rumusan masalah 1).

Metrik utama per rumusan masalah:
  Rumusan 1 → rest_req_duration (completion time), rest_req_success (transmisi)
  Rumusan 2 → rest_data_sent (= wire_size pre-computed), rest_data_received
  Rumusan 4 → rest_active_requests, rest_req_duration distribusi saat spike

Catatan metodologis (dokumentasikan di laporan):
  rest_data_sent = multipart/form-data body size aktual (termasuk boundary,
  Content-Disposition, dan Content-Type per field). HTTP/1.1 request line
  dan header tidak terhitung — simetris dengan gRPC (HTTP/2 HEADERS + HPACK
  tidak terhitung). Keduanya comparable sebagai application payload size.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time

from src.config.config import TEST_DATASET, METADATA, TIMEOUT
from src.metrics.metrics import collector

_active_requests = 0

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

    # random.choice: distribusi gambar acak per-call, simetris dengan grpc_task.py
    tc = random.choice(TEST_DATASET)

    with _active_lock:
        _active_requests += 1
    _rec("rest_active_requests", _active_requests)

    # wire_size pre-computed di config.py — nol overhead di measurement window
    # mencakup multipart boundary + Content-Disposition + Content-Type per field
    wire_size = tc["wire_size"]

    files = {"file": (tc["filename"], tc["data"], "image/jpeg")}
    data  = {
        "user_id":       METADATA["user_id"],
        "client_sha256": tc["hash_hex"],
        "metadata":      json.dumps(METADATA["meta_tags"]),
    }

    t_send_start = time.perf_counter()

    try:
        with client.post(
            "/analyze-skin",
            files=files,
            data=data,
            timeout=TIMEOUT,
            stream=True,           # stream=True agar t_ttfb bisa diukur aktual
            name="REST /analyze-skin",
            catch_response=True,
        ) as res:

            # TTFB: dari t_send_start sampai byte pertama response.
            # Mencakup upload + server processing — tidak bisa dipisah di HTTP/1.1.
            # Ini bukan setara dengan grpc_req_waiting (yang murni server processing).
            # Hanya rest_req_duration vs grpc_req_duration yang comparable (rumusan 1).
            t_ttfb = time.perf_counter()

            content = res.content   # baca seluruh body setelah TTFB terukur
            t_end   = time.perf_counter()

            req_duration_ms = (t_end - t_send_start) * 1000
            ttfb_ms         = (t_ttfb - t_send_start) * 1000
            receiving_ms    = max((t_end - t_ttfb) * 1000, 0.0)
            response_size   = len(content) if content else 0

            # ── Rumusan 1: completion time & breakdown ────────────────────
            _rec("rest_req_duration",  req_duration_ms)  # = "completion time" rumusan 1
            _rec("rest_req_ttfb",      ttfb_ms)          # upload+waiting, tidak dipisah
            _rec("rest_req_receiving", receiving_ms)

            # ── Rumusan 2: payload size — pre-computed, nol overhead ──────
            _rec("rest_data_sent",     wire_size)         # multipart wire size
            _rec("rest_data_received", response_size)

            failure_reason = ""

            if res.status_code < 200 or res.status_code >= 300:
                failure_reason = f"HTTP {res.status_code}: {res.text[:200]}"
            else:
                try:
                    body    = res.json()
                    warning = _assert_with_warning(body, tc)
                    if warning:
                        # Label mismatch = warning, bukan failure transmisi.
                        # Success rate (rumusan 1) tidak terpengaruh akurasi model AI.
                        print(f"[rest_task] LABEL WARNING: {warning}")
                        _rec("rest_label_warning", 1, error=warning)
                    if not _assert_structure(body):
                        failure_reason = "response structure invalid"
                except Exception as e:
                    failure_reason = f"JSON parse error: {e}"

            # ── Rumusan 1: success = keberhasilan transmisi ───────────────
            if failure_reason:
                _rec("rest_req_success", 0, error=failure_reason)
                _rec("iterations",       1, error=failure_reason)
                res.failure(failure_reason)
            else:
                _rec("rest_req_success", 1)
                _rec("iterations",       1)
                res.success()

    except Exception as e:
        t_end     = time.perf_counter()
        error_msg = f"{type(e).__name__}: {e}"
        elapsed   = (t_end - t_send_start) * 1000

        _rec("rest_req_duration",  elapsed,    error=error_msg)
        _rec("rest_req_ttfb",      elapsed,    error=error_msg)
        _rec("rest_req_receiving", 0,          error=error_msg)
        _rec("rest_data_sent",     0,          error=error_msg)
        _rec("rest_data_received", 0,          error=error_msg)
        _rec("rest_req_success",   0,          error=error_msg)
        _rec("iterations",         1,          error=error_msg)

    finally:
        with _active_lock:
            _active_requests -= 1
        _rec("rest_active_requests", _active_requests)


def _assert_structure(body: dict) -> bool:
    """
    Validasi struktur response — tanpa cek label AI.
    Failure di sini = server tidak mengembalikan format yang benar.
    Success rate (rumusan 1) murni mencerminkan keberhasilan transmisi.
    """
    if not isinstance(body.get("analysis_id"), str):
        return False
    if not isinstance(body.get("server_sha256"), str):
        return False
    results = body.get("results", [])
    if not isinstance(results, list) or not results:
        return False
    top  = results[0]
    conf = top.get("confidence", -1)
    if not (0 <= conf <= 1):
        return False
    for f in ("label", "description", "recommendation"):
        if not isinstance(top.get(f), str):
            return False
    return True


def _assert_with_warning(body: dict, tc: dict) -> str | None:
    """
    Cek label vs expected — warning only, tidak mempengaruhi success rate.
    Dipisah agar akurasi model AI server tidak mencemari metrik transmisi.
    """
    results = body.get("results", [])
    if not results:
        return None
    top      = results[0]
    got      = top.get("label", "")
    expected = tc["expected_label"]
    if got != expected:
        return (
            f"label mismatch: got '{got}' expected '{expected}' "
            f"(file: {tc['filename']})"
        )
    return None