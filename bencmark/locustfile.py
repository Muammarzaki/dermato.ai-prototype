"""
locustfile.py

Entry point Locust untuk benchmark gRPC vs REST.

Mengikuti struktur kode asli:
  - SCENARIO, NETWORK dibaca via os.environ (sama seperti di task files)
  - CsvListener distart di on_init, di-stop di on_quit
  - analyze_skin() di grpc_task & rest_task namanya sama, alias berbeda

Jalankan:
  SCENARIO=load NETWORK=normal \
    locust -f locustfile.py RestUser --headless -u 10 -r 2 --run-time 9m

  SCENARIO=smoke NETWORK=3g \
    locust -f locustfile.py GrpcUser --headless
"""

from __future__ import annotations

import os
import time

from locust import HttpUser, User, task, between, events, LoadTestShape
from locust.runners import WorkerRunner

# Baca env — konsisten dengan cara task files membaca SCENARIO/NETWORK
SCENARIO    = os.environ.get("SCENARIO",    "load")
NETWORK     = os.environ.get("NETWORK",     "normal")
PROTO       = os.environ.get("PROTO",       "unknown")
GRPC_ADDR   = os.environ.get("GRPC_ADDR",  "127.0.0.1:8008")
REST_ADDR   = os.environ.get("REST_ADDR",  "http://127.0.0.1:8088")
RESULTS_DIR = os.environ.get("RESULTS_DIR","results")
EXP_NAME    = os.environ.get("EXP_NAME",   "percobaan")

from src.tasks.rest_task import analyze_skin as _rest_task
from src.tasks.grpc_task import analyze_skin as _grpc_task
from src.utils.grpc_client import make_channel, make_stub
from src.metrics.metrics import CsvListener

# ─── Load Shape ───────────────────────────────────────────────────────────────

_SHAPES: dict[str, list[tuple[int, int, float]]] = {
    "smoke": [
        (60, 1, 1),
        (10, 0, 1),
    ],
    "load": [
        (120, 10, 1),
        (300, 10, 0),
        (120, 0, 1),
    ],
    "stress": [
        (120, 20, 1),
        (300, 20, 0),
        (120, 40, 1),
        (300, 40, 0),
        (120, 0, 2),
    ],
    "spike": [
        (60, 10, 1),
        (30, 50, 5),
        (180, 50, 0),
        (60, 10, 2),
        (60, 0, 2),
    ],
    "soak": [
        (30, 15, 1),
        (1800, 15, 0),
        (30, 0, 2),
    ],
}

_shape = _SHAPES.get(SCENARIO, _SHAPES["load"])


class BenchmarkShape(LoadTestShape):
    def tick(self):
        run_time = self.get_run_time()
        elapsed = 0

        for duration, target, rate in _shape:
            elapsed += duration
            if run_time < elapsed:
                # Kalau target 0, benar-benar idle (jangan dipaksa jadi 1)
                if target <= 0:
                    return 0, 1
                return target, rate if rate > 0 else 1

        return None


# ─── REST User ────────────────────────────────────────────────────────────────

class RestUser(HttpUser):
    host = REST_ADDR
    wait_time = between(1, 3)

    @task
    def skin_analysis(self):
        _rest_task(self.client)


# ─── gRPC User ────────────────────────────────────────────────────────────────

class GrpcUser(User):
    wait_time = between(1, 3)
    _MAX_RETRIES = 3

    def on_start(self):
        """
        Buat channel sekali per user, di-reuse semua iterasi.

        Ini membuat simulasi lebih mirip 1 device = 1 koneksi.
        """
        from src.metrics.metrics import collector

        last_err = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                self._channel = make_channel(GRPC_ADDR)
                self._stub = make_stub(self._channel)
                return
            except Exception as e:
                last_err = e
                wait = attempt * 2
                print(f"[GrpcUser] on_start attempt {attempt} gagal: {e} — retry {wait}s")
                time.sleep(wait)

        # Semua retry gagal — record ke CSV agar tidak hilang
        err_msg = f"on_start gagal {self._MAX_RETRIES}x: {last_err}"
        print(f"[GrpcUser] {err_msg}")
        collector.record("grpc", SCENARIO, NETWORK, "grpc_req_failed", 1, error=err_msg)
        collector.record("grpc", SCENARIO, NETWORK, "grpc_req_success_rate", 0, error=err_msg)
        collector.record("grpc", SCENARIO, NETWORK, "iterations", 1, error=err_msg)
        self._stub = None
        self._channel = None

    def on_stop(self):
        # Tutup channel milik user ini sendiri
        if getattr(self, "_channel", None):
            self._channel.close()
        self._stub = None
        self._channel = None

    @task
    def skin_analysis(self):
        # Guard: on_start gagal → skip task (sudah di-record sebagai error)
        if not getattr(self, "_stub", None):
            return
        _grpc_task(self._stub, self.environment)


# ─── CSV Listener lifecycle ───────────────────────────────────────────────────

_csv_listener: CsvListener | None = None


@events.init.add_listener
def on_locust_init(environment, **_):
    global _csv_listener

    if isinstance(environment.runner, WorkerRunner):
        return  # distributed worker tidak buat CSV sendiri

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{EXP_NAME}_{NETWORK}_{SCENARIO}_{PROTO}_metrics.csv")

    _csv_listener = CsvListener(out_path=csv_path, interval=1.0)
    _csv_listener.start()

    print(f"[locustfile] Scenario  : {SCENARIO}")
    print(f"[locustfile] Network   : {NETWORK}")
    print(f"[locustfile] Proto     : {PROTO}")
    print(f"[locustfile] REST addr : {REST_ADDR}")
    print(f"[locustfile] gRPC addr : {GRPC_ADDR}")
    print(f"[locustfile] CSV out   : {csv_path}")


@events.quitting.add_listener
def on_quit(environment, **_):
    if _csv_listener:
        _csv_listener.stop()