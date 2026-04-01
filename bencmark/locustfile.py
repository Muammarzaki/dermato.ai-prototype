"""
locustfile.py

Entry point Locust untuk benchmark gRPC vs REST.

Jalankan:
  SCENARIO=load NETWORK=normal \
    locust -f locustfile.py RestUser --headless -u 10 -r 2 --run-time 9m

  SCENARIO=smoke NETWORK=3g \
    locust -f locustfile.py GrpcUser --headless

Panduan metrik untuk analisis hasil:

  Rumusan 1 — Success rate & completion time:
    success_rate = df[df.metric=='grpc_req_success'].value.mean()
                   (gunakan 'rest_req_success' untuk REST)
    completion_time = kolom 'value' pada baris metric='grpc_req_duration'
    Hanya _req_duration yang bisa dibandingkan langsung antara gRPC dan REST.

  Rumusan 2 — Payload efficiency:
    grpc_data_sent = proto_frame_size (pre-computed)
    rest_data_sent = multipart wire size (pre-computed)

  Rumusan 4 — Konkurensi & recovery time:
    Throughput: count(iterations per detik) dari kolom timestamp
    Konkurensi: grpc_active_streams / rest_active_requests per timestamp
    Recovery time: analisis post-process dari timestamp success 0 → 1

  Label warning (terpisah dari success rate):
    grpc_label_warning / rest_label_warning — monitor tapi jangan campur
    dengan success rate transmisi.
"""

from __future__ import annotations

import os
import time

from locust import HttpUser, User, task, between, events, LoadTestShape
from locust.runners import WorkerRunner

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
        elapsed  = 0
        for duration, target, rate in _shape:
            elapsed += duration
            if run_time < elapsed:
                if target <= 0:
                    return 0, 1
                return target, rate if rate > 0 else 1
        return None


# ─── REST User ────────────────────────────────────────────────────────────────

class RestUser(HttpUser):
    """
    Satu VU = satu HTTP session dengan keep-alive (default requests.Session).
    Koneksi persistent antar request dalam session yang sama —
    setara dengan gRPC yang mempertahankan channel selama think time.
    """
    host      = REST_ADDR
    wait_time = between(1, 3)

    @task
    def skin_analysis(self):
        _rest_task(self.client)


# ─── gRPC User ────────────────────────────────────────────────────────────────

class GrpcUser(User):
    """
    Satu VU = satu gRPC channel (insecure, persistent).
    Channel tidak di-share antar VU — setiap VU merepresentasikan satu device.
    Channel tetap hidup selama think time (between 1–3s) — setara dengan
    HTTP keep-alive di RestUser.
    """
    wait_time    = between(1, 3)
    _MAX_RETRIES = 3

    def on_start(self):
        from src.metrics.metrics import collector

        last_err = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                self._channel = make_channel(GRPC_ADDR)
                self._stub    = make_stub(self._channel)
                return
            except Exception as e:
                last_err = e
                wait     = attempt * 2
                print(f"[GrpcUser] on_start attempt {attempt} gagal: {e} — retry {wait}s")
                time.sleep(wait)

        err_msg = f"on_start gagal {self._MAX_RETRIES}x: {last_err}"
        print(f"[GrpcUser] {err_msg}")
        collector.record("grpc", SCENARIO, NETWORK, "grpc_req_success", 0, error=err_msg)
        collector.record("grpc", SCENARIO, NETWORK, "iterations",       1, error=err_msg)
        self._stub    = None
        self._channel = None

    def on_stop(self):
        if getattr(self, "_channel", None):
            self._channel.close()
        self._stub    = None
        self._channel = None

    @task
    def skin_analysis(self):
        if not getattr(self, "_stub", None):
            return
        _grpc_task(self._stub, self.environment)


# ─── CSV Listener lifecycle ───────────────────────────────────────────────────

_csv_listener: CsvListener | None = None


@events.init.add_listener
def on_locust_init(environment, **_):
    global _csv_listener

    if isinstance(environment.runner, WorkerRunner):
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(
        RESULTS_DIR,
        f"{EXP_NAME}_{NETWORK}_{SCENARIO}_{PROTO}_metrics.csv"
    )

    # interval=None → CsvListener pilih otomatis berdasarkan SCENARIO
    _csv_listener = CsvListener(out_path=csv_path, interval=None)
    _csv_listener.start()

    # Import dataset untuk tampilkan ringkasan pre-computed sizes
    from src.config.config import TEST_DATASET
    print(f"\n[locustfile] ── Benchmark Config ──────────────────────")
    print(f"[locustfile] Scenario  : {SCENARIO}")
    print(f"[locustfile] Network   : {NETWORK}")
    print(f"[locustfile] Proto     : {PROTO}")
    print(f"[locustfile] REST addr : {REST_ADDR}")
    print(f"[locustfile] gRPC addr : {GRPC_ADDR}")
    print(f"[locustfile] CSV out   : {csv_path}")
    print(f"[locustfile] ── Dataset ({len(TEST_DATASET)} gambar) ─────────────────")
    for tc in TEST_DATASET:
        print(
            f"[locustfile]   {tc['filename']}: "
            f"raw={len(tc['data'])//1024}KB "
            f"wire={tc['wire_size']//1024}KB "
            f"proto={tc['proto_frame_size']//1024}KB"
        )
    print(f"[locustfile] ── Metrik ─────────────────────────────────")
    print(f"[locustfile] Success rate  : df[metric=='grpc_req_success'].value.mean()")
    print(f"[locustfile] Payload gRPC  : metric='grpc_data_sent' (proto_frame_size)")
    print(f"[locustfile] Payload REST  : metric='rest_data_sent'  (multipart wire)")
    print(f"[locustfile] Recovery time : post-process dari timestamp req_success=0→1")
    print(f"[locustfile] ────────────────────────────────────────────\n")


@events.quitting.add_listener
def on_quit(environment, **_):
    if _csv_listener:
        _csv_listener.stop()