"""
locustfile.py  —  dipanggil dari dalam folder /benchmark

Menggantikan benchmark.sh + grpc.test.js + rest.test.js

Contoh jalankan (dari dalam folder benchmark/):

  # REST saja
  SCENARIO=load REST_ADDR=http://10.128.0.2:8088 \
    locust -f locustfile.py RestUser --headless

  # gRPC saja
  SCENARIO=stress GRPC_ADDR=10.128.0.2:8008 \
    locust -f locustfile.py GrpcUser --headless

  # Keduanya sekaligus
  SCENARIO=spike GRPC_ADDR=10.128.0.2:8008 REST_ADDR=http://10.128.0.2:8088 \
    locust -f locustfile.py RestUser GrpcUser --headless

  # Simpan CSV
  locust -f locustfile.py RestUser GrpcUser --headless \
    --csv=results/run_$(date +%Y%m%d_%H%M%S)

  # Web UI
  locust -f locustfile.py
"""

from __future__ import annotations

import os

from locust import HttpUser, User, task, between, LoadTestShape, events
from locust.runners import WorkerRunner

from src.config.config import GRPC_ADDR, REST_ADDR, SCENARIOS
from src.tasks.rest_task import analyze_skin as rest_analyze
from src.tasks.grpc_task import analyze_skin as grpc_analyze
from src.utils.grpc_client import make_channel, make_stub

# ─── Active scenario ─────────────────────────────────────────────────────────
SCENARIO = os.environ.get("SCENARIO", "load")
_shape   = SCENARIOS.get(SCENARIO, SCENARIOS["load"])


# ─── Load shape ───────────────────────────────────────────────────────────────

class BenchmarkShape(LoadTestShape):
    """
    Menerjemahkan SCENARIOS dict ke Locust LoadTestShape.
    Ekuivalen dengan stages di k6 config.js.
    """
    def tick(self):
        run_time = self.get_run_time()
        elapsed  = 0
        for duration, target, rate in _shape:
            elapsed += duration
            if run_time < elapsed:
                return (target, rate or 1)
        return None  # stop test


# ─── REST User ────────────────────────────────────────────────────────────────

class RestUser(HttpUser):
    """
    Satu user = satu HTTP session dengan connection keep-alive.
    Ekuivalen dengan rest.test.js.
    """
    host      = REST_ADDR
    wait_time = between(1, 3)

    @task
    def skin_analysis(self):
        rest_analyze(self.client)


# ─── gRPC User ────────────────────────────────────────────────────────────────

class GrpcUser(User):
    """
    Satu user = satu gRPC channel HTTP/2 persistent.
    Ekuivalen dengan grpc.test.js — tanpa reconnect tiap iterasi,
    dan tanpa b64encode karena binary dikirim langsung.
    """
    wait_time = between(1, 3)

    def on_start(self):
        self._channel = make_channel(GRPC_ADDR)
        self._stub    = make_stub(self._channel)

    def on_stop(self):
        if hasattr(self, "_channel"):
            self._channel.close()

    @task
    def skin_analysis(self):
        grpc_analyze(self._stub, self.environment)


# ─── Startup info ─────────────────────────────────────────────────────────────

@events.init.add_listener
def on_init(environment, **_):
    if not isinstance(environment.runner, WorkerRunner):
        print(f"[locust] scenario : {SCENARIO}")
        print(f"[locust] REST     : {REST_ADDR}")
        print(f"[locust] gRPC     : {GRPC_ADDR}")
        print(f"[locust] shape    : {_shape}")
