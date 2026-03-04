"""
locustfile.py — dipanggil dari dalam folder bencmark/

Contoh:
  SCENARIO=load GRPC_ADDR=127.0.0.1:8008 \
    locust -f locustfile.py GrpcUser --headless

  SCENARIO=spike NETWORK=poor GRPC_ADDR=10.0.0.1:8008 REST_ADDR=http://10.0.0.1:8088 \
    locust -f locustfile.py RestUser GrpcUser --headless \
    --csv=results/spike_poor
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from locust import HttpUser, User, task, between, LoadTestShape, events
from locust.runners import WorkerRunner, MasterRunner

from src.config.config import GRPC_ADDR, REST_ADDR, SCENARIOS
from src.tasks.rest_task  import analyze_skin as rest_analyze
from src.tasks.grpc_task  import analyze_skin as grpc_analyze
from src.utils.grpc_client import make_channel, make_stub
from src.metrics.metrics   import CsvListener

# ─── Env ─────────────────────────────────────────────────────────────────────
SCENARIO   = os.environ.get("SCENARIO",   "load")
NETWORK    = os.environ.get("NETWORK",    "normal")
EXP_NAME   = os.environ.get("EXP_NAME",  "experiment")
RESULTS_DIR= os.environ.get("RESULTS_DIR","results")

_shape = SCENARIOS.get(SCENARIO, SCENARIOS["load"])

# ─── CSV output path (sama pola dengan k6: proto_net_scenario_ts.csv) ────────
_ts       = time.strftime("%Y%m%d_%H%M%S")
_csv_path = str(Path(RESULTS_DIR) / f"{EXP_NAME}_{NETWORK}_{SCENARIO}_{_ts}_metrics.csv")

_csv_listener: CsvListener | None = None


# ─── Load Shape ───────────────────────────────────────────────────────────────

class BenchmarkShape(LoadTestShape):
    def tick(self):
        run_time = self.get_run_time()
        elapsed  = 0
        for duration, target, rate in _shape:
            elapsed += duration
            if run_time < elapsed:
                return (target, rate or 1)
        return None


# ─── Users ────────────────────────────────────────────────────────────────────

class RestUser(HttpUser):
    host      = REST_ADDR
    wait_time = between(1, 3)

    @task
    def skin_analysis(self):
        rest_analyze(self.client)


class GrpcUser(User):
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


# ─── Lifecycle hooks ─────────────────────────────────────────────────────────

@events.init.add_listener
def on_init(environment, **_):
    global _csv_listener

    if isinstance(environment.runner, WorkerRunner):
        return

    print(f"[locust] scenario    : {SCENARIO}")
    print(f"[locust] network     : {NETWORK}")
    print(f"[locust] REST        : {REST_ADDR}")
    print(f"[locust] gRPC        : {GRPC_ADDR}")
    print(f"[locust] shape       : {_shape}")
    print(f"[locust] metrics csv : {_csv_path}")

    _csv_listener = CsvListener(_csv_path, interval=2.0)
    _csv_listener.start()


@events.quitting.add_listener
def on_quit(environment, **_):
    if _csv_listener:
        _csv_listener.stop()