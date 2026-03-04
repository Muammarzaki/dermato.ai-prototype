"""
src/metrics/metrics.py

Custom metrics yang setara dengan k6 Trend/Counter/Rate/Gauge.
Dikumpulkan per-request lalu ditulis ke CSV timestamped oleh CsvListener.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ─── Raw metric event (satu baris CSV) ───────────────────────────────────────

@dataclass
class MetricEvent:
    timestamp:    float   # unix epoch detik
    protocol:     str     # grpc | rest
    scenario:     str
    network:      str
    metric:       str     # nama metric
    value:        float
    tags:         str = ""


# ─── Thread-safe collector ────────────────────────────────────────────────────

class MetricsCollector:
    """
    Kumpulkan metric events dari semua greenlet secara thread-safe.
    CsvListener drain dan tulis ke file secara periodik.
    """

    def __init__(self):
        self._lock   = threading.Lock()
        self._buffer: list[MetricEvent] = []

    def record(self, protocol: str, scenario: str, network: str,
               metric: str, value: float, tags: str = "") -> None:
        ev = MetricEvent(
            timestamp = time.time(),
            protocol  = protocol,
            scenario  = scenario,
            network   = network,
            metric    = metric,
            value     = value,
            tags      = tags,
        )
        with self._lock:
            self._buffer.append(ev)

    def drain(self) -> list[MetricEvent]:
        with self._lock:
            out, self._buffer = self._buffer, []
        return out


# Singleton — di-import semua task
collector = MetricsCollector()


# ─── CSV Writer ───────────────────────────────────────────────────────────────

class CsvListener:
    """
    Locust event listener yang drain collector dan tulis ke CSV
    setiap interval detik, mirip format --out csv= di k6.

    Kolom CSV:
      timestamp, protocol, scenario, network, metric, value, tags
    """

    _HEADER = ["timestamp", "protocol", "scenario", "network", "metric", "value", "tags"]

    def __init__(self, out_path: str, interval: float = 2.0):
        self.path     = Path(out_path)
        self.interval = interval
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(self.path, "w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(self._HEADER)
        self._file.flush()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)
        self._flush()
        self._file.close()
        print(f"[csv] Metrics saved → {self.path}")

    def _run(self):
        while not self._stop.wait(self.interval):
            self._flush()

    def _flush(self):
        events = collector.drain()
        if not events:
            return
        for ev in events:
            self._writer.writerow([
                f"{ev.timestamp:.3f}",
                ev.protocol,
                ev.scenario,
                ev.network,
                ev.metric,
                f"{ev.value:.4f}",
                ev.tags,
            ])
        self._file.flush()