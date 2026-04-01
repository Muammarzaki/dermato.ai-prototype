"""
src/metrics/metrics.py

Custom metrics yang setara dengan k6 Trend/Counter/Rate/Gauge.
Dikumpulkan per-request lalu ditulis ke CSV timestamped oleh CsvListener.

Kolom CSV:
  timestamp, protocol, scenario, network, metric, value, tags, error

Naming convention metrik (penting untuk analisis):
  *_req_duration   → completion time per request dalam ms (rumusan 1)
                     Gunakan ini sebagai "waktu penyelesaian transmisi".
                     grpc_req_duration vs rest_req_duration comparable langsung.

  *_req_success    → 1 (sukses) atau 0 (gagal) per request — BUKAN rate.
                     Hitung success rate saat analisis:
                     df[df.metric=='grpc_req_success'].value.mean()
                     Denominator yang benar: count rows dengan metric='iterations'

  *_data_sent      → application-layer payload size dalam bytes (rumusan 2)
                     grpc: proto_frame_size (protobuf serialized, pre-computed)
                     rest: multipart wire size (boundary + headers, pre-computed)
                     Keduanya TIDAK termasuk HTTP transport headers — simetris.

  *_label_warning  → 1 jika label model AI tidak sesuai expected.
                     TIDAK mempengaruhi success rate — dipisah agar akurasi
                     model AI server tidak mencemari metrik transmisi (rumusan 1).

  *_active_*       → konkurensi snapshot per-event (rumusan 4)
                     Hanya valid untuk single-node deployment.

CsvListener:
  - Interval flush adaptif: lebih cepat di smoke/load, lebih lambat di soak
    agar I/O tidak menjadi bottleneck di skenario panjang.
  - Size guard: print warning jika buffer > 5000 event sebelum di-drain.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_SCENARIO = os.environ.get("SCENARIO", "load")

# Interval flush adaptif:
# smoke/load  → 2s  : data granular, durasi pendek, I/O aman
# stress/spike → 3s : volume event tinggi, kurangi frekuensi flush
# soak        → 5s  : 30 menit, minimalkan akumulasi disk I/O overhead
_FLUSH_INTERVALS = {
    "smoke":  2.0,
    "load":   2.0,
    "stress": 3.0,
    "spike":  3.0,
    "soak":   5.0,
}
_DEFAULT_INTERVAL    = 2.0
_BUFFER_WARN_THRESHOLD = 5_000


# ─── Raw metric event ────────────────────────────────────────────────────────

@dataclass
class MetricEvent:
    timestamp: float
    protocol:  str     # grpc | rest
    scenario:  str
    network:   str
    metric:    str
    value:     float
    tags:      str = ""
    error:     str = ""


# ─── Thread-safe collector ────────────────────────────────────────────────────

class MetricsCollector:
    def __init__(self):
        self._lock   = threading.Lock()
        self._buffer: list[MetricEvent] = []

    def record(
        self,
        protocol: str,
        scenario: str,
        network:  str,
        metric:   str,
        value:    float,
        tags:     str = "",
        error:    str = "",
    ) -> None:
        ev = MetricEvent(
            timestamp = time.time(),
            protocol  = protocol,
            scenario  = scenario,
            network   = network,
            metric    = metric,
            value     = value,
            tags      = tags,
            error     = error,
        )
        with self._lock:
            self._buffer.append(ev)

    def drain(self) -> list[MetricEvent]:
        with self._lock:
            out, self._buffer = self._buffer, []
        return out

    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)


# Singleton — di-import semua task
collector = MetricsCollector()


# ─── CSV Writer ───────────────────────────────────────────────────────────────

class CsvListener:
    """
    Drain collector dan tulis ke CSV setiap interval detik.

    Interval flush dipilih adaptif berdasarkan SCENARIO env var agar
    overhead I/O tidak mempengaruhi hasil benchmark di skenario panjang (soak).
    """

    _HEADER = [
        "timestamp",
        "protocol",
        "scenario",
        "network",
        "metric",
        "value",
        "tags",
        "error",
    ]

    def __init__(self, out_path: str, interval: float | None = None):
        self.path     = Path(out_path)
        self.interval = interval if interval is not None \
                        else _FLUSH_INTERVALS.get(_SCENARIO, _DEFAULT_INTERVAL)
        self._stop    = threading.Event()
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="CsvListener"
        )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(self.path, "w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(self._HEADER)
        self._file.flush()

        print(
            f"[csv] Listener started — interval={self.interval}s "
            f"scenario={_SCENARIO} path={self.path}"
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=15)
        self._flush()
        self._file.close()
        print(f"[csv] Metrics saved → {self.path}")

    def _run(self):
        while not self._stop.wait(self.interval):
            self._check_buffer_health()
            self._flush()

    def _check_buffer_health(self) -> None:
        size = collector.buffer_size()
        if size > _BUFFER_WARN_THRESHOLD:
            print(
                f"[csv] WARNING: buffer size={size} events melebihi threshold "
                f"{_BUFFER_WARN_THRESHOLD} — disk I/O mungkin tertinggal. "
                f"Pertimbangkan mengurangi jumlah VU atau menambah interval flush."
            )

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
                ev.error,
            ])
        self._file.flush()