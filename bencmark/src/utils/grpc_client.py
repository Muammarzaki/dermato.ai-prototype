"""
src/utils/grpc_client.py

Versi Akademik (Fair Benchmarking)
- Menggunakan Channel Pool agar setara dengan REST Connection Pooling.
- Parameter statis (tidak mendeteksi simulasi jaringan).
- Mengandalkan BDP Probe bawaan HTTP/2 untuk adaptasi jaringan.
"""

from __future__ import annotations

import os
import sys
import threading
import itertools
import json
from pathlib import Path

from gevent import monkey
monkey.patch_all()

import grpc.experimental.gevent as grpc_gevent
grpc_gevent.init_gevent()

_lock        = threading.Lock()
_proto_ready = False
_stub_class  = None
_pb2         = None

_channel_pools: dict[str, list] = {}
_pool_iterators: dict[str, itertools.cycle] = {}
_channel_lock = threading.Lock()

# Pool size adil untuk mendistribusikan beban 50 VU
POOL_SIZE = int(os.environ.get("GRPC_CHANNEL_POOL_SIZE", "10"))

def _compile_proto() -> None:
    global _proto_ready, _stub_class, _pb2
    if _proto_ready: return

    with _lock:
        if _proto_ready: return
        proto_dir  = Path(__file__).parents[3] / "protobuf"
        out_dir    = Path(__file__).parent / "_pb2"
        out_dir.mkdir(exist_ok=True)

        out_dir_str = str(out_dir)
        if out_dir_str not in sys.path:
            sys.path.insert(0, out_dir_str)

        proto_file = proto_dir / "skin_analyzer.proto"
        pb2_file = out_dir / "skin_analyzer_pb2.py"

        if not pb2_file.exists() or proto_file.stat().st_mtime > pb2_file.stat().st_mtime:
            from grpc_tools import protoc
            rc = protoc.main([
                "grpc_tools.protoc",
                f"-I{proto_dir}",
                f"--python_out={out_dir}",
                f"--grpc_python_out={out_dir}",
                str(proto_file),
            ])
            if rc != 0: raise RuntimeError("Kompilasi protoc gagal")

        import importlib
        _pb2        = importlib.import_module("skin_analyzer_pb2")
        pb2_grpc    = importlib.import_module("skin_analyzer_pb2_grpc")
        _stub_class = pb2_grpc.SkinAnalysisServiceStub
        _proto_ready = True

def get_shared_channel(address: str):
    import grpc
    _compile_proto()

    if address in _pool_iterators:
        return next(_pool_iterators[address])

    with _channel_lock:
        if address not in _pool_iterators:
            print(f"[grpc_client] Inisialisasi Channel Pool untuk {address} | Pool: {POOL_SIZE}")

            # Service config untuk fitur bawaan gRPC (Retry transparan)
            service_config = json.dumps({
                "methodConfig": [{
                    "name": [{}],
                    "retryPolicy": {
                        "maxAttempts": 3,
                        "initialBackoff": "0.1s",
                        "maxBackoff": "2s",
                        "backoffMultiplier": 2,
                        "retryableStatusCodes": ["UNAVAILABLE"]
                    }
                }]
            })

            pool = []
            for i in range(POOL_SIZE):
                channel = grpc.insecure_channel(
                    address,
                    options=[
                        ("grpc.max_send_message_length",    -1),
                        ("grpc.max_receive_message_length", -1),

                        ("grpc.keepalive_time_ms",          20_000),
                        ("grpc.keepalive_timeout_ms",       10_000),
                        ("grpc.keepalive_permit_without_calls", 1),

                        # Konfigurasi Statis: Biarkan BDP probe yang bekerja keras beradaptasi
                        # 1MB adalah jalan tengah yang logis untuk payload gambar
                        ("grpc.max_concurrent_streams",      100),
                        ("grpc.http2.initial_window_size",   1 * 1024 * 1024),
                        ("grpc.http2.bdp_probe",             1),

                        ("grpc.enable_retries",              1),
                        ("grpc.service_config",              service_config),
                        ("grpc.initial_reconnect_backoff_ms", 100),
                        ("grpc.max_reconnect_backoff_ms",    3000),
                    ],
                )
                pool.append(channel)

            _channel_pools[address] = pool
            _pool_iterators[address] = itertools.cycle(pool)

        return next(_pool_iterators[address])

def make_stub(channel):
    _compile_proto()
    return _stub_class(channel)

def get_pb2():
    _compile_proto()
    return _pb2

def make_channel(address: str):
    return get_shared_channel(address)