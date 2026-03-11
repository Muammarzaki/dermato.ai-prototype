"""
src/utils/grpc_client.py

Optimasi utama: shared channel singleton per alamat server.

Kenapa shared channel lebih baik dari per-VU channel:
  - HTTP/2 multiplexing: 1 channel bisa handle N concurrent stream sekaligus
  - TLS/TCP handshake hanya 1x di awal, bukan 50x (untuk 50 VU)
  - Connection pool lebih efisien di sisi OS (1 socket vs 50 socket)
  - Ini adalah best practice resmi gRPC: https://grpc.io/docs/guides/performance/
    "reuse a single channel for all requests from a client process"

Sebelumnya: 50 VU = 50 channel = 50 TCP connection = 50x handshake overhead
Sekarang  : 50 VU = 1 channel  = 1 TCP connection  = 50 concurrent HTTP/2 stream
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

# ─── Gevent patch — harus sebelum import grpc apapun ─────────────────────────
from gevent import monkey
monkey.patch_all()

import grpc.experimental.gevent as grpc_gevent
grpc_gevent.init_gevent()
# ─────────────────────────────────────────────────────────────────────────────

_lock        = threading.Lock()
_proto_ready = False
_stub_class  = None
_pb2         = None

# ─── Shared channel registry ─────────────────────────────────────────────────
# key: address string → value: grpc.Channel
# Satu channel per alamat server, di-share semua GrpcUser
_channels: dict[str, object] = {}
_channel_lock = threading.Lock()


def _compile_proto() -> None:
    global _proto_ready, _stub_class, _pb2

    if _proto_ready:
        return

    with _lock:
        if _proto_ready:
            return

        proto_dir  = Path(__file__).parents[3] / "protobuf"
        out_dir    = Path(__file__).parent / "_pb2"
        out_dir.mkdir(exist_ok=True)

        out_dir_str = str(out_dir)
        if out_dir_str not in sys.path:
            sys.path.insert(0, out_dir_str)

        proto_file = proto_dir / "skin_analyzer.proto"
        if not proto_file.exists():
            raise FileNotFoundError(
                f"Proto tidak ditemukan: {proto_file}\n"
                "Pastikan folder protobuf/ ada di root project."
            )

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
            if rc != 0:
                raise RuntimeError("Kompilasi protoc gagal")

        import importlib
        _pb2        = importlib.import_module("skin_analyzer_pb2")
        pb2_grpc    = importlib.import_module("skin_analyzer_pb2_grpc")
        _stub_class = pb2_grpc.SkinAnalysisServiceStub
        _proto_ready = True


def get_shared_channel(address: str):
    """
    Ambil atau buat shared channel untuk address tertentu.

    Thread-safe: double-checked locking supaya hanya 1 channel
    yang dibuat meski dipanggil dari 50 VU sekaligus.

    Channel ini TIDAK boleh di-close oleh individual VU —
    lifecycle-nya dikelola oleh proses Locust.
    """
    import grpc

    _compile_proto()

    # Fast path — channel sudah ada
    if address in _channels:
        return _channels[address]

    # Slow path — buat channel baru dengan lock
    with _channel_lock:
        if address in _channels:
            return _channels[address]

        channel = grpc.insecure_channel(
            address,
            options=[
                # Ukuran pesan tidak dibatasi (gambar bisa 4MB+)
                ("grpc.max_send_message_length",    -1),
                ("grpc.max_receive_message_length", -1),

                # Keepalive: pastikan connection tetap hidup selama benchmark
                ("grpc.keepalive_time_ms",          20_000),
                ("grpc.keepalive_timeout_ms",       10_000),
                ("grpc.keepalive_permit_without_calls", 1),

                # HTTP/2 concurrent stream limit per connection
                # Default gRPC = 100, naikkan ke 200 untuk spike test (50 VU × buffer)
                ("grpc.max_concurrent_streams",        200),

                # Initial window size HTTP/2 — naikkan untuk throughput lebih tinggi
                # Default = 65535 bytes (64KB), naikkan ke 4MB
                # Ini memungkinkan server kirim response tanpa tunggu window update
                ("grpc.http2.initial_window_size",   4 * 1024 * 1024),

                # BDP (Bandwidth-Delay Product) probing — aktifkan untuk
                # auto-tune window size berdasarkan kondisi jaringan aktual
                ("grpc.http2.bdp_probe",              1),
            ],
        )
        _channels[address] = channel
        print(f"[grpc_client] Shared channel dibuat → {address}")
        return channel


def make_stub(channel):
    """Buat stub dari channel yang diberikan."""
    _compile_proto()
    return _stub_class(channel)


def get_pb2():
    _compile_proto()
    return _pb2


# ─── Legacy compatibility (masih bisa dipanggil tapi return shared channel) ──
def make_channel(address: str):
    """
    Deprecated: gunakan get_shared_channel().
    Dipertahankan untuk kompatibilitas — return shared channel,
    bukan channel baru.
    """
    return get_shared_channel(address)