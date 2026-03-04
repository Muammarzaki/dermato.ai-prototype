"""
src/utils/grpc_client.py
Menggantikan src/utils/grpc.utils.js

Channel gRPC dibuat sekali per Locust user dan di-reuse
semua iterasi via HTTP/2 multiplexing — tidak ada reconnect
tiap request seperti di k6 sebelumnya.

Proto di-compile otomatis dari ../../../protobuf/skin_analyzer.proto
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

# ─── Gevent patch untuk grpc ──────────────────────────────────────────────────
# Locust menggunakan gevent (greenlet). grpc standard memakai thread native
# yang tidak kompatibel. Patch ini harus dipanggil SEBELUM import grpc apapun.
from gevent import monkey
monkey.patch_all()

import grpc.experimental.gevent as grpc_gevent
grpc_gevent.init_gevent()
# ─────────────────────────────────────────────────────────────────────────────

_lock         = threading.Lock()
_proto_ready  = False
_stub_class   = None
_pb2          = None


def _compile_proto() -> None:
    global _proto_ready, _stub_class, _pb2

    if _proto_ready:
        return

    with _lock:
        if _proto_ready:
            return

        proto_dir = Path(__file__).parents[3] / "protobuf"
        out_dir   = Path(__file__).parent / "_pb2"
        out_dir.mkdir(exist_ok=True)
        (out_dir / "__init__.py").touch()

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

        # Tambah parent ke sys.path agar import bekerja
        parent = str(out_dir.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)

        import importlib
        _pb2       = importlib.import_module("_pb2.skin_analyzer_pb2")
        pb2_grpc   = importlib.import_module("_pb2.skin_analyzer_pb2_grpc")
        _stub_class = pb2_grpc.SkinAnalysisServiceStub
        _proto_ready = True


def make_channel(address: str):
    """
    Buat insecure gRPC channel dengan HTTP/2 keepalive.
    Panggil sekali di on_start(), bukan di tiap task.
    """
    import grpc
    _compile_proto()

    return grpc.insecure_channel(
        address,
        options=[
            ("grpc.max_send_message_length",        -1),
            ("grpc.max_receive_message_length",      -1),
            ("grpc.keepalive_time_ms",           20_000),
            ("grpc.keepalive_timeout_ms",        10_000),
            ("grpc.keepalive_permit_without_calls",   1),
        ],
    )


def make_stub(channel):
    _compile_proto()
    return _stub_class(channel)


def get_pb2():
    _compile_proto()
    return _pb2