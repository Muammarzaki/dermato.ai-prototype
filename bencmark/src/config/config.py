"""
src/config/config.py

Semua nilai yang butuh komputasi berat (wire size multipart, protobuf frame size)
dihitung SEKALI saat module di-load — bukan saat benchmark berjalan.

Ini penting untuk fairness benchmark:
  - Overhead PreparedRequest.prepare() (~5–15ms untuk gambar 4MB) tidak masuk
    ke measurement window rest_req_duration.
  - Overhead SerializeToString() per chunk tidak masuk ke grpc_req_sending.

Field tambahan di tiap entry TEST_DATASET:
  wire_size        → ukuran multipart/form-data body aktual termasuk boundary
                     dan header per-field. Dipakai langsung di rest_task.py
                     sebagai rest_data_sent. (rumusan masalah 2)

  proto_frame_size → total ukuran serialized protobuf semua chunk + meta message,
                     dihitung dengan chunk size CHUNK_SIZE_BYTES. Dipakai di
                     grpc_task.py sebagai grpc_data_sent untuk paritas dengan
                     wire_size REST. (rumusan masalah 2)

Catatan metodologis (dokumentasikan di laporan):
  Kedua ukuran ini adalah application-layer payload size:
  - REST : body multipart (boundary + headers per-field + file bytes)
  - gRPC : protobuf frames (message framing + field tags + varint prefix)
  Yang TIDAK terhitung di kedua sisi: HTTP transport headers (HTTP/1.1 request
  line + headers untuk REST; HTTP/2 HEADERS frame + HPACK untuk gRPC).
  Ini limitasi yang SIMETRIS dan harus dinyatakan di metodologi penelitian.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

# ─── Addresses ───────────────────────────────────────────────────────────────
GRPC_ADDR = os.environ.get("GRPC_ADDR", "127.0.0.1:8008")
REST_ADDR = os.environ.get("REST_ADDR", "http://127.0.0.1:8088")

# ─── Transfer ────────────────────────────────────────────────────────────────
# Chunk size harus konsisten antara config (untuk pre-compute) dan grpc_task
CHUNK_SIZE_BYTES = 256 * 1024   # 256 KB — source of truth, dipakai grpc_task.py
TIMEOUT          = 1200

# ─── Metadata ────────────────────────────────────────────────────────────────
METADATA = {
    "user_id":    "user-locust-test",
    "image_type": "image/jpeg",
    "meta_tags": {
        "source":      "locust-load-test",
        "environment": "testing",
    },
}

# ─── Scenarios ───────────────────────────────────────────────────────────────
SCENARIOS = {
    "smoke":  [(60, 1, 1), (10, 0, 1)],
    "load":   [(120, 10, 1), (300, 10, 0), (120, 0, 1)],
    "stress": [(120, 20, 1), (300, 20, 0), (120, 40, 1), (300, 40, 0), (120, 0, 2)],
    "spike":  [(60, 10, 1), (30, 50, 5), (180, 50, 0), (60, 10, 2), (60, 0, 2)],
    "soak":   [(30, 15, 1), (1800, 15, 0), (30, 0, 2)],
}

# ─── Test images ─────────────────────────────────────────────────────────────
_IMAGE_DIR = Path(__file__).parents[2] / "test-images"

_IMAGE_MANIFEST = [
    ("tahi_lalat_1.5mb.jpg", "Eczema"),
    ("tahi_lalat_2mb.jpg",   "Cacar Air"),
    ("tahi_lalat_2.5mb.jpg", "Cacar Air"),
    ("tahi_lalat_3.7mb.jpg", "Cacar Air"),
    ("tahi_lalat_4mb.jpg",   "Cacar Air"),
]

# Seed tetap agar urutan dataset reproducible antar-run
DATASET_SEED = int(os.environ.get("DATASET_SEED", "20260401"))


def _compute_multipart_wire_size(filename: str, data: bytes) -> int:
    """
    Hitung ukuran body multipart/form-data yang akan dikirim ke server.

    Menggunakan requests.PreparedRequest agar boundary, Content-Disposition,
    dan Content-Type header per-field ikut terhitung. Dipanggil sekali saat
    load — bukan di dalam loop benchmark.

    Fallback ke Content-Length header jika body bukan bytes murni.
    """
    import requests as _req

    _files = {"file": (filename, data, "image/jpeg")}
    _data  = {
        "user_id":       METADATA["user_id"],
        "client_sha256": hashlib.sha256(data).hexdigest(),
        "metadata":      json.dumps(METADATA["meta_tags"]),
    }
    try:
        prepared = _req.Request("POST", "http://x", files=_files, data=_data).prepare()
        body = prepared.body
        if isinstance(body, bytes):
            return len(body)
        cl = prepared.headers.get("Content-Length")
        if cl:
            return int(cl)
        # generator body — baca sekali untuk ukur (hanya saat load, bukan benchmark)
        if hasattr(body, "read"):
            return len(body.read())
    except Exception as e:
        print(f"[config] _compute_multipart_wire_size fallback untuk {filename}: {e}")

    # Fallback akhir: raw file size + estimasi overhead boundary
    return len(data) + 512


def _compute_proto_frame_size(data: bytes, hash_bytes: bytes) -> int:
    """
    Hitung total ukuran serialized protobuf yang dikirim dalam satu streaming call.

    Mencakup:
      1. meta message (ImageInfo): field tags + varint + string content
      2. setiap chunk message: field tag + varint length prefix + chunk bytes

    Ini adalah application-layer frame size — setara dengan multipart wire size
    di sisi REST untuk perbandingan payload efficiency (rumusan masalah 2).

    Dipanggil sekali saat load — tidak ada overhead di dalam loop benchmark.
    """
    # Import lazy — hanya tersedia setelah protobuf di-compile
    # Jika protobuf belum tersedia saat config di-load (misalnya saat unit test),
    # gunakan estimasi: len(data) + overhead per-chunk
    try:
        import sys
        from pathlib import Path as _Path

        pb2_dir = str(_Path(__file__).parents[2] / "src" / "utils" / "_pb2")
        if pb2_dir not in sys.path:
            sys.path.insert(0, pb2_dir)

        import skin_analyzer_pb2 as _pb2

        # Meta message
        meta = _pb2.AnalyzeSkinRequest(
            info=_pb2.ImageInfo(
                user_id=METADATA["user_id"],
                image_type=METADATA["image_type"],
                client_sha256=hash_bytes,
                metadata={
                    **METADATA["meta_tags"],
                    "file_size": str(len(data)),
                },
            )
        )
        total = len(meta.SerializeToString())

        # Chunk messages
        offset = 0
        data_view = memoryview(data)
        while offset < len(data):
            end   = min(offset + CHUNK_SIZE_BYTES, len(data))
            chunk = _pb2.AnalyzeSkinRequest(chunk=bytes(data_view[offset:end]))
            total += len(chunk.SerializeToString())
            offset = end

        return total

    except Exception as e:
        print(f"[config] _compute_proto_frame_size fallback (protobuf belum tersedia): {e}")
        # Estimasi: overhead protobuf per chunk ~5 bytes (field tag 1 + varint 4)
        n_chunks  = (len(data) + CHUNK_SIZE_BYTES - 1) // CHUNK_SIZE_BYTES
        overhead  = n_chunks * 5 + 128   # 128 bytes estimasi meta message
        return len(data) + overhead


def _load(filename: str, expected_label: str) -> dict | None:
    path = _IMAGE_DIR / filename
    if not path.exists():
        print(f"[config] WARNING: skip {path}")
        return None

    data        = path.read_bytes()
    digest      = hashlib.sha256(data)
    hash_hex    = digest.hexdigest()
    hash_bytes  = digest.digest()

    wire_size        = _compute_multipart_wire_size(filename, data)
    proto_frame_size = _compute_proto_frame_size(data, hash_bytes)

    print(
        f"[config] {filename}: "
        f"raw={len(data)/1024:.0f}KB "
        f"multipart_wire={wire_size/1024:.0f}KB "
        f"proto_frame={proto_frame_size/1024:.0f}KB"
    )

    return {
        "filename":         filename,
        "expected_label":   expected_label,
        "data":             data,
        "hash_hex":         hash_hex,
        "hash_bytes":       hash_bytes,
        # Pre-computed sizes — dipakai langsung di task files, nol overhead saat benchmark
        "wire_size":        wire_size,        # REST multipart body size (rumusan 2)
        "proto_frame_size": proto_frame_size,
    }


TEST_DATASET: list[dict] = [
    tc for tc in (_load(fn, lbl) for fn, lbl in _IMAGE_MANIFEST)
    if tc is not None
]

if not TEST_DATASET:
    raise RuntimeError(
        f"[config] Tidak ada gambar ditemukan di {_IMAGE_DIR}.\n"
        "Pastikan folder test-images/ ada di root project."
    )

_rng = random.Random(DATASET_SEED)
_rng.shuffle(TEST_DATASET)

print(f"[config] {len(TEST_DATASET)} gambar dimuat dari {_IMAGE_DIR} | seed={DATASET_SEED}")