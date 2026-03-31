"""
src/config/config.py
"""

import os
import hashlib
from pathlib import Path

# ─── Addresses ───────────────────────────────────────────────────────────────
GRPC_ADDR = os.environ.get("GRPC_ADDR", "127.0.0.1:8008")
REST_ADDR = os.environ.get("REST_ADDR", "http://127.0.0.1:8088")

# ─── Transfer ────────────────────────────────────────────────────────────────
CHUNK_SIZE = 64 * 1024  # 64 KB
TIMEOUT    = 1200

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


def _load(filename: str, expected_label: str) -> dict | None:
    path = _IMAGE_DIR / filename
    if not path.exists():
        print(f"[config] WARNING: skip {path}")
        return None
    data   = path.read_bytes()
    digest = hashlib.sha256(data)
    return {
        "filename":       filename,
        "expected_label": expected_label,
        "data":           data,
        "hash_hex":       digest.hexdigest(),  # REST — string hex
        "hash_bytes":     digest.digest(),     # gRPC — raw bytes
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

print(f"[config] {len(TEST_DATASET)} gambar dimuat dari {_IMAGE_DIR}")