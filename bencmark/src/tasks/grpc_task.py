"""
src/tasks/grpc_task.py
Menggantikan src/tests/grpc.test.js

Perbedaan utama vs k6:
  - Binary dikirim LANGSUNG sebagai bytes ke protobuf field
  - Tidak ada b64encode/decode overhead sama sekali
  - Channel di-reuse per user (HTTP/2 multiplexing)
"""

from __future__ import annotations

import itertools
import time
from typing import Iterator

from src.config.config import TEST_DATASET, METADATA, CHUNK_SIZE, TIMEOUT
from src.utils.grpc_client import make_stub, get_pb2

_cycle = itertools.cycle(TEST_DATASET)


def _request_iter(tc: dict) -> Iterator:
    """
    Generator stream request:
      Frame 0  : metadata (ImageInfo)
      Frame 1+ : potongan binary image — raw bytes, tanpa encoding
    """
    pb2 = get_pb2()

    # Frame 0 — metadata
    yield pb2.AnalyzeSkinRequest(
        info=pb2.ImageInfo(
            user_id       = METADATA["user_id"],
            image_type    = METADATA["image_type"],
            client_sha256 = tc["hash_b64"],
            metadata      = METADATA["meta_tags"],
        )
    )

    # Frame 1..N — binary chunks langsung
    data   = tc["data"]
    offset = 0
    while offset < len(data):
        end = min(offset + CHUNK_SIZE, len(data))
        yield pb2.AnalyzeSkinRequest(chunk=data[offset:end])  # ← raw bytes
        offset = end


def analyze_skin(stub, environment) -> None:
    """
    Kirim satu client-streaming RPC.
    stub        = make_stub(channel)  — dibuat di on_start(), di-reuse
    environment = self.environment
    """
    tc        = next(_cycle)
    t0        = time.perf_counter()
    exc       = None

    try:
        res = stub.AnalyzeSkin(_request_iter(tc), timeout=TIMEOUT)
        exc = _assert(res, tc)
        if exc:
            exc = AssertionError(exc)

    except Exception as e:
        exc = e

    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        environment.events.request.fire(
            request_type    = "gRPC",
            name            = "SkinAnalysisService/AnalyzeSkin",
            response_time   = elapsed_ms,
            response_length = 0,
            exception       = exc,
            context         = {},
        )


def _assert(res, tc: dict) -> str | None:
    failures = []

    if not isinstance(getattr(res, "analysis_id", None), str):
        failures.append("missing analysisId")

    results = list(getattr(res, "results", []))
    if not results:
        failures.append("results kosong")
    else:
        top  = results[0]
        conf = getattr(top, "confidence", -1)
        if not (0 <= conf <= 1):
            failures.append(f"confidence out of range: {conf}")
        for field in ("label", "description", "recommendation"):
            if not isinstance(getattr(top, field, None), str):
                failures.append(f"missing {field}")
        if getattr(top, "label", "") != tc["expected_label"]:
            failures.append(
                f"wrong label: got '{getattr(top, 'label', '')}' "
                f"expected '{tc['expected_label']}'"
            )

    return " | ".join(failures) if failures else None