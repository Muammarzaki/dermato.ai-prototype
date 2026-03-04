"""
src/tasks/rest_task.py
Menggantikan src/tests/rest.test.js
"""

from __future__ import annotations

import itertools
import json

from src.config.config import TEST_DATASET, METADATA, TIMEOUT

_cycle = itertools.cycle(TEST_DATASET)


def analyze_skin(client) -> None:
    """
    Kirim satu request multipart POST ke /analyze-skin.
    client = self.client (HttpSession Locust).
    """
    tc = next(_cycle)

    files = {"file": (tc["filename"], tc["data"], "image/jpeg")}
    data  = {
        "user_id":       METADATA["user_id"],
        "client_sha256": tc["hash_hex"],
        "metadata":      json.dumps(METADATA["meta_tags"]),
    }

    with client.post(
        "/analyze-skin",
        files=files,
        data=data,
        timeout=TIMEOUT,
        name="REST /analyze-skin",
        catch_response=True,
    ) as res:

        if res.status_code < 200 or res.status_code >= 300:
            res.failure(f"HTTP {res.status_code}: {res.text[:200]}")
            return

        try:
            body = res.json()
        except Exception as e:
            res.failure(f"JSON parse error: {e}")
            return

        err = _assert(body, tc)
        if err:
            res.failure(err)
        else:
            res.success()


def _assert(body: dict, tc: dict) -> str | None:
    failures = []

    if not isinstance(body.get("analysis_id"), str):
        failures.append("missing analysis_id")
    if not isinstance(body.get("server_sha256"), str):
        failures.append("missing server_sha256")

    results = body.get("results", [])
    if not isinstance(results, list) or not results:
        failures.append("results kosong")
    else:
        top  = results[0]
        conf = top.get("confidence", -1)
        if not (0 <= conf <= 1):
            failures.append(f"confidence out of range: {conf}")
        for field in ("label", "description", "recommendation"):
            if not isinstance(top.get(field), str):
                failures.append(f"missing {field}")
        if top.get("label") != tc["expected_label"]:
            failures.append(
                f"wrong label: got '{top.get('label')}' "
                f"expected '{tc['expected_label']}'"
            )

    return " | ".join(failures) if failures else None