#!/usr/bin/env python3
"""Concurrent latency benchmark for Seedream 5.0 Lite image edit."""

from __future__ import annotations

import base64
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from check_seedream_edit import (
    BASE_URL,
    INPUT_IMAGES,
    MODELS,
    PROMPT,
    auth_headers,
    build_reference_payload,
    download_url,
    extract_urls_from_response,
    guess_ext,
    load_api_key,
    mask_secret,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "benchmark_concurrent"
MODEL_ID = "doubao-seedream-5-0-lite-260128"
CONCURRENCY = 5


def build_payload() -> dict[str, Any]:
    reference_payload = build_reference_payload()
    return {
        "model": MODEL_ID,
        "prompt": PROMPT,
        "image": reference_payload,
        "response_format": "url",
        "size": "2304x1728",
        "watermark": False,
    }


def run_one_request(api_key: str, request_id: int) -> dict[str, Any]:
    payload = build_payload()
    headers = auth_headers(api_key)

    t0 = time.perf_counter()
    api_started_at = datetime.now().isoformat(timespec="milliseconds")

    try:
        resp = requests.post(
            f"{BASE_URL}/images/generations",
            headers=headers,
            json=payload,
            timeout=1200,
        )
        api_elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            return {
                "request_id": request_id,
                "ok": False,
                "api_started_at": api_started_at,
                "api_elapsed_ms": api_elapsed_ms,
                "total_elapsed_ms": api_elapsed_ms,
                "error": f"HTTP {resp.status_code}: {detail}",
            }

        data = resp.json()
        if data.get("error"):
            return {
                "request_id": request_id,
                "ok": False,
                "api_started_at": api_started_at,
                "api_elapsed_ms": api_elapsed_ms,
                "total_elapsed_ms": api_elapsed_ms,
                "error": json.dumps(data["error"], ensure_ascii=False),
            }

        urls = extract_urls_from_response(data)
        if not urls:
            return {
                "request_id": request_id,
                "ok": False,
                "api_started_at": api_started_at,
                "api_elapsed_ms": api_elapsed_ms,
                "total_elapsed_ms": api_elapsed_ms,
                "error": "No image url returned",
            }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []
        download_elapsed_ms = 0
        for idx, url in enumerate(urls, start=1):
            t_download = time.perf_counter()
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            download_elapsed_ms += int((time.perf_counter() - t_download) * 1000)
            ext = guess_ext(r.headers.get("Content-Type"), url)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = OUTPUT_DIR / f"{MODEL_ID}_req{request_id:02d}_{idx}_{stamp}{ext}"
            out_path.write_bytes(r.content)
            saved_paths.append(str(out_path))

        total_elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "request_id": request_id,
            "ok": True,
            "api_started_at": api_started_at,
            "api_elapsed_ms": api_elapsed_ms,
            "download_elapsed_ms": download_elapsed_ms,
            "total_elapsed_ms": total_elapsed_ms,
            "image_url": urls[0],
            "saved_paths": saved_paths,
            "image_bytes": out_path.stat().st_size if saved_paths else 0,
        }
    except Exception as exc:
        total_elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "request_id": request_id,
            "ok": False,
            "api_started_at": api_started_at,
            "api_elapsed_ms": total_elapsed_ms,
            "total_elapsed_ms": total_elapsed_ms,
            "error": str(exc),
        }


def summarize(results: list[dict[str, Any]], wall_ms: int) -> None:
    ok_results = [item for item in results if item.get("ok")]
    fail_results = [item for item in results if not item.get("ok")]

    print(f"Model: {MODEL_ID}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Reference images: {[str(p) for p in INPUT_IMAGES]}")
    print(f"Wall clock (all requests): {wall_ms} ms")
    print()

    for item in sorted(results, key=lambda x: x["request_id"]):
        status = "OK" if item.get("ok") else "FAIL"
        print(f"[{status}] request #{item['request_id']:02d}")
        print(f"  started_at: {item.get('api_started_at')}")
        if item.get("ok"):
            print(f"  api_elapsed_ms: {item['api_elapsed_ms']}")
            print(f"  download_elapsed_ms: {item.get('download_elapsed_ms', 0)}")
            print(f"  total_elapsed_ms: {item['total_elapsed_ms']}")
            print(f"  image_bytes: {item.get('image_bytes', 0)}")
            for path in item.get("saved_paths", []):
                print(f"  saved: {path}")
        else:
            print(f"  total_elapsed_ms: {item.get('total_elapsed_ms')}")
            print(f"  error: {item.get('error')}")
        print()

    if ok_results:
        api_values = [item["api_elapsed_ms"] for item in ok_results]
        total_values = [item["total_elapsed_ms"] for item in ok_results]
        print("Latency summary (successful requests):")
        print(f"  api_elapsed_ms: min={min(api_values)}, max={max(api_values)}, avg={statistics.mean(api_values):.1f}")
        print(
            f"  total_elapsed_ms: min={min(total_values)}, max={max(total_values)}, avg={statistics.mean(total_values):.1f}"
        )
        if len(api_values) > 1:
            print(f"  api_elapsed_ms stdev={statistics.pstdev(api_values):.1f}")
            print(f"  total_elapsed_ms stdev={statistics.pstdev(total_values):.1f}")

    if fail_results:
        print(f"Failed requests: {len(fail_results)}/{len(results)}")


def main() -> int:
    api_key = load_api_key().strip()
    if not api_key:
        print("未找到 API_KEY")
        return 1

    missing = [path for path in INPUT_IMAGES if not path.exists()]
    if missing:
        print("缺少参考图:")
        for path in missing:
            print(f"  - {path}")
        return 1

    print(f"API Key: {mask_secret(api_key)}")
    print(f"Endpoint: {BASE_URL}/images/generations")
    print(f"Three-reference fusion: image=[ref-01, ref-02, ref-03]")
    print(f"Prompt length: {len(PROMPT)} chars")
    print()

    wall_start = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(run_one_request, api_key, request_id): request_id
            for request_id in range(1, CONCURRENCY + 1)
        }
        for future in as_completed(futures):
            results.append(future.result())

    wall_ms = int((time.perf_counter() - wall_start) * 1000)
    summarize(results, wall_ms)

    report_path = OUTPUT_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "concurrency": CONCURRENCY,
                "wall_ms": wall_ms,
                "results": sorted(results, key=lambda x: x["request_id"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Report saved: {report_path}")
    return 0 if all(item.get("ok") for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
