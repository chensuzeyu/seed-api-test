#!/usr/bin/env python3
"""
Seedance 2.0 style transfer smoke test.

Mode: 编辑视频 (reference_image + reference_video + text)
Hard constraints: resolution=720p, duration=4

Media: local files are uploaded to TOS; Seedance receives HTTPS presigned URLs.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

from ark_common import (
    BASE_URL,
    INPUT_VIDEO,
    OUTPUT_SEEDANCE2,
    auth_headers,
    get_model_ids,
    load_api_key,
    mask_secret,
    pick_seedance2_model,
    save_json,
    upload_to_tos_and_presign,
)

# Hard requirements from plan — do not change for this smoke test.
RESOLUTION = "720p"
DURATION = 4
RATIO = "adaptive"
GENERATE_AUDIO = False
WATERMARK = False

META_JSON = OUTPUT_SEEDANCE2 / "seedance2_last_run.json"
POLL_INTERVAL_SEC = 15
POLL_TIMEOUT_SEC = 15 * 60

STYLE_TRANSFER_PROMPT = (
    "参考图片1的科技夜晚室内风格与光照，对视频1做背景风格迁移。"
    "硬性约束：保持视频1的运镜、机位、镜头运动、前景夹爪造型与主体动作完全一致；"
    "仅替换背景为极简科技别墅家庭夜晚场景，明亮落地灯光，窗外别墅园林夜景；"
    "画面风格与图片1一致，电影级光照，真实材质，超写实。"
)


def latest_styled_frame() -> Path:
    candidates = sorted(
        OUTPUT_SEEDANCE2.glob("first_frame_tech_night_*.jpg"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"未找到编辑后的首图，请先运行 run_seedream_style_first_frame.py: {OUTPUT_SEEDANCE2}"
        )
    return candidates[0]


def create_task(api_key: str, payload: Dict[str, Any]) -> Tuple[bool, str, Dict]:
    resp = requests.post(
        f"{BASE_URL}/contents/generations/tasks",
        headers=auth_headers(api_key),
        json=payload,
        timeout=120,
    )
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    if resp.status_code >= 400:
        return False, f"HTTP {resp.status_code}: {json.dumps(body, ensure_ascii=False)}", {
            "http_status": resp.status_code,
            "body": body,
        }
    task_id = body.get("id") if isinstance(body, dict) else None
    if not isinstance(task_id, str) or not task_id:
        return False, f"创建任务未返回 id: {body}", {"body": body}
    return True, task_id, body


def get_task(api_key: str, task_id: str) -> Dict:
    resp = requests.get(
        f"{BASE_URL}/contents/generations/tasks/{task_id}",
        headers=auth_headers(api_key),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def poll_task(api_key: str, task_id: str) -> Dict:
    started = time.time()
    while True:
        data = get_task(api_key, task_id)
        status = data.get("status")
        print(f"  status={status!r} elapsed={int(time.time() - started)}s")
        if status in {"succeeded", "failed", "expired", "cancelled"}:
            return data
        if time.time() - started > POLL_TIMEOUT_SEC:
            raise TimeoutError(f"轮询超时 {POLL_TIMEOUT_SEC}s, last={data}")
        time.sleep(POLL_INTERVAL_SEC)


def build_payload(model_id: str, image_url: str, video_url: str) -> Dict[str, Any]:
    return {
        "model": model_id,
        "content": [
            {"type": "text", "text": STYLE_TRANSFER_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": image_url},
                "role": "reference_image",
            },
            {
                "type": "video_url",
                "video_url": {"url": video_url},
                "role": "reference_video",
            },
        ],
        "resolution": RESOLUTION,
        "duration": DURATION,
        "ratio": RATIO,
        "generate_audio": GENERATE_AUDIO,
        "watermark": WATERMARK,
    }


def download_video(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def assert_output_spec(task: Dict) -> List[str]:
    errors: List[str] = []
    if task.get("duration") != DURATION:
        errors.append(f"duration expected {DURATION}, got {task.get('duration')!r}")
    if task.get("resolution") != RESOLUTION:
        errors.append(f"resolution expected {RESOLUTION!r}, got {task.get('resolution')!r}")
    return errors


def main() -> int:
    api_key = load_api_key()
    if not api_key:
        print("未找到 API_KEY：请配置同目录 .env")
        return 1
    if not INPUT_VIDEO.exists():
        print(f"输入视频不存在: {INPUT_VIDEO}")
        return 1

    try:
        styled = latest_styled_frame()
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    model_ids = get_model_ids(api_key)
    label, model_id = pick_seedance2_model(model_ids)
    print(f"API Key: {mask_secret(api_key)}")
    print(f"Seedance model: {label} / {model_id}")
    print(f"styled first frame: {styled}")
    print(f"reference video: {INPUT_VIDEO}")
    print(f"hard params: resolution={RESOLUTION} duration={DURATION} ratio={RATIO}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_key = f"seed-api-test/seedance2/{stamp}/26-08-19_Origin_4s.mp4"
    image_key = f"seed-api-test/seedance2/{stamp}/{styled.name}"

    print(f"uploading video to TOS: {video_key}")
    video_url = upload_to_tos_and_presign(INPUT_VIDEO, video_key)
    print(f"video presign ok: {video_url[:100]}...")

    print(f"uploading image to TOS: {image_key}")
    image_url = upload_to_tos_and_presign(styled, image_key)
    print(f"image presign ok: {image_url[:100]}...")

    payload = build_payload(model_id, image_url, video_url)
    meta: Dict[str, Any] = {
        "model": model_id,
        "styled_frame": str(styled),
        "video": str(INPUT_VIDEO),
        "tos": {"video_key": video_key, "image_key": image_key},
        "hard_params": {
            "resolution": RESOLUTION,
            "duration": DURATION,
            "ratio": RATIO,
            "generate_audio": GENERATE_AUDIO,
            "watermark": WATERMARK,
        },
        "request_content_roles": ["text", "reference_image", "reference_video"],
    }

    print("\n== create Seedance edit-video task ==")
    ok, result, create_body = create_task(api_key, payload)
    meta["create"] = create_body if isinstance(create_body, dict) else {"raw": create_body}
    if not ok:
        save_json(META_JSON, meta)
        print(f"[FAIL create] {result}")
        return 1

    task_id = result
    meta["task_id"] = task_id
    print(f"task_id={task_id}")
    try:
        task = poll_task(api_key, task_id)
    except Exception as exc:
        meta["error"] = str(exc)
        save_json(META_JSON, meta)
        print(f"[FAIL poll] {exc}")
        return 1

    meta["task"] = {
        k: task.get(k)
        for k in (
            "id",
            "model",
            "status",
            "duration",
            "resolution",
            "ratio",
            "generate_audio",
            "error",
            "usage",
        )
        if k in task
    }
    status = task.get("status")
    if status != "succeeded":
        meta["error"] = task.get("error") or task
        save_json(META_JSON, meta)
        print(f"[FAIL status={status}] {json.dumps(task.get('error') or task, ensure_ascii=False)}")
        return 1

    content = task.get("content") or {}
    out_url = content.get("video_url") if isinstance(content, dict) else None
    if not isinstance(out_url, str) or not out_url:
        meta["error"] = f"missing content.video_url: {task}"
        save_json(META_JSON, meta)
        print(f"[FAIL] {meta['error']}")
        return 1

    out_path = OUTPUT_SEEDANCE2 / f"styled_26-08-19_Origin_4s_{RESOLUTION}_{DURATION}s_{stamp}.mp4"
    print(f"downloading -> {out_path}")
    download_video(out_url, out_path)
    meta["saved"] = str(out_path)
    meta["output_video_url_prefix"] = out_url[:80] + "..."

    spec_errors = assert_output_spec(task)
    meta["spec_errors"] = spec_errors
    save_json(META_JSON, meta)

    if spec_errors:
        print("[WARN] downloaded but API response spec mismatch:")
        for err in spec_errors:
            print(f"  - {err}")
        print(f"saved anyway: {out_path}")
        return 2

    print("[OK] Seedance 2.0 style transfer succeeded")
    print(f"saved: {out_path}")
    print(
        f"duration={task.get('duration')} "
        f"resolution={task.get('resolution')} "
        f"ratio={task.get('ratio')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
