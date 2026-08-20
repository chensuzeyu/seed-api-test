#!/usr/bin/env python3
"""Edit a video with Seedance 2.0, reading prompt YAML from prompts/."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
import yaml

from ark_common import (
    BASE_URL,
    ROOT,
    auth_headers,
    get_model_ids,
    load_api_key,
    mask_secret,
    pick_seedance2_model,
    save_json,
    upload_to_tos_and_presign,
)

DEFAULT_PROMPT_FILE = ROOT / "prompts" / "prompts_20260820_video.yaml"
POLL_INTERVAL_SEC = 15
POLL_TIMEOUT_SEC = 15 * 60


def load_prompt_config(path: Path) -> Dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid yaml: {path}")
    return data


def infer_slug(first_frame: Path, source_video: Path) -> str:
    stem = first_frame.stem
    video_stem = source_video.stem
    prefix = f"{video_stem}_"
    if stem.startswith(prefix):
        rest = stem[len(prefix) :]
        slug = rest.split("_")[0] if rest else "style"
        return slug or "style"
    return "style"


def output_path(out_dir: Path, source_video: Path, slug: str, version: str, index: int, n: int) -> Path:
    if n <= 1:
        return out_dir / f"{source_video.stem}_{slug}_{version}.mp4"
    return out_dir / f"{source_video.stem}_{slug}_{version}-{index}.mp4"


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


def build_payload(
    model_id: str,
    prompt: str,
    image_url: str,
    video_url: str,
    resolution: str,
    duration: int,
    ratio: str,
    generate_audio: bool,
    watermark: bool,
) -> Dict[str, Any]:
    return {
        "model": model_id,
        "content": [
            {"type": "text", "text": prompt},
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
        "resolution": resolution,
        "duration": duration,
        "ratio": ratio,
        "generate_audio": generate_audio,
        "watermark": watermark,
    }


def download_video(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def main() -> int:
    prompt_file = DEFAULT_PROMPT_FILE
    cfg = load_prompt_config(prompt_file)
    active = cfg.get("active") or {}
    source_video = ROOT / str(active.get("source_video") or "")
    first_frame = ROOT / str(active.get("first_frame") or "")
    out_dir = ROOT / str(active.get("out_dir") or "output")
    version = str(active.get("version") or "v1")
    n = int(active.get("n") or 1)
    resolution = str(active.get("resolution") or "720p")
    duration = int(active.get("duration") or 4)
    ratio = str(active.get("ratio") or "adaptive")
    generate_audio = bool(active.get("generate_audio") or False)
    watermark = bool(active.get("watermark") or False)
    prompt = str(cfg.get("template") or "").strip()
    slug = infer_slug(first_frame, source_video)

    if not source_video.exists():
        print(f"源视频不存在: {source_video}")
        return 1
    if not first_frame.exists():
        print(f"首图不存在: {first_frame}")
        return 1
    if not prompt:
        print("YAML template 为空")
        return 1
    if n != 1:
        print("本次脚本先支持 n=1；并发请改 YAML 后再扩展")
        return 1

    api_key = load_api_key()
    if not api_key:
        print("未找到 API_KEY：请配置同目录 .env")
        return 1

    model_ids = get_model_ids(api_key)
    label, model_id = pick_seedance2_model(model_ids)
    dest = output_path(out_dir, source_video, slug, version, 1, n)

    print(f"API Key: {mask_secret(api_key)}")
    print(f"Seedance model: {label} / {model_id}")
    print(f"Prompt file: {prompt_file}")
    print(f"first_frame: {first_frame}")
    print(f"source_video: {source_video}")
    print(f"params: resolution={resolution} duration={duration} ratio={ratio}")
    print(f"output: {dest}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_key = f"seed-api-test/seedance2/{stamp}/{source_video.name}"
    image_key = f"seed-api-test/seedance2/{stamp}/{first_frame.name}"

    print(f"uploading video to TOS: {video_key}")
    video_url = upload_to_tos_and_presign(source_video, video_key)
    print(f"video presign ok: {video_url[:100]}...")

    print(f"uploading image to TOS: {image_key}")
    image_url = upload_to_tos_and_presign(first_frame, image_key)
    print(f"image presign ok: {image_url[:100]}...")

    payload = build_payload(
        model_id,
        prompt,
        image_url,
        video_url,
        resolution,
        duration,
        ratio,
        generate_audio,
        watermark,
    )
    meta: Dict[str, Any] = {
        "prompt_file": str(prompt_file),
        "model": model_id,
        "first_frame": str(first_frame),
        "source_video": str(source_video),
        "slug": slug,
        "version": version,
        "prompt": prompt,
        "tos": {"video_key": video_key, "image_key": image_key},
        "hard_params": {
            "resolution": resolution,
            "duration": duration,
            "ratio": ratio,
            "generate_audio": generate_audio,
            "watermark": watermark,
        },
        "request_content_roles": ["text", "reference_image", "reference_video"],
    }

    print("\n== create Seedance edit-video task ==")
    ok, result, create_body = create_task(api_key, payload)
    meta["create"] = create_body if isinstance(create_body, dict) else {"raw": create_body}
    if not ok:
        save_json(out_dir / f"{source_video.stem}_{slug}_{version}_last_run.json", meta)
        print(f"[FAIL create] {result}")
        return 1

    task_id = result
    meta["task_id"] = task_id
    print(f"task_id={task_id}")
    try:
        task = poll_task(api_key, task_id)
    except Exception as exc:
        meta["error"] = str(exc)
        save_json(out_dir / f"{source_video.stem}_{slug}_{version}_last_run.json", meta)
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
        save_json(out_dir / f"{source_video.stem}_{slug}_{version}_last_run.json", meta)
        print(f"[FAIL status={status}] {json.dumps(task.get('error') or task, ensure_ascii=False)}")
        return 1

    content = task.get("content") or {}
    out_url = content.get("video_url") if isinstance(content, dict) else None
    if not isinstance(out_url, str) or not out_url:
        meta["error"] = f"missing content.video_url: {task}"
        save_json(out_dir / f"{source_video.stem}_{slug}_{version}_last_run.json", meta)
        print(f"[FAIL] {meta['error']}")
        return 1

    print(f"downloading -> {dest}")
    download_video(out_url, dest)
    meta["saved"] = str(dest)

    spec_errors: List[str] = []
    if task.get("duration") != duration:
        spec_errors.append(f"duration expected {duration}, got {task.get('duration')!r}")
    if task.get("resolution") != resolution:
        spec_errors.append(f"resolution expected {resolution!r}, got {task.get('resolution')!r}")
    meta["spec_errors"] = spec_errors
    save_json(out_dir / f"{source_video.stem}_{slug}_{version}_last_run.json", meta)

    if spec_errors:
        print("[WARN] downloaded but API response spec mismatch:")
        for err in spec_errors:
            print(f"  - {err}")
        print(f"saved anyway: {dest}")
        return 2

    print("[OK] Seedance 2.0 style transfer succeeded")
    print(f"saved: {dest}")
    print(
        f"duration={task.get('duration')} "
        f"resolution={task.get('resolution')} "
        f"ratio={task.get('ratio')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
