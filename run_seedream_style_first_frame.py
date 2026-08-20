#!/usr/bin/env python3
"""
Edit the extracted first frame with Seedream 5.0 Pro (Lite fallback).
Style: 科技夜晚 — keep foreground, replace background.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from ark_common import (
    BASE_URL,
    OUTPUT_SEEDANCE2,
    auth_headers,
    get_model_ids,
    image_data_url,
    load_api_key,
    mask_secret,
    pick_seedream_model,
    save_json,
)

FIRST_FRAME = OUTPUT_SEEDANCE2 / "26-08-19_Origin_4s_first.jpg"
META_JSON = OUTPUT_SEEDANCE2 / "seedream_style_last_run.json"

STYLE_PROMPT = (
    "以参考图为基础，将场景转换为【科技别墅极简家庭风格】。"
    "硬性约束："
    "- 保持原有前景布局、机位、视角、空间结构及主要物体位置；"
    "- 保持前景夹爪造型不变，确保双视角内容、光照、元素等完全一致；"
    "- 桌面外的背景布局需要极简干净低频，换位为无人家庭背景，极简且低频；"
    "- 采用建筑摄影构图、电影级光照、真实材质和高细节超写实表现；"
    "- 输出的图片比例和结构与原图严格一致。"
    "风格设定："
    "- 时间与光线：夜晚，明亮落地灯光照明，无明显阴影轮廓，无明显光源方向；"
    "- 材质与装饰：极简桌面和极为低频的背景，背景和家具物品数量极少，造型保持极简；"
    "- 氛围与效果：明亮夜晚科技感极简家庭氛围，落地窗外有别墅院内园林夜景。"
)


def extract_urls_from_response(data: Dict) -> List[str]:
    urls: List[str] = []
    payload = data.get("data")
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                url = item.get("url")
                if isinstance(url, str) and url:
                    urls.append(url)
    return urls


def invoke_seedream(
    api_key: str, model_id: str, image_path: Path
) -> Tuple[bool, str, Optional[Path], Dict]:
    payload = {
        "model": model_id,
        "prompt": STYLE_PROMPT,
        "image": image_data_url(image_path),
        "response_format": "url",
        "size": "2K",
        "watermark": False,
    }
    resp = requests.post(
        f"{BASE_URL}/images/generations",
        headers=auth_headers(api_key),
        json=payload,
        timeout=1200,
    )
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}

    if resp.status_code >= 400:
        return False, f"HTTP {resp.status_code}: {body}", None, {"request": {"model": model_id}, "response": body}

    if isinstance(body, dict) and body.get("error"):
        return False, json.dumps(body.get("error"), ensure_ascii=False), None, {
            "request": {"model": model_id},
            "response": body,
        }

    urls = extract_urls_from_response(body) if isinstance(body, dict) else []
    OUTPUT_SEEDANCE2.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if urls:
        r = requests.get(urls[0], timeout=120)
        r.raise_for_status()
        out_path = OUTPUT_SEEDANCE2 / f"first_frame_tech_night_{model_id}_{stamp}.jpg"
        out_path.write_bytes(r.content)
        return True, "ok", out_path, {
            "request": {"model": model_id, "prompt": STYLE_PROMPT, "size": "2K"},
            "response": body,
            "saved": str(out_path),
        }

    # b64 fallback
    items = body.get("data", []) if isinstance(body, dict) else []
    if isinstance(items, list) and items:
        item0 = items[0]
        if isinstance(item0, dict) and isinstance(item0.get("b64_json"), str):
            import base64

            b64_value = item0["b64_json"]
            if b64_value.startswith("data:") and "," in b64_value:
                b64_value = b64_value.split(",", 1)[1]
            out_path = OUTPUT_SEEDANCE2 / f"first_frame_tech_night_{model_id}_{stamp}.jpg"
            out_path.write_bytes(base64.b64decode(b64_value))
            return True, "ok", out_path, {
                "request": {"model": model_id, "prompt": STYLE_PROMPT, "size": "2K"},
                "response": body,
                "saved": str(out_path),
            }

    return False, "No image url or b64_json returned", None, {
        "request": {"model": model_id},
        "response": body,
    }


def main() -> int:
    api_key = load_api_key()
    if not api_key:
        print("未找到 API_KEY：请配置同目录 .env")
        return 1
    if not FIRST_FRAME.exists():
        print(f"首帧不存在，请先运行 extract_first_frame.py: {FIRST_FRAME}")
        return 1

    model_ids = get_model_ids(api_key)
    label, model_id = pick_seedream_model(model_ids)
    print(f"API Key: {mask_secret(api_key)}")
    print(f"Seedream model: {label} / {model_id}")
    print(f"Input: {FIRST_FRAME}")

    ok, msg, saved, meta = invoke_seedream(api_key, model_id, FIRST_FRAME)
    save_json(META_JSON, meta)
    if ok and saved:
        print("[OK] Seedream style edit succeeded")
        print(f"saved: {saved}")
        return 0

    print(f"[FAIL] {msg}")
    print(f"meta saved: {META_JSON}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
