#!/usr/bin/env python3
"""
Seedream image-edit connectivity test for Ark API keys.

What it does:
1. Reads API_KEY from the local .env file or environment
2. Uses the local reference images in ./inputs/
3. Optionally runs a three-reference fusion edit test
4. Saves any successful outputs under ./output/

Notes:
- This script only touches files inside this directory.
- It avoids printing the full API key.
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
INPUT_IMAGES = [
    ROOT / "inputs" / "ref-01.jpg",
    ROOT / "inputs" / "ref-02.jpg",
    ROOT / "inputs" / "ref-03.png",
]
OUTPUT_DIR = ROOT / "output"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ENABLE_THREE_REFERENCE_FUSION = True
MULTI_IMAGE_FIELD = "image"

# Image-edit smoke test. Older Seedream 4.0 / 4.5 / 5.0 Lite IDs are deprecated.
MODELS = [
    ("Seedream 5.0 Pro", "doubao-seedream-5-0-pro-260628"),
]

PROMPT = (
    "请以 ref-01 作为主图进行编辑，尽量保留 ref-01 的原始风格、主体外观、材质、构图、色彩和整体视觉效果，"
    "不要把画面改成与原图差异很大的全新风格。"
    "在保持 ref-01 主体和主画面基本不变的前提下，加入一个机械夹爪，明确去抓取 ref-01 里的主体物体。"
    "同时读取 ref-02 和 ref-03，并将这两张参考图中的关键物体自然融合进同一画面，"
    "要求三张参考图中的物体都能被识别出来，但整体仍以 ref-01 的原图风格和原图气质为主。"
    "最终效果应当像基于 ref-01 做的一次真实、小幅、可控的多参考图融合编辑，而不是完全重新生成一张新图。"
)


def load_dotenv_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_api_key() -> str:
    env_values = load_dotenv_file(ENV_FILE)
    return os.environ.get("API_KEY") or env_values.get("API_KEY", "")


def mask_secret(secret: str) -> str:
    if len(secret) <= 10:
        return "***"
    return f"{secret[:6]}...{secret[-4:]}"


def auth_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def get_model_status_map(api_key: str) -> Dict[str, Optional[str]]:
    resp = requests.get(
        f"{BASE_URL}/models",
        headers=auth_headers(api_key),
        timeout=30,
    )
    resp.raise_for_status()
    out: Dict[str, Optional[str]] = {}
    for item in resp.json().get("data", []):
        model_id = item.get("id")
        if isinstance(model_id, str):
            out[model_id] = item.get("status")
    return out


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "jpg":
        suffix = "jpeg"
    mime = f"image/{suffix or 'jpeg'}"
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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


def build_reference_payload() -> List[str] | str:
    encoded_images = [image_data_url(path) for path in INPUT_IMAGES]
    if ENABLE_THREE_REFERENCE_FUSION:
        return encoded_images
    return encoded_images[0]


def download_url(url: str, dest: Path) -> None:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def guess_ext(content_type: Optional[str], url: str) -> str:
    if content_type:
        ct = content_type.split(";", 1)[0].strip().lower()
        if ct == "image/jpeg":
            return ".jpg"
        if ct == "image/png":
            return ".png"
        if ct == "image/webp":
            return ".webp"
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def save_b64_image(b64_value: str, dest: Path) -> None:
    if b64_value.startswith("data:") and "," in b64_value:
        b64_value = b64_value.split(",", 1)[1]
    dest.write_bytes(base64.b64decode(b64_value))


def build_output_path(model_id: str, idx: int, ext: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return OUTPUT_DIR / f"{model_id}_edit_{idx}_{stamp}{ext}"


def invoke_seedream_once(api_key: str, payload: Dict) -> Tuple[bool, str, Optional[List[Path]]]:
    resp = requests.post(
        f"{BASE_URL}/images/generations",
        headers=auth_headers(api_key),
        json=payload,
        timeout=1200,
    )

    if resp.status_code >= 400:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return False, f"HTTP {resp.status_code}: {body}", None

    data = resp.json()
    if data.get("error"):
        return False, json.dumps(data.get("error"), ensure_ascii=False), None

    urls = extract_urls_from_response(data)
    saved: List[Path] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if urls:
        for idx, url in enumerate(urls, start=1):
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            ext = guess_ext(r.headers.get("Content-Type"), url)
            out_path = build_output_path(payload["model"], idx, ext)
            out_path.write_bytes(r.content)
            saved.append(out_path)
        return True, json.dumps(data, ensure_ascii=False, indent=2), saved

    # Fallback for any future b64_json response format.
    payload_items = data.get("data", [])
    if isinstance(payload_items, list) and payload_items:
        for idx, item in enumerate(payload_items, start=1):
            if isinstance(item, dict):
                b64_value = item.get("b64_json")
                if isinstance(b64_value, str) and b64_value:
                    out_path = build_output_path(payload["model"], idx, ".jpg")
                    save_b64_image(b64_value, out_path)
                    saved.append(out_path)
        if saved:
            return True, json.dumps(data, ensure_ascii=False, indent=2), saved

    return False, "No image url or b64_json returned", None


def invoke_seedream(api_key: str, model_id: str) -> Tuple[bool, str, Optional[List[Path]]]:
    reference_payload = build_reference_payload()
    base_payload = {
        "model": model_id,
        "prompt": PROMPT,
        "response_format": "url",
        "size": "2304x1728",
        "watermark": False,
    }

    payload_variants: List[Tuple[str, Dict]] = []
    if ENABLE_THREE_REFERENCE_FUSION:
        payload_variants.append(
            (
                "image_multi",
                {**base_payload, MULTI_IMAGE_FIELD: reference_payload},
            )
        )
    else:
        payload_variants.append(
            ("single_image", {**base_payload, "image": reference_payload})
        )

    errors: List[str] = []
    for variant_name, payload in payload_variants:
        ok, msg, saved = invoke_seedream_once(api_key, payload)
        if ok:
            tagged_msg = json.dumps(
                {
                    "request_variant": variant_name,
                    "request_image_count": (
                        len(reference_payload) if isinstance(reference_payload, list) else 1
                    ),
                    "response": json.loads(msg),
                },
                ensure_ascii=False,
                indent=2,
            )
            return True, tagged_msg, saved
        errors.append(f"{variant_name}: {msg}")

    return False, " | ".join(errors), None


def main() -> int:
    api_key = load_api_key().strip()
    if not api_key:
        print("未找到 API_KEY：请确认 /mnt/nas/chensuzeyu/tmp/seed-api-test/.env 已配置")
        return 1

    missing_images = [path for path in INPUT_IMAGES if not path.exists()]
    if missing_images:
        print("未找到参考图：")
        for path in missing_images:
            print(f"  - {path}")
        return 1

    model_status = get_model_status_map(api_key)
    print(f"API Key: {mask_secret(api_key)}")
    print(f"Base URL: {BASE_URL}")
    print(f"Three-reference fusion enabled: {ENABLE_THREE_REFERENCE_FUSION}")
    print(f"Multi-image request field: {MULTI_IMAGE_FIELD}")
    print("Reference images:")
    for path in INPUT_IMAGES:
        print(f"  - {path}")
    print(f"Prompt: {PROMPT}")
    print()

    for label, model_id in MODELS:
        status = model_status.get(model_id)
        print(f"== {label} / {model_id} ==")
        print(f"model list status: {status!r}")
        ok, msg, saved = invoke_seedream(api_key, model_id)
        if ok:
            print("[OK] image edit succeeded")
            if saved:
                for path in saved:
                    print(f"saved: {path}")
            print(msg)
        else:
            print(f"[FAIL] {msg}")
        print()

    print("说明: 图编冒烟只调用 Seedream 5.0 Pro。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
