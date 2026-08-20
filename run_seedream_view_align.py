#!/usr/bin/env python3
"""One-off: align v3-2 camera to the original first frame, keep styled look."""

from __future__ import annotations

import json
from pathlib import Path

import requests

from ark_common import (
    BASE_URL,
    ROOT,
    auth_headers,
    get_model_ids,
    image_data_url,
    load_api_key,
    mask_secret,
    pick_seedream_model,
    save_json,
)
from run_seedream_yaml_edit import save_image_bytes

STYLED = ROOT / "output/20260820-dream/first_frame/stack-part2_tech-night_v3-2.jpg"
ORIGIN = ROOT / "inputs/20260820-dream/stack-part2_first.jpg"
DEST = ROOT / "output/20260820-dream/first_frame/stack-part2_tech-night_v3-2-align.jpg"

PROMPT = """以这一张原图为基础做风格替换，最高优先级是机位完全不变。
必须沿用原图的摄像机高度、俯仰、焦距、透视和裁切：低机位、贴近桌面、桌面近景铺满画面，椅子只露出部分靠背。
禁止抬高镜头、禁止拉远、禁止拍成完整餐厅全景或拍出整圈座椅。
机械臂从左上进入的角度、长度、夹爪相对桌面的位置必须与原图一致。
两个碗的位置、大小、间距必须与原图一致。
将大理石桌、陶瓷碗、浅色软包椅替换为原木桌、原木碗、原木椅；
夜晚室内明亮落地灯光，无明显阴影轮廓；落地窗外为别墅园林夜景；背景家具极少、造型极简。
不要改变物体种类和左右关系。输出画幅与原图一致。"""


def main() -> int:
    if not STYLED.exists() or not ORIGIN.exists():
        print(f"missing input: styled={STYLED.exists()} origin={ORIGIN.exists()}")
        return 1
    api_key = load_api_key()
    label, model_id = pick_seedream_model(get_model_ids(api_key))
    print(f"API Key: {mask_secret(api_key)}")
    print(f"model: {label} / {model_id}")
    payload = {
        "model": model_id,
        "prompt": PROMPT,
        "image": image_data_url(ORIGIN),
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
    print(f"HTTP {resp.status_code}")
    if resp.status_code >= 400 or (isinstance(body, dict) and body.get("error")):
        print(json.dumps(body, ensure_ascii=False)[:2000])
        return 1
    DEST.parent.mkdir(parents=True, exist_ok=True)
    save_image_bytes(body, DEST)
    save_json(
        DEST.with_name(DEST.stem + "_last_run.json"),
        {"saved": str(DEST), "model": model_id, "prompt": PROMPT, "response": body},
    )
    print(f"saved: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
