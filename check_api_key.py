#!/usr/bin/env python3
"""
Test a Volcengine Ark API key for:
1. model list connectivity
2. a simple text Q&A call with doubao-seed-2-0-mini-260428

The script keeps secrets out of logs and only reads files in this directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests


ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL = "doubao-seed-2-0-mini-260428"
TIMEOUT = 30


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


def get_model_list(api_key: str) -> Tuple[List[str], Dict]:
    resp = requests.get(
        f"{BASE_URL}/models",
        headers=auth_headers(api_key),
        timeout=TIMEOUT,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"GET /models failed: HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    ids: List[str] = []
    for item in data.get("data", []):
        model_id = item.get("id")
        if isinstance(model_id, str):
            ids.append(model_id)
    return ids, data


def chat_once(api_key: str) -> Dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "请只回复一个字：好。"},
        ],
        "temperature": 0,
    }
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=auth_headers(api_key),
        json=payload,
        timeout=TIMEOUT,
    )
    if resp.status_code >= 400:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise RuntimeError(f"POST /chat/completions failed: HTTP {resp.status_code}: {body}")
    return resp.json()


def pick_preview(items: Iterable[str], limit: int = 20) -> List[str]:
    preview: List[str] = []
    for item in items:
        preview.append(item)
        if len(preview) >= limit:
            break
    return preview


def main() -> int:
    api_key = load_api_key().strip()
    if not api_key:
        print("未找到 API_KEY：请确认 /mnt/nas/chensuzeyu/tmp/seed-api-test/.env 已配置")
        return 1

    print(f"API Key: {mask_secret(api_key)}")
    print(f"Base URL: {BASE_URL}")
    print(f"Target model: {MODEL}")
    print()

    try:
        model_ids, model_payload = get_model_list(api_key)
        print(f"[OK] model list 可访问，共返回 {len(model_ids)} 个模型")
        if model_ids:
            print("前几个模型：")
            for mid in pick_preview(model_ids):
                print(f"  - {mid}")
        else:
            print("model list 返回成功，但 data 为空")

        if MODEL in set(model_ids):
            print(f"[OK] 目标模型已在 model list 中：{MODEL}")
        else:
            print(f"[WARN] 目标模型未出现在 model list 里：{MODEL}")

        print()
        chat_payload = chat_once(api_key)
        answer = (
            chat_payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        print("[OK] 简单问答成功")
        print(f"模型回复: {answer!r}")
        print()
        print("Raw response:")
        print(json.dumps(chat_payload, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        msg = str(exc)
        print(f"[FAIL] {msg}")
        if "ModelNotOpen" in msg or "activate the model service" in msg:
            print("诊断: 该 API Key 能访问模型列表，但当前账号尚未开通/激活目标模型的推理权限。")
            print(f"建议: 去 Ark Console 里激活 {MODEL}，再重新运行本脚本。")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
