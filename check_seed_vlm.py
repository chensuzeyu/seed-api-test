#!/usr/bin/env python3
"""
Lightweight inference smoke test for Doubao Seed VLM models:
1. pure text reply
2. image understanding (local ref image)

Targets:
- doubao-seed-2-1-pro-260628
- doubao-seed-2-1-turbo-260628
- doubao-seed-evolving

Reads API_KEY from .env / env. Does not print the full key.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
INPUT_IMAGE = ROOT / "inputs" / "ref-01.jpg"
OUTPUT_DIR = ROOT / "output" / "seed_vlm"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
TIMEOUT = 120

# Official Chat API: reasoning_effort default is medium (not high).
# Values for Seed series: minimal / low / medium / high.
REASONING_EFFORT = "low"
THINKING = {"type": "enabled"}

MODELS = [
    ("Seed-2.1 Pro", "doubao-seed-2-1-pro-260628"),
    ("Seed-2.1 Turbo", "doubao-seed-2-1-turbo-260628"),
    ("Seed-Evolving", "doubao-seed-evolving"),
]

TEXT_PROMPT = "请只用一句话介绍你自己，不超过30个字。"
VISION_PROMPT = "请用一句话描述这张图片里最主要的内容，不超过40个字。"


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


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "jpg":
        suffix = "jpeg"
    mime = f"image/{suffix or 'jpeg'}"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def extract_message_fields(payload: Dict[str, Any]) -> Tuple[str, str]:
    choices = payload.get("choices") or []
    if not choices:
        return "", ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                part = item.get("text")
                if isinstance(part, str):
                    parts.append(part)
        text = "".join(parts)
    reasoning_text = reasoning if isinstance(reasoning, str) else ""
    return text, reasoning_text


def chat_completions(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int = 256,
) -> Tuple[bool, str, Optional[Dict[str, Any]], Optional[int]]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "thinking": THINKING,
        "reasoning_effort": REASONING_EFFORT,
    }
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=auth_headers(api_key),
        json=payload,
        timeout=TIMEOUT,
    )
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}

    if resp.status_code >= 400:
        if isinstance(body, dict):
            err = body.get("error") or body
            msg = json.dumps(err, ensure_ascii=False)
        else:
            msg = str(body)
        return False, f"HTTP {resp.status_code}: {msg}", None, resp.status_code

    text, _reasoning = extract_message_fields(body if isinstance(body, dict) else {})
    return True, text, body if isinstance(body, dict) else None, resp.status_code


def diagnose(msg: str) -> Optional[str]:
    lower = msg.lower()
    if "modelnotopen" in lower or "activate the model service" in lower:
        return "模型出现在列表中，但当前 key 尚未开通/激活该模型推理权限。"
    if "invalidendpointormodel.modelnotfound" in lower or "does not exist" in lower:
        return "模型 ID 无效或不存在，请核对模型名。"
    if "authentication" in lower or "unauthorized" in lower or "invalid api" in lower:
        return "API Key 无效或无权限。"
    if "quota" in lower or "rate limit" in lower or "429" in msg:
        return "可能触发限流或额度不足。"
    return None


def build_case_result(
    case: str,
    ok: bool,
    text_or_err: str,
    raw: Optional[Dict[str, Any]],
    status: Optional[int],
) -> Dict[str, Any]:
    reasoning = ""
    if ok and raw:
        _, reasoning = extract_message_fields(raw)
    return {
        "case": case,
        "ok": ok,
        "http_status": status,
        "reply": text_or_err if ok else None,
        "reasoning_content": reasoning or None,
        "error": None if ok else text_or_err,
        "usage": (raw or {}).get("usage") if ok else None,
        "diagnosis": None if ok else diagnose(text_or_err),
    }


def run_text_case(api_key: str, model: str) -> Dict[str, Any]:
    ok, text_or_err, raw, status = chat_completions(
        api_key,
        model,
        [
            {"role": "system", "content": "你是一个简洁助手。"},
            {"role": "user", "content": TEXT_PROMPT},
        ],
    )
    return build_case_result("text", ok, text_or_err, raw, status)


def run_vision_case(api_key: str, model: str, image_url: str) -> Dict[str, Any]:
    ok, text_or_err, raw, status = chat_completions(
        api_key,
        model,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    )
    return build_case_result("vision", ok, text_or_err, raw, status)


def shorten(text: str, limit: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def print_case(result: Dict[str, Any]) -> None:
    label = "文本" if result["case"] == "text" else "多模态(图文)"
    if result["ok"]:
        print(f"  [{label}] OK  HTTP {result.get('http_status')}")
        print(f"    回复: {shorten(result.get('reply') or '')}")
        reasoning = result.get("reasoning_content") or ""
        if reasoning:
            print(f"    思维链: {shorten(reasoning, 160)} (len={len(reasoning)})")
        usage = result.get("usage") or {}
        if usage:
            details = usage.get("completion_tokens_details") or {}
            print(
                "    usage: "
                f"prompt={usage.get('prompt_tokens')} "
                f"completion={usage.get('completion_tokens')} "
                f"reasoning={details.get('reasoning_tokens')} "
                f"total={usage.get('total_tokens')}"
            )
    else:
        print(f"  [{label}] FAIL HTTP {result.get('http_status')}")
        print(f"    error: {result.get('error')}")
        if result.get("diagnosis"):
            print(f"    诊断: {result['diagnosis']}")


def main() -> int:
    api_key = load_api_key().strip()
    if not api_key:
        print("未找到 API_KEY：请确认同目录 .env 已配置")
        return 1

    if not INPUT_IMAGE.exists():
        print(f"缺少参考图: {INPUT_IMAGE}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_url = image_data_url(INPUT_IMAGE)

    print(f"API Key: {mask_secret(api_key)}")
    print(f"Base URL: {BASE_URL}")
    print(f"Endpoint: POST {BASE_URL}/chat/completions")
    print(f"thinking: {json.dumps(THINKING, ensure_ascii=False)}")
    print(f"reasoning_effort: {REASONING_EFFORT}")
    print(f"Reference image: {INPUT_IMAGE}")
    print(f"Models: {len(MODELS)}")
    print()

    summary: List[Dict[str, Any]] = []
    fail_count = 0

    for label, model_id in MODELS:
        print(f"=== {label} ({model_id}) ===")
        text_result = run_text_case(api_key, model_id)
        print_case(text_result)
        vision_result = run_vision_case(api_key, model_id, image_url)
        print_case(vision_result)
        print()

        if not text_result["ok"]:
            fail_count += 1
        if not vision_result["ok"]:
            fail_count += 1

        summary.append(
            {
                "label": label,
                "model": model_id,
                "text": text_result,
                "vision": vision_result,
            }
        )

    out_path = OUTPUT_DIR / "last_run.json"
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果已写入: {out_path}")

    total = len(MODELS) * 2
    ok_count = total - fail_count
    print(f"汇总: {ok_count}/{total} 通过")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
