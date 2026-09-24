#!/usr/bin/env python3
"""Shared helpers for Ark API smoke tests in this directory."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
OUTPUT_SEEDANCE2 = ROOT / "output" / "seedance2_test"
INPUT_VIDEO = ROOT / "inputs" / "seedance2_test" / "26-08-19_Origin_4s.mp4"


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
    return (os.environ.get("API_KEY") or env_values.get("API_KEY", "")).strip()


def mask_secret(secret: str) -> str:
    if len(secret) <= 10:
        return "***"
    return f"{secret[:6]}...{secret[-4:]}"


def auth_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
    }


def get_model_ids(api_key: str) -> List[str]:
    resp = requests.get(
        f"{BASE_URL}/models",
        headers=auth_headers(api_key),
        timeout=30,
    )
    resp.raise_for_status()
    out: List[str] = []
    for item in resp.json().get("data", []):
        model_id = item.get("id")
        if isinstance(model_id, str):
            out.append(model_id)
    return out


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "jpg":
        suffix = "jpeg"
    mime = f"image/{suffix or 'jpeg'}"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def video_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def download_url(url: str, dest: Path, timeout: int = 300) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# 首图编辑默认：Seedream 5.0 Pro（账号当前开通的 endpoint）
DEFAULT_SEEDREAM_MODEL_ID = "doubao-seedream-5-0-pro-260628"


def pick_seedream_model(model_ids: List[str]) -> tuple[str, str]:
    """Return (label, model_id). Seedream 5.0 Pro only."""
    if DEFAULT_SEEDREAM_MODEL_ID in model_ids:
        return "Seedream 5.0 Pro", DEFAULT_SEEDREAM_MODEL_ID

    pro_candidates = [
        mid
        for mid in model_ids
        if "seedream" in mid.lower() and "5" in mid and "pro" in mid.lower()
    ]
    if pro_candidates:
        pro_candidates.sort(reverse=True)
        return "Seedream 5.0 Pro", pro_candidates[0]

    raise RuntimeError(
        "账户未开通 Seedream 5.0 Pro。可见 seedream 模型: "
        + ", ".join(m for m in model_ids if "seedream" in m.lower())
    )


def pick_seedance2_model(model_ids: List[str]) -> tuple[str, str]:
    """Prefer standard Seedance 2.0 over fast/mini."""
    preferred = [
        "doubao-seedance-2-0-260128",
        "doubao-seedance-2-0",
        "seedance-2-0-260128",
    ]
    for mid in preferred:
        if mid in model_ids:
            return "Seedance 2.0", mid

    standard = [
        mid
        for mid in model_ids
        if "seedance" in mid.lower()
        and "2" in mid
        and "fast" not in mid.lower()
        and "mini" not in mid.lower()
    ]
    if standard:
        standard.sort(reverse=True)
        return "Seedance 2.0", standard[0]

    any_2 = [mid for mid in model_ids if "seedance" in mid.lower() and "2" in mid]
    if any_2:
        any_2.sort(key=lambda x: ("fast" in x.lower(), "mini" in x.lower(), x))
        return "Seedance 2.0 (alt)", any_2[0]

    raise RuntimeError(
        "账户未开通 Seedance 2.0。可见 seedance 模型: "
        + ", ".join(m for m in model_ids if "seedance" in m.lower())
        or "(无)"
    )


def load_tos_config() -> Dict[str, str]:
    env = load_dotenv_file(ENV_FILE)
    cfg = {
        "access_key_id": os.environ.get("ACCESS_KEY_ID") or env.get("ACCESS_KEY_ID", ""),
        "access_key_secret": os.environ.get("ACCESS_KEY_SECRET")
        or env.get("ACCESS_KEY_SECRET", ""),
        "endpoint": os.environ.get("TOS_ENDPOINT")
        or env.get("TOS_ENDPOINT", "tos-cn-beijing.volces.com"),
        "region": os.environ.get("TOS_REGION") or env.get("TOS_REGION", "cn-beijing"),
        "bucket": os.environ.get("TOS_BUCKET") or env.get("TOS_BUCKET", "knowin-oss"),
        "presign_expire": os.environ.get("TOS_PRESIGN_EXPIRE")
        or env.get("TOS_PRESIGN_EXPIRE", "7200"),
    }
    if not cfg["access_key_id"] or not cfg["access_key_secret"]:
        raise RuntimeError("缺少 TOS ACCESS_KEY_ID / ACCESS_KEY_SECRET（请写入 .env）")
    return cfg


def tos_client():
    from tos import TosClientV2

    cfg = load_tos_config()
    return (
        TosClientV2(
            cfg["access_key_id"],
            cfg["access_key_secret"],
            cfg["endpoint"],
            cfg["region"],
        ),
        cfg,
    )


def upload_to_tos_and_presign(local_path: Path, object_key: str) -> str:
    """
    Upload a local file to private TOS bucket and return a GET presigned HTTPS URL.
    Seedance reference_video requires a publicly fetchable web URL.
    """
    from tos import HttpMethodType

    client, cfg = tos_client()
    expire = int(cfg["presign_expire"])
    with local_path.open("rb") as fh:
        client.put_object(bucket=cfg["bucket"], key=object_key, content=fh)
    resp = client.pre_signed_url(
        HttpMethodType.Http_Method_Get,
        cfg["bucket"],
        object_key,
        expires=expire,
    )
    url = getattr(resp, "signed_url", None) or getattr(resp, "sign_url", None)
    if not isinstance(url, str) or not url.startswith("http"):
        raise RuntimeError(f"TOS presign failed for {object_key}: {resp!r}")
    return url