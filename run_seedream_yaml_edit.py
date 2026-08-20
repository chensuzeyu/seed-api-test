#!/usr/bin/env python3
"""Edit a first-frame image with Seedream, reading prompt YAML from prompts/."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml

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

DEFAULT_PROMPT_FILE = ROOT / "prompts" / "prompts_20260820.yaml"


def load_prompt_config(path: Path) -> Dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid yaml: {path}")
    return data


def assemble_prompt(cfg: Dict, style_name: str) -> Tuple[str, Dict]:
    styles = cfg.get("styles") or {}
    style = styles.get(style_name)
    if not isinstance(style, dict):
        raise RuntimeError(f"style not found: {style_name}")
    template = str(cfg.get("template") or "")
    replacements = {
        "【目标风格】": str(style.get("目标风格") or ""),
        "【时间】": str(style.get("时间") or ""),
        "【风格效果】": str(style.get("风格效果") or ""),
    }
    prompt = template
    for token, value in replacements.items():
        prompt = prompt.replace(token, value)
    return prompt.strip(), style


def output_path(out_dir: Path, source: Path, slug: str, version: str, index: int) -> Path:
    stem = source.stem
    if stem.endswith("_first"):
        stem = stem[: -len("_first")]
    # refining a prior styled output: bottle-part2_city-skyline_v1-2 -> bottle-part2
    stem = re.sub(r"_[a-z0-9-]+_v\d+(-\d+)?(-align)?$", "", stem)
    return out_dir / f"{stem}_{slug}_{version}-{index}.jpg"


def extract_urls(data: Dict) -> List[str]:
    urls: List[str] = []
    payload = data.get("data")
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                url = item.get("url")
                if isinstance(url, str) and url:
                    urls.append(url)
    return urls


def save_image_bytes(body: Dict, dest: Path) -> None:
    urls = extract_urls(body) if isinstance(body, dict) else []
    if urls:
        r = requests.get(urls[0], timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return
    items = body.get("data", []) if isinstance(body, dict) else []
    if isinstance(items, list) and items:
        item0 = items[0]
        if isinstance(item0, dict) and isinstance(item0.get("b64_json"), str):
            import base64

            b64_value = item0["b64_json"]
            if b64_value.startswith("data:") and "," in b64_value:
                b64_value = b64_value.split(",", 1)[1]
            dest.write_bytes(base64.b64decode(b64_value))
            return
    raise RuntimeError("No image url or b64_json returned")


def invoke_one(
    api_key: str, model_id: str, image_path: Path, prompt: str, dest: Path
) -> Tuple[bool, str, Optional[Path], Dict]:
    payload = {
        "model": model_id,
        "prompt": prompt,
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

    meta = {"request": {"model": model_id, "size": "2K"}, "response": body, "saved": str(dest)}
    if resp.status_code >= 400:
        return False, f"HTTP {resp.status_code}: {body}", None, meta
    if isinstance(body, dict) and body.get("error"):
        return False, json.dumps(body.get("error"), ensure_ascii=False), None, meta

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_image_bytes(body, dest)
    except Exception as exc:
        return False, str(exc), None, meta
    return True, "ok", dest, meta


def main() -> int:
    prompt_file = DEFAULT_PROMPT_FILE
    cfg = load_prompt_config(prompt_file)
    active = cfg.get("active") or {}
    source = ROOT / str(active.get("source") or "")
    style_name = str(active.get("style") or "")
    out_dir = ROOT / str(active.get("out_dir") or "output")
    version = str(active.get("version") or "v1")
    n = int(active.get("n") or 1)

    if not source.exists():
        print(f"源图不存在: {source}")
        return 1

    prompt, style = assemble_prompt(cfg, style_name)
    slug = str(style.get("slug") or style_name)

    api_key = load_api_key()
    if not api_key:
        print("未找到 API_KEY：请配置同目录 .env")
        return 1

    model_ids = get_model_ids(api_key)
    label, model_id = pick_seedream_model(model_ids)
    print(f"API Key: {mask_secret(api_key)}")
    print(f"Seedream model: {label} / {model_id}")
    print(f"Prompt file: {prompt_file}")
    print(f"Input: {source}")
    print(f"Style: {style_name} ({slug})")
    print(f"Concurrent: {n} -> {version}-1 ... {version}-{n}")

    jobs = [(i, output_path(out_dir, source, slug, version, i)) for i in range(1, n + 1)]
    results: List[Dict] = []
    ok_count = 0

    with ThreadPoolExecutor(max_workers=n) as pool:
        future_map = {
            pool.submit(invoke_one, api_key, model_id, source, prompt, dest): (idx, dest)
            for idx, dest in jobs
        }
        for fut in as_completed(future_map):
            idx, dest = future_map[fut]
            ok, msg, saved, meta = fut.result()
            results.append({"index": idx, "ok": ok, "msg": msg, "saved": str(saved) if saved else None, "meta": meta})
            if ok and saved:
                ok_count += 1
                print(f"[OK] {version}-{idx}: {saved}")
            else:
                print(f"[FAIL] {version}-{idx}: {msg}")

    results.sort(key=lambda item: int(item["index"]))
    save_json(out_dir / f"{source.stem}_{slug}_{version}_last_run.json", {
        "prompt_file": str(prompt_file),
        "source": str(source),
        "style": style_name,
        "slug": slug,
        "version": version,
        "n": n,
        "model": model_id,
        "prompt": prompt,
        "results": results,
    })
    print(f"done: {ok_count}/{n}")
    return 0 if ok_count == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
