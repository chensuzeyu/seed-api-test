#!/usr/bin/env python3
"""Extract the first frame from the Seedance2 style-transfer test video."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from ark_common import INPUT_VIDEO, OUTPUT_SEEDANCE2, save_json

OUTPUT_FRAME = OUTPUT_SEEDANCE2 / "26-08-19_Origin_4s_first.jpg"
META_JSON = OUTPUT_SEEDANCE2 / "first_frame_meta.json"


def extract_first_frame(video_path: Path, out_path: Path) -> dict:
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_sec = (frame_count / fps) if fps > 0 else 0.0

        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("读取首帧失败")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), frame):
            raise RuntimeError(f"写入首帧失败: {out_path}")

        return {
            "video": str(video_path),
            "first_frame": str(out_path),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": round(duration_sec, 3),
            "duration_int_for_seedance": max(4, min(15, int(round(duration_sec)))),
        }
    finally:
        cap.release()


def main() -> int:
    meta = extract_first_frame(INPUT_VIDEO, OUTPUT_FRAME)
    save_json(META_JSON, meta)
    print("[OK] first frame extracted")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
