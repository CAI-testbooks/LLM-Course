from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
from PIL import Image


@dataclass
class VideoSamplingResult:
    frames: List[Image.Image]
    fps_used: float
    n_frames: int
    duration_sec: Optional[float]


def sample_video_frames(
    video_path: str,
    fps: float = 1.0,
    max_frames: int = 16,
    max_side: int = 1024,
) -> VideoSamplingResult:
    """
    Sample frames from a video file.

    - fps: target sampling FPS (approx).
    - max_frames: hard cap to avoid OOM.
    - max_side: resize if needed for speed/memory.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Get metadata (might be 0 for some videos)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration_sec = None
    if native_fps > 0 and frame_count > 0:
        duration_sec = float(frame_count / native_fps)

    # Decide stride
    if native_fps <= 0:
        # fall back: read every N frames
        stride = 1
        fps_used = fps
    else:
        stride = max(1, int(round(native_fps / max(fps, 1e-6))))
        fps_used = native_fps / stride

    frames: List[Image.Image] = []
    idx = 0
    grabbed = True
    while grabbed and len(frames) < max_frames:
        grabbed = cap.grab()
        if not grabbed:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            # Resize if too big
            w, h = img.size
            scale = max(w, h) / float(max_side)
            if scale > 1.0:
                img = img.resize((int(w / scale), int(h / scale)))

            frames.append(img)
        idx += 1

    cap.release()
    return VideoSamplingResult(frames=frames, fps_used=fps_used, n_frames=len(frames), duration_sec=duration_sec)
