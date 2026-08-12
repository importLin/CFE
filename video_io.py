# -*- coding: utf-8 -*-
"""Minimal video I/O. Convention: reads return RGB, writes take RGB
(cv2 stores BGR on disk; conversions are handled here, nowhere else)."""

import cv2
import numpy as np


def probe(path):
    """Container metadata without decoding the frames -> dict with
    frame_count, width, height and fps. `fps` falls back to 25.0 when the
    container reports 0 (some AVI files do), so writers always get a usable
    value."""
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"cannot open {path}"
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
    }
    cap.release()
    return info


def read_video(path):
    """-> (T, H, W, C) uint8 RGB. Decodes all frames (metadata-safe).

    Frames are read sequentially rather than seeked by index, because the
    frame count advertised by the container is often wrong -- decoding until
    exhaustion is the only reliable way to get the true length."""
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"cannot open {path}"
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    assert frames, f"no decodable frames in {path}"
    return np.stack(frames)


def _center_crop_square(frame):
    """Center-crop an (H, W, C) frame to a square of its short side.
    No-op if the frame is already square. -> (S, S, C) with S = min(H, W)."""
    h, w = frame.shape[:2]
    if h == w:
        return frame
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return frame[y0:y0 + s, x0:x0 + s]


def read_clip(path, num_frames=32, size=224):
    """Preprocessed read: uniform-sample `num_frames` across the whole video,
    then, if the source geometry differs from `size` x `size`, center-crop the
    frames to their short side (a square) and resize. Cropping before resizing
    preserves the aspect ratio, so content is not stretched. Idempotent on
    already-preprocessed clips. -> (num_frames, size, size, C) uint8 RGB.

    Being idempotent matters: encryption and verification both call this, so a
    clip that already matches the model spec must pass through untouched --
    otherwise resampling would perturb the ciphertext."""
    frames = read_video(path)
    idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    frames = frames[idx]
    if frames.shape[1] != size or frames.shape[2] != size:
        frames = np.stack([
            cv2.resize(_center_crop_square(f), (size, size),
                       interpolation=cv2.INTER_AREA)
            for f in frames])
    return frames


def write_video(path, frames, fps=25.0, fourcc="FFV1"):
    """frames: (T, H, W, C) uint8 RGB. FFV1 = lossless.

    Losslessness is required here: the consistency check only holds if the
    ciphertext survives the round trip to disk bit-for-bit. Compression
    robustness (Motion-JPEG / H.264) is evaluated separately in the paper."""
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    assert vw.isOpened(), f"cannot open writer for {path}"
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
