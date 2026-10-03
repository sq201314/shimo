# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

import cv2
import os
import time
from datetime import datetime
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(BASE_DIR, "data", "images")
TOTAL = 20
DURATION = 3.0
GIF_PATH = os.path.join(BASE_DIR, "8.gif")


def load_gif_frames():
    gif = Image.open(GIF_PATH)
    frames = []
    for i in range(gif.n_frames):
        gif.seek(i)
        rgb = gif.convert("RGB")
        bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        frames.append(bgr)
    return frames


def run_capture(callback=None):
    """
    callback(frame, count, total) -> bool  返回False可中断
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    gif_frames = load_gif_frames()
    gif_count = len(gif_frames)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return 0

    count = 0
    interval = DURATION / TOTAL
    start = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_idx = 0

    while count < TOTAL:
        ret, cam_frame = cap.read()
        if not ret:
            break

        bg = gif_frames[frame_idx % gif_count].copy()
        frame_idx += 1

        bh, bw = bg.shape[:2]
        cam_resized = cv2.resize(cam_frame, (bw, bh))
        ch, cw = cam_resized.shape[:2]
        x_off = (bw - cw) // 2
        y_off = (bh - ch) // 2
        bg[y_off:y_off+ch, x_off:x_off+cw] = cam_resized

        elapsed = time.time() - start
        cv2.putText(bg, f"{count + 1}/{TOTAL}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        if callback:
            if not callback(bg, count, TOTAL):
                break

        if elapsed >= count * interval:
            fname = f"img_{ts}_{count + 1:03d}.png"
            cv2.imwrite(os.path.join(SAVE_DIR, fname), bg)
            count += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    return count