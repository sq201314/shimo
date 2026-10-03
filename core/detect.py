# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

import cv2
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model import SimpleCNN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
CLASSES_FILE = os.path.join(BASE_DIR, "data", "labeled_images", "classes.txt")
GIF_PATH = os.path.join(BASE_DIR, "8.gif")
IMG_SIZE = 128
WIN_SIZE = 64
STEP = 32
CONF_THRESH = 0.8
BATCH_DETECT = 256   # 滑动窗口批量推理，充分利用GPU


def load_classes():
    with open(CLASSES_FILE, "r", encoding="utf-8", errors="replace") as f:
        return [c.strip() for c in f.readlines() if c.strip()]


def load_gif_frames():
    gif = Image.open(GIF_PATH)
    frames = []
    for i in range(gif.n_frames):
        gif.seek(i)
        rgb = gif.convert("RGB")
        bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        frames.append(bgr)
    return frames


def run_detect(callback=None):
    """
    callback(frame, detections) -> bool  返回False可中断
    detections: list of (x1, y1, x2, y2, class_name, confidence)
    """
    # GPU优化设置
    torch.backends.cudnn.benchmark = True

    classes = load_classes()
    num_classes = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(num_classes).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    model.to(memory_format=torch.channels_last)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    gif_frames = load_gif_frames()
    gif_count = len(gif_frames)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    frame_idx = 0
    while True:
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

        # 收集所有窗口坐标和对应的crop
        windows = []
        coords = []
        for y in range(0, bh - WIN_SIZE, STEP):
            for x in range(0, bw - WIN_SIZE, STEP):
                crop = bg[y:y+WIN_SIZE, x:x+WIN_SIZE]
                img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                windows.append(transform(img))
                coords.append((x, y, x+WIN_SIZE, y+WIN_SIZE))

        # 批量GPU推理
        detections = []
        for i in range(0, len(windows), BATCH_DETECT):
            batch = torch.stack(windows[i:i+BATCH_DETECT]).to(device, non_blocking=True)
            batch = batch.to(memory_format=torch.channels_last)
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = model(batch)
                probs = torch.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, dim=1)

            for j in range(confs.size(0)):
                if confs[j].item() > CONF_THRESH:
                    x1, y1, x2, y2 = coords[i + j]
                    detections.append((x1, y1, x2, y2,
                                       classes[preds[j].item()], confs[j].item()))

        for x1, y1, x2, y2, cls, conf in detections:
            cv2.rectangle(bg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(bg, f"{cls} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if callback:
            if not callback(bg, detections):
                break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()