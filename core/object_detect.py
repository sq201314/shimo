# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

"""
目标检测模块（多模型支持）
扫描 models/ 目录下所有 .pt 文件，每个 .pt 为一个独立模型
模型文件以目标名称命名：tank.pt、balloon.pt、ship.pt 等
"""
import cv2
import os
import threading
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 不同目标的框颜色（BGR）
MODEL_COLORS = {
    "tank":    (0, 165, 255),   # 橙色 - 坦克
    "balloon": (255, 0, 255),   # 粉色 - 气球
    "ship":    (255, 255, 0),   # 青色 - 轮船
}
DEFAULT_COLOR = (0, 255, 0)

CONF_THRESH = 0.7

# 每个模型只识别的目标关键词（类别名包含关键词才显示）
MODEL_KEYWORDS = {
    "qiqiu1": ["balloon", "ballon", "balon", "baloon", "ball"],
    "tank": ["tank"],
    "warship": ["warship", "ship"],
}

_loaded_models = []  # [(name, model, color, classes), ...]
_device = None
_load_lock = threading.Lock()  # 保护 _loaded_models（多线程并行加载）


def _load_classes(model_path):
    """从同目录的 classes.txt 加载类别名"""
    classes_file = os.path.join(os.path.dirname(model_path), "classes.txt")
    classes = {}
    if os.path.exists(classes_file):
        with open(classes_file, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                name = line.strip()
                if name:
                    classes[i] = name
    return classes


def list_model_files():
    """列出所有待加载的目标检测模型文件 [(name, path), ...]"""
    result = []
    if not os.path.exists(MODELS_DIR):
        return result
    for fname in sorted(os.listdir(MODELS_DIR)):
        if not fname.endswith(".pt") or fname.startswith("yolo"):
            continue
        name = fname[:-3]
        result.append((name, os.path.join(MODELS_DIR, fname)))
    return result


def preload_one(pt_path):
    """加载单个模型（可多线程并行调用）"""
    global _device
    with _load_lock:
        if _device is None:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = _device

    name = os.path.splitext(os.path.basename(pt_path))[0]
    print(f"[目标检测] 加载模型: {name}")
    model = YOLO(pt_path)
    model.to(device)
    # 只预热一次（防止并发CUDA崩溃 + 提升加载速度）
    from gpulock import warmup_once

    def _warm():
        dummy = torch.zeros(1, 3, 640, 640).to(device)
        model.predict(dummy, verbose=False)
        del dummy
        torch.cuda.empty_cache()

    warmup_once(_warm)

    classes = model.names if model.names else _load_classes(pt_path)
    color = MODEL_COLORS.get(name, DEFAULT_COLOR)
    with _load_lock:
        _loaded_models.append((name, model, color, classes))
    print(f"  → {name} 类别: {classes}, 颜色: {color}")


def preload():
    """顺序预加载 models/ 下所有 .pt 模型（兼容接口）"""
    for name, path in list_model_files():
        preload_one(path)
    print(f"[目标检测] 共加载 {len(_loaded_models)} 个模型")


def get_model_names():
    """返回已加载的模型名称列表"""
    return [name for name, _, _, _ in _loaded_models]


def run_single(get_frame, model_name, callback=None):
    """
    单模型目标检测
    get_frame: 获取最新帧的函数，返回 (ret, frame)
    model_name: 模型名称（如 "tank"、"balloon"、"ship"）
    """
    target = None
    for name, model, color, classes in _loaded_models:
        if name == model_name:
            target = (name, model, color, classes)
            break

    if not target:
        print(f"[目标检测] 模型不存在: {model_name}")
        return

    name, model, color, classes = target
    print(f"[目标检测] 使用模型: {name}")

    # 计算允许显示的类别ID（只识别对应目标）
    keywords = MODEL_KEYWORDS.get(name)
    allowed_ids = None
    if keywords:
        allowed_ids = {
            cid for cid, cname in classes.items()
            if any(k in str(cname).lower() for k in keywords)
        }

    while True:
        try:
            ret, frame = get_frame()
            if not ret or frame is None:
                break
        except Exception as e:
            print(f"[目标检测] 读取帧失败: {e}")
            break

        results = model.predict(frame, conf=CONF_THRESH, verbose=False,
                                device=_device, imgsz=640)

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                # 只显示对应目标类别
                if allowed_ids is not None and cls_id not in allowed_ids:
                    continue
                conf = float(box.conf[0])
                if conf < CONF_THRESH:
                    continue
                label = classes.get(cls_id, classes.get(str(cls_id), str(cls_id)))
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2, label, conf))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

        if callback:
            if not callback(frame, detections):
                break

        if cv2.waitKey(1) & 0xFF == 27:
            break