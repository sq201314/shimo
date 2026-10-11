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
import re
import threading
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 模型文件名 -> 目标类型（用于取颜色和关键词）
MODEL_TYPE = {
    "qiqiu1": "balloon",
    "tank": "tank",
    "warship": "ship",
}

# 不同目标类型的框颜色（BGR），键为 MODEL_TYPE 的值
MODEL_COLORS = {
    "tank":    (0, 165, 255),   # 橙色 - 坦克
    "balloon": (255, 0, 255),   # 粉色 - 气球
    "ship":    (255, 255, 0),   # 青色 - 轮船
}
DEFAULT_COLOR = (0, 255, 0)

CONF_THRESH = 0.6  # 置信度 >= 0.6 才显示

# 每个目标类型只识别的关键词（类别名整词匹配关键词才显示）
# 不放 "ball"：会误纳 basketball/football 等
TYPE_KEYWORDS = {
    "balloon": ["balloon", "ballon", "balon", "baloon"],
    "tank": ["tank"],
    "ship": ["warship", "ship"],
}


def _get_type(model_name):
    """模型名 -> 目标类型（未知模型用模型名兜底）"""
    return MODEL_TYPE.get(model_name, model_name)


def _match_keywords(cls_name, keywords):
    """整词匹配：'basketball' 不命中 'ball'，'hot air balloon' 命中 'balloon'"""
    words = set(re.split(r"[^a-z0-9]+", str(cls_name).lower()))
    return any(k in words for k in keywords)

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
        dummy = torch.zeros(1, 3, 1440, 1440).to(device)
        model.predict(dummy, verbose=False, imgsz=1440)
        del dummy
        torch.cuda.empty_cache()

    warmup_once(_warm)

    classes = model.names if model.names else _load_classes(pt_path)
    color = MODEL_COLORS.get(_get_type(name), DEFAULT_COLOR)
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


def run_single(get_frame, model_name, callback=None, box_smooth=None):
    """
    单模型目标检测
    get_frame: 获取最新帧的函数，返回 (ret, frame)
    model_name: 模型名称（如 "tank"、"balloon"、"ship"）
    box_smooth: 可选 zoom_track.BoxSmoother，画框前平滑坐标（防放大后框抖）
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

    # 计算允许显示的类别ID（只识别对应目标类型）
    keywords = TYPE_KEYWORDS.get(_get_type(name))
    allowed_ids = None
    if keywords:
        allowed_ids = {
            cid for cid, cname in classes.items()
            if _match_keywords(cname, keywords)
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
                                device=_device, imgsz=1440)

        raw = []
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
                raw.append((x1, y1, x2, y2, label, conf))
        # 画框前平滑坐标（检测框逐帧抖动经放大后会变成画面/框抖动）
        detections = box_smooth.smooth(raw) if box_smooth else raw
        for x1, y1, x2, y2, label, conf in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        if callback:
            if not callback(frame, detections):
                break

        if cv2.waitKey(1) & 0xFF == 27:
            break