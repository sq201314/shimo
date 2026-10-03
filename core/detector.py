# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

"""
目标识别 + 人体检测模块
使用 YOLO 预训练模型，共用外部摄像头
"""
import cv2
import os
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 军事目标检测模型路径
MILITARY_MODEL = os.path.join(BASE_DIR, "models", "military", "best.pt")

# 检测框颜色
COLOR_TARGET = (0, 165, 255)   # 橙色（目标识别）
COLOR_PERSON = (0, 255, 0)     # 绿色（人体检测）
CONF_THRESH = 0.4

# 简化类别名称（去掉国家前缀，只保留类型）
SHORT_NAMES = {
    "PLA Amphibious assault vehicle": "两栖突击车",
    "PLA Infantry fighting vehicle": "步战车",
    "PLA Tank": "坦克",
    "PLA Vehicle": "车辆",
    "RUS Infantry fighting vehicle": "步战车",
    "RUS TANK": "坦克",
    "RUS Vehicle": "车辆",
    "TW Infantry fighting vehicle": "步战车",
    "TW Tank": "坦克",
    "US Amphibious assault vehicle": "两栖突击车",
    "US Infantry fighting vehicle": "步战车",
    "US Tank": "坦克",
    "US Vehicle": "车辆",
}

_model = None
_device = None


def preload_models():
    """预加载 YOLO 模型到 GPU，供所有检测功能共用"""
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = YOLO(MILITARY_MODEL)
    _model.to(_device)
    # 预热推理
    dummy = torch.zeros(1, 3, 640, 640).to(_device)
    _model.predict(dummy, verbose=False)
    print(f"[预加载] 军事检测模型已加载到 {_device}")
    return _model


def get_model():
    """获取已加载的模型"""
    global _model
    if _model is None:
        preload_models()
    return _model


def run_object_detect(cap, callback=None):
    """
    目标识别：检测所有军事目标（坦克/步战车/两栖车/车辆等）
    callback(frame, detections) -> bool
    """
    model = get_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF_THRESH, verbose=False,
                                device=_device, imgsz=640)

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                raw_name = model.names.get(cls_id, str(cls_id))
                label = SHORT_NAMES.get(raw_name, raw_name)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, label, conf))
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_TARGET, 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, COLOR_TARGET, 2)

        if callback:
            if not callback(frame, detections):
                break

        if cv2.waitKey(1) & 0xFF == 27:
            break


def run_person_detect(cap, callback=None):
    """
    人体检测：检测画面中的所有人
    callback(frame, detections) -> bool
    """
    model = get_model()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF_THRESH, verbose=False,
                                device=_device, imgsz=640)

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections.append((x1, y1, x2, y2, "人体", conf))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)
                    cv2.putText(frame, f"人体 {conf:.2f}",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, COLOR_PERSON, 2)

        frame_idx += 1
        if callback:
            if not callback(frame, detections):
                break

        if cv2.waitKey(1) & 0xFF == 27:
            break