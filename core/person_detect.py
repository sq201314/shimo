# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

"""
人体检测模块
YOLO 检测人体边框（只画框，不画骨骼）
置信度 >= 0.7 才显示框框
"""
import cv2
import os
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 使用 COCO 预训练模型（包含 person 类别）
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11n.pt")

# 检测参数
CONF_THRESH = 0.7  # 置信度 >= 0.7 才显示
COLOR_PERSON = (0, 255, 0)  # 绿色边框

_model = None
_device = None


def _load_model():
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = YOLO(MODEL_PATH)
        _model.to(_device)
        # 只预热一次（防止并发CUDA崩溃 + 提升加载速度）
        from gpulock import warmup_once

        def _warm():
            dummy = torch.zeros(1, 3, 640, 640).to(_device)
            _model.predict(dummy, verbose=False)
            del dummy
            torch.cuda.empty_cache()

        warmup_once(_warm)
        print(f"[人体检测] 模型已加载到 {_device}")


def preload():
    """预加载模型"""
    _load_model()


def run(get_frame, callback=None):
    """
    人体检测主循环
    get_frame: 获取最新帧的函数，返回 (ret, frame)
    callback(frame, detections) -> bool  返回False中断
    """
    _load_model()

    while True:
        try:
            ret, frame = get_frame()
            if not ret or frame is None:
                break
        except Exception as e:
            print(f"[人体检测] 读取帧失败: {e}")
            break

        results = _model.predict(frame, conf=CONF_THRESH, verbose=False,
                                 device=_device, imgsz=640)

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = _model.names.get(cls_id, "")
                # 只检测 person 类别
                if cls_id == 0 or cls_name.lower() == "person":
                    conf = float(box.conf[0])
                    if conf < CONF_THRESH:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2, "人体", conf))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)
                    cv2.putText(frame, f"人体 {conf:.2f}",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, COLOR_PERSON, 2)

        if callback:
            if not callback(frame, detections):
                break

        if cv2.waitKey(1) & 0xFF == 27:
            break