# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

"""
MediaPipe Pose 人体姿态骨骼检测模块（MediaPipe 1.0+ Tasks API）
33个关键点 + 骨架连线绘制
共用外部传入的摄像头对象，不自行开关摄像头
"""
import cv2
import os
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode
)
from mediapipe import Image as MPImage, ImageFormat

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "pose_landmarker_full.task")  # full精度版（lite为后备）

# 骨架连线定义（按部位分组，MediaPipe 33关键点）
POSE_CONNECTIONS = {
    "face": [
        (0, 1), (1, 2), (2, 3),       # 左眼
        (0, 4), (4, 5), (5, 6),       # 右眼
        (2, 7), (5, 8),                # 耳朵
        (9, 10),                       # 嘴
        (0, 9), (0, 10),              # 鼻到嘴
    ],
    "torso": [
        (11, 12),                      # 双肩
        (11, 23), (12, 24),           # 肩到髋
        (23, 24),                      # 双髋
    ],
    "left_arm": [
        (11, 13), (13, 15),           # 肩-肘-腕
        (15, 17), (15, 19), (15, 21), # 腕到手指
    ],
    "right_arm": [
        (12, 14), (14, 16),           # 肩-肘-腕
        (16, 18), (16, 20), (16, 22), # 腕到手指
    ],
    "left_leg": [
        (23, 25), (25, 27),           # 髋-膝-踝
        (27, 29), (29, 31),           # 踝-脚跟-脚尖
    ],
    "right_leg": [
        (24, 26), (26, 28),           # 髋-膝-踝
        (28, 30), (30, 32),           # 踝-脚跟-脚尖
    ],
}

# 各部位颜色（BGR）
COLORS = {
    "face":       (255, 255, 255),  # 白色
    "torso":      (255, 255, 0),    # 青色
    "left_arm":   (255, 128, 0),    # 蓝色
    "right_arm":  (0, 200, 0),      # 绿色
    "left_leg":   (200, 0, 200),    # 紫色
    "right_leg":  (0, 165, 255),    # 橙色
}

POINT_COLOR = (0, 0, 255)           # 红色关键点
POINT_RADIUS = 4

_landmarker = None      # 全局缓存的姿态模型（只创建一次，绝不close）
_ts_counter = 0         # 全局时间戳（VIDEO模式要求单调递增，跨运行也必须递增）


def draw_landmarks(frame, landmarks, w, h):
    """在帧上绘制骨骼和关键点"""
    for part, connections in POSE_CONNECTIONS.items():
        color = COLORS[part]
        for i, j in connections:
            lm_i = landmarks[i]
            lm_j = landmarks[j]
            if lm_i.visibility < 0.5 or lm_j.visibility < 0.5:
                continue
            x1, y1 = int(lm_i.x * w), int(lm_i.y * h)
            x2, y2 = int(lm_j.x * w), int(lm_j.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    for lm in landmarks:
        if lm.visibility < 0.5:
            continue
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), POINT_RADIUS, POINT_COLOR, -1, cv2.LINE_AA)


def preload():
    """预加载姿态模型：创建一次并缓存，绝不 close()（close会卡45秒）"""
    global _landmarker
    if _landmarker is not None:
        return
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=5,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    _landmarker = PoseLandmarker.create_from_options(opts)
    print("[姿态骨骼] 模型初始化完成")


def get_landmarker():
    """获取缓存的姿态模型（自动初始化）"""
    if _landmarker is None:
        preload()
    return _landmarker


def run_pose_detect(get_frame, callback=None):
    """
    MediaPipe Pose 实时检测（复用缓存的模型实例）
    get_frame: 获取最新帧的函数，返回 (ret, frame)
    callback(frame, people) -> bool  返回False中断
    """
    global _ts_counter
    landmarker = get_landmarker()
    last_landmarks = None  # 缓存上一次检测结果
    DETECT_INTERVAL = 2   # 每2帧检测一次（精度与流畅度平衡）

    while True:
        try:
            ret, frame = get_frame()
            if not ret or frame is None:
                break
        except Exception as e:
            print(f"[姿态骨骼] 读取帧失败: {e}")
            break

        h, w = frame.shape[:2]

        # 每隔几帧做一次检测，其余帧复用上次结果
        if _ts_counter % DETECT_INTERVAL == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)

            # 时间戳必须全局单调递增（实例复用，停止后重启也不能回退）
            _ts_counter += 1
            timestamp_ms = _ts_counter * 33
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                last_landmarks = []
                for pose_lm in result.pose_landmarks:
                    landmarks = pose_lm if isinstance(pose_lm, list) else pose_lm.landmark
                    last_landmarks.append(landmarks)
            else:
                last_landmarks = None
        else:
            _ts_counter += 1

        # 绘制骨骼（用缓存的结果）
        people = []
        if last_landmarks:
            for landmarks in last_landmarks:
                draw_landmarks(frame, landmarks, w, h)
                people.append({
                    "keypoints": [(lm.x, lm.y, lm.visibility)
                                  for lm in landmarks],
                    "count": 33,
                })

        if callback:
            if not callback(frame, people):
                break

        if cv2.waitKey(1) & 0xFF == 27:
            break