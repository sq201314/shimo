# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

"""
自动裁剪放大跟踪模块（仅目标检测 / 人体检测接入，姿态骨骼不接入）

流程：
- 检测始终在原尺寸全画面帧上进行（帧缓冲取帧，与显示裁剪无关）
- 识别到目标后，按主目标位置自动裁剪周围区域放大显示
- 裁剪窗口中心对准主目标并做时间基低通平滑：滤掉检测框逐帧抖动
  （否则抖动经放大倍数放大成画面抖动），目标静止时收敛到正中；
  目标贴画面边缘时窗口贴边，此时目标无法居中
- 窗口尺寸按时间平滑（与帧率无关，推近速度恒定）；
  但目标突然变大（靠近镜头）时窗口立即扩大以保证目标完整，此时会跳变
- 窗口宽高比始终 = 原画面宽高比（平滑与硬下限都保持，显示画幅形状不变）
- 目标短暂丢失（≤ LOST_HOLD 秒）冻结窗口继续裁剪，防止一闪一闪；
  真正丢失超过 LOST_HOLD 秒 → 立即回到原比例全画面
- BoxSmoother：检测框坐标也做同样的时间基平滑（检测模块画框用），
  否则画面稳住后框会相对画面抖
- 放大跟踪期间由主程序在预览框右下角固定位置显示原画面小窗（PIP_SIZE×PIP_SIZE）
- 前置条件：frame 为非空 H×W×3 BGR 数组；detections 为
  [(x1, y1, x2, y2, label, conf), ...] 形式的 list（只读，不会被修改）
"""
import math
import time

import cv2
import numpy as np

__all__ = ["ZoomTracker", "BoxSmoother", "build_origin_pip",
           "PIP_SIZE", "PIP_MARGIN", "PIP_ON", "PIP_OFF",
           "MARGIN", "MAX_ZOOM", "MIN_CROP_W", "TIME_CONST",
           "CENTER_TIME_CONST", "MAX_DT", "LOST_HOLD"]

# 裁剪窗口相对目标尺寸的边距（目标四周留白倍数）
MARGIN = 2.0
# 最大放大倍数（分辨率足够时的上限；实际倍数随画面大小自适应，见 MIN_CROP_W）
MAX_ZOOM = 4.0
# 裁剪窗口最小宽度（实际像素）：放大倍数按实际画面大小处理——
# 取 max(画面宽/MAX_ZOOM, MIN_CROP_W)，分辨率不足就少放大，防止放大发糊。
# 摄像头预设最高 2K 时的实际倍数：
#   640 → 2x（裁剪320px）  1280 → 4x（裁剪320px）  2K 2560 → 4x（裁剪640px）
MIN_CROP_W = 320
# 尺寸平滑时间常数（秒）：与帧率无关，约 0.15 秒推近到目标尺寸的 ~63%
TIME_CONST = 0.15
# 中心平滑时间常数（秒）：低通滤掉检测框逐帧抖动（放大后抖动会被放大数倍），
# 目标静止时收敛到正中，运动时约 0.2 秒跟随滞后
CENTER_TIME_CONST = 0.2
# 单帧时间步上限（秒）：程序停顿后不产生瞬间大跳变
MAX_DT = 0.1
# 目标丢失后保持裁剪窗口的秒数（短暂丢失不回原画面，防闪烁）
LOST_HOLD = 0.7

# 预览框右下角原画面小窗（仅供主程序定位覆盖标签用，模块内不使用）
PIP_SIZE = 200
PIP_MARGIN = 6
# 小窗显示/隐藏滞回阈值（裁剪宽 < 画面宽×PIP_ON 显示，> ×PIP_OFF 隐藏）
# 宽高比恒定 ⇒ 宽比即高比，只看宽度即可
PIP_ON = 0.90
PIP_OFF = 0.97

# 小窗画布异常输入时的兜底黑画布
_EMPTY = np.zeros((1, 1, 3), np.uint8)


def build_origin_pip(frame, size=PIP_SIZE):
    """
    把原画面等比缩进 size×size 小窗画布（黑底居中，不变形）
    返回 size×size 画布，由主程序显示在预览框右下角固定位置
    非法输入（None/空帧/非正size）返回兜底黑画布，不抛异常
    """
    def _blank(n):
        return np.zeros((max(int(n), 1), max(int(n), 1), 3), np.uint8)

    size = int(size)  # 防 float（copyMakeBorder 的边界参数要求 int）
    if size < 1:
        return _EMPTY.copy()
    if frame is None or frame.size == 0:
        return _blank(size)
    # 统一转成 3 通道 BGR；无法识别的通道数兜底黑画布
    if frame.ndim == 2:  # 灰度 H×W → BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1:  # H×W×1 → BGR
        frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:  # BGRA → BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        return _blank(size)
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    tw, th = max(1, int(w * scale)), max(1, int(h * scale))
    thumb = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
    # 黑边补齐成正方形（letterbox），避免拉伸变形
    pad_l = (size - tw) // 2
    pad_t = (size - th) // 2
    return cv2.copyMakeBorder(thumb, pad_t, size - th - pad_t,
                              pad_l, size - tw - pad_l,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


class BoxSmoother:
    """
    检测框坐标时间基 EMA 平滑（供检测模块画框前调用）
    滤掉检测框逐帧随机抖动——否则放大后框相对画面乱抖
    多框时按"离上次中心最近"配对（与主目标选择规则一致），
    新出现/配不上的框直接用原始坐标（不从别处飞过来）
    """

    def __init__(self, time_const=CENTER_TIME_CONST, max_dt=MAX_DT):
        self.time_const = time_const
        self.max_dt = max_dt
        self._tick = None
        self._boxes = []  # 上一帧平滑结果 [(cx, cy, (x1,y1,x2,y2)), ...]

    def reset(self):
        self._tick = None
        self._boxes = []

    def smooth(self, detections):
        """
        输入 [(x1, y1, x2, y2, label, conf), ...]（原始坐标）
        返回同结构 list，坐标为平滑后的整数
        """
        if detections is None or len(detections) == 0:
            self._boxes = []
            return []
        now = time.monotonic()
        dt = 0.0 if self._tick is None else min(now - self._tick, self.max_dt)
        self._tick = now
        if not self._boxes:
            k = 1.0  # 首帧：直接用原始坐标
        else:
            k = 1.0 - math.exp(-dt / self.time_const) if dt > 0 else 0.0

        used = [False] * len(self._boxes)
        out, new_boxes = [], []
        for d in detections:
            cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
            # 配对：离该框中心最近且未被占用的上一帧平滑框
            best, best_dist = None, None
            for i, (pcx, pcy, _) in enumerate(self._boxes):
                if used[i]:
                    continue
                dist = (pcx - cx) ** 2 + (pcy - cy) ** 2
                if best_dist is None or dist < best_dist:
                    best, best_dist = i, dist
            if best is None or k >= 1.0:
                s = (float(d[0]), float(d[1]), float(d[2]), float(d[3]))
            else:
                used[best] = True
                prev = self._boxes[best][2]
                s = tuple(p + (r - p) * k for p, r in zip(prev, d[:4]))
            new_boxes.append(((s[0] + s[2]) / 2, (s[1] + s[3]) / 2, s))
            out.append((int(round(s[0])), int(round(s[1])),
                        int(round(s[2])), int(round(s[3])), *d[4:]))
        self._boxes = new_boxes
        return out


class ZoomTracker:
    """单个检测会话一个实例：有目标裁剪放大（目标居中），丢失防闪烁后回原比例"""

    def __init__(self, margin=MARGIN, max_zoom=MAX_ZOOM,
                 min_crop_w=MIN_CROP_W, time_const=TIME_CONST,
                 center_time_const=CENTER_TIME_CONST,
                 lost_hold=LOST_HOLD, max_dt=MAX_DT):
        self.margin = margin
        self.max_zoom = max_zoom
        self.min_crop_w = min_crop_w
        self.time_const = time_const
        self.center_time_const = center_time_const
        self.lost_hold = lost_hold
        self.max_dt = max_dt
        self._size = None          # 平滑后的窗口宽 (w)；None=未跟踪
        self._rect = None          # 当前实际裁剪窗口 (x, y, w, h)（丢失冻结用）
        self._center = None        # 平滑后的窗口中心（抑制检测框逐帧抖动）
        self._last_seen = 0.0      # 最后一次识别到目标的时刻
        self._last_tick = None     # 上次 apply 的时刻（时间平滑用）
        self._primary_center = None  # 主目标中心（防多目标来回切换闪烁）
        self._pip_on = False       # 小窗显示状态（带滞回防闪烁）

    def reset(self):
        """清空跟踪状态（回到原比例、隐藏小窗、清时间步）"""
        self._size = None
        self._rect = None
        self._center = None
        self._last_seen = 0.0
        self._last_tick = None  # 重新识别时首帧 dt=0，不会产生推近跳变
        self._primary_center = None
        self._pip_on = False

    def apply(self, frame, detections):
        """
        有目标 → 裁剪放大主目标（居中跟随）；短暂丢失 → 冻结窗口；真正丢失 → 原比例
        frame: BGR帧（检测模块已画好框）
        detections: [(x1, y1, x2, y2, label, conf), ...]（帧原始坐标，只读）
        返回 (显示帧, 平移到裁剪坐标的 detections, 是否显示原画面小窗)
        第三项由小窗滞回状态决定（即"放大到位"才显示小窗）
        """
        h, w = frame.shape[:2]
        now = time.monotonic()
        # 时间步：封顶 max_dt，防程序停顿后的瞬间大跳变
        dt = 0.0 if self._last_tick is None else \
            min(now - self._last_tick, self.max_dt)
        self._last_tick = now

        if detections is None or len(detections) == 0:
            if self._rect is None:
                return frame, detections, False  # 未跟踪：原比例、无小窗
            if now - self._last_seen > self.lost_hold:
                self.reset()  # 真正丢失 → 立即回到原比例
                return frame, detections, False
            # 短暂丢失：冻结窗口继续裁剪（防止一闪一闪）
            crop, shifted = self._crop(frame, self._rect, detections)
            return crop, shifted, self._pip_on

        self._last_seen = now
        x1, y1, x2, y2 = self._pick_primary(detections)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

        # 中心低通平滑：检测框每帧有几像素随机抖动，直接吸附会经放大倍数
        # 放大成画面抖动；用时间基 EMA 滤掉噪声，目标静止时收敛到正中
        if self._center is None:
            self._center = (cx, cy)  # 首帧直接对准
        else:
            ck = 1.0 - math.exp(-dt / self.center_time_const) if dt > 0 else 0.0
            self._center = (self._center[0] + (cx - self._center[0]) * ck,
                            self._center[1] + (cy - self._center[1]) * ck)
        cx, cy = self._center

        # 期望窗口宽：保持画面宽高比、包住目标×边距、
        # 放大上限按实际画面大小处理 = max(倍数上限, 最小裁剪像素)：
        # 分辨率够 → 满 MAX_ZOOM 倍；分辨率不够 → 按 MIN_CROP_W 像素兜底不发糊
        aspect = w / h
        want_w = max(bw * self.margin, bh * self.margin * aspect,
                     w / self.max_zoom,
                     min(float(self.min_crop_w), w))
        want_w = min(want_w, float(w))

        if self._size is None:
            # 首次识别：第一帧保持全画面（k=0 零跳变），下一帧起平滑推近
            self._size = float(w)
            k = 0.0
        else:
            # 按时间的指数平滑（与帧率无关，推镜速度恒定）
            k = 1.0 - math.exp(-dt / self.time_const) if dt > 0 else 0.0
        sw = self._size + (want_w - self._size) * k
        # 硬下限（保持宽高比）：窗口任何时刻都不小于目标框×1.1，
        # 目标突然变大时立即扩大，放大过程中目标完整可见；
        # 高度下限经 aspect 折算，sw/sh 恒等于 w/h 不破坏画幅形状
        sw = max(sw, min(max(bw, bh * aspect) * 1.1, w))
        self._size = sw
        sh = sw / aspect

        # 窗口中心对准（平滑后的）主目标 → 目标保持在画面中间且不随检测噪声抖动
        # （静止收敛正中、运动约 0.2s 跟随；贴画面边缘时窗口贴边）
        x = min(max(cx - sw / 2, 0.0), w - sw)
        y = min(max(cy - sh / 2, 0.0), h - sh)
        # 中心平滑有滞后：目标大幅跳变/移动期间，窗口必须始终完整包住目标框
        # （在 [x2-sw, x1] 范围内微调窗口位置；收敛后此修正不生效）
        lo, hi = max(x2 - sw, 0.0), min(x1, w - sw)
        if lo <= hi:
            x = min(max(x, lo), hi)
        lo, hi = max(y2 - sh, 0.0), min(y1, h - sh)
        if lo <= hi:
            y = min(max(y, lo), hi)
        self._rect = (x, y, sw, sh)

        # 小窗显示状态带滞回：防止在阈值附近抖动导致小窗闪烁
        # （宽高比恒定，sw/w 与 sh/h 等价，只看宽度即可）
        if sw < w * PIP_ON:
            self._pip_on = True
        elif sw > w * PIP_OFF:
            self._pip_on = False

        crop, shifted = self._crop(frame, self._rect, detections)
        return crop, shifted, self._pip_on

    def _pick_primary(self, detections):
        """
        主目标选择（稳定防闪烁）：
        已跟踪过 → 取离上次主目标中心最近的框（多目标不会来回切换）；
        首次识别 → 取面积最大框（与锁定红框同一规则，abs 防坐标反向）
        返回前规范化坐标（x1≤x2, y1≤y2），保证 bw/bh 非负
        """
        if self._primary_center is None:
            d = max(detections, key=lambda t: abs((t[2] - t[0]) * (t[3] - t[1])))
        else:
            px, py = self._primary_center
            d = min(detections, key=lambda t:
                    (((t[0] + t[2]) / 2 - px) ** 2
                     + ((t[1] + t[3]) / 2 - py) ** 2))
        x1, x2 = (d[0], d[2]) if d[0] <= d[2] else (d[2], d[0])
        y1, y2 = (d[1], d[3]) if d[1] <= d[3] else (d[3], d[1])
        self._primary_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        return x1, y1, x2, y2

    @staticmethod
    def _crop(frame, rect, detections):
        """
        按窗口裁剪画面并平移检测坐标（窗口=全画面时裁剪结果与原帧一致）
        注意：返回的 crop 是 frame 的视图而非拷贝，必须在同一帧内消费完毕
        （本项目调用链：get_frame 已复制帧缓冲、显示在同帧内完成，满足此约束）
        """
        h, w = frame.shape[:2]
        x, y, cw, ch = rect
        iw = min(max(1, int(round(cw))), w)
        ih = min(max(1, int(round(ch))), h)
        ix = max(0, min(int(round(x)), w - iw))
        iy = max(0, min(int(round(y)), h - ih))
        crop = frame[iy:iy + ih, ix:ix + iw]
        shifted = [(d[0] - ix, d[1] - iy, d[2] - ix, d[3] - iy, *d[4:])
                   for d in detections]
        return crop, shifted
