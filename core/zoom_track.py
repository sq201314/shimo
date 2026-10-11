# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

"""
自动裁剪放大跟踪模块（仅目标检测 / 人体检测接入，姿态骨骼不接入）

流程：
- 检测始终在原尺寸全画面帧上进行（帧缓冲取帧，与显示裁剪无关）
- 识别到目标后，按主目标位置自动裁剪周围区域放大显示
- 裁剪窗口中心始终对准主目标（放大部分始终显示在画面中间），
  只有窗口尺寸平滑变化（形成平滑推近动画，不瞬间跳变）
- 目标短暂丢失（≤ LOST_HOLD 秒）冻结窗口继续裁剪，防止一闪一闪；
  真正丢失超过 LOST_HOLD 秒 → 立即回到原比例全画面
- 放大跟踪期间由主程序在预览框右下角固定位置显示原画面小窗（PIP_SIZE×PIP_SIZE）
"""
import time

import cv2

# 裁剪窗口相对目标尺寸的边距（目标四周留白倍数）
MARGIN = 2.0
# 最大放大倍数（裁剪宽度不小于画面宽 / MAX_ZOOM，目标本来就大时则不放大）
MAX_ZOOM = 4.0
# 窗口尺寸平滑系数：每帧向目标尺寸靠近的比例（越小越平滑）
ALPHA = 0.18
# 目标丢失后保持裁剪窗口的秒数（短暂丢失不回原画面，防闪烁）
LOST_HOLD = 0.7

# 预览框右下角原画面小窗（主程序按此常量摆放固定位置的覆盖标签）
PIP_SIZE = 200
PIP_MARGIN = 6
# 小窗显示/隐藏滞回阈值（裁剪宽 < 画面宽×PIP_ON 显示，> ×PIP_OFF 隐藏）
PIP_ON = 0.90
PIP_OFF = 0.97


def build_origin_pip(frame, size=PIP_SIZE):
    """
    把原画面等比缩进 size×size 小窗画布（黑底居中，不变形）
    返回 size×size 画布，由主程序显示在预览框右下角固定位置
    """
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


class ZoomTracker:
    """单个检测会话一个实例：有目标裁剪放大（目标居中），丢失防闪烁后回原比例"""

    def __init__(self, margin=MARGIN, max_zoom=MAX_ZOOM, alpha=ALPHA,
                 lost_hold=LOST_HOLD):
        self.margin = margin
        self.max_zoom = max_zoom
        self.alpha = alpha
        self.lost_hold = lost_hold
        self._size = None          # 平滑后的窗口尺寸 (w, h)；None=未跟踪
        self._rect = None          # 当前实际裁剪窗口 (x, y, w, h)（丢失冻结用）
        self._last_seen = 0.0      # 最后一次识别到目标的时刻
        self._primary_center = None  # 主目标中心（防多目标来回切换闪烁）
        self._pip_on = False       # 小窗显示状态（带滞回防闪烁）

    def reset(self):
        """清空跟踪状态（回到原比例、隐藏小窗）"""
        self._size = None
        self._rect = None
        self._primary_center = None
        self._pip_on = False

    def apply(self, frame, detections):
        """
        有目标 → 裁剪放大主目标（居中跟随）；短暂丢失 → 冻结窗口；真正丢失 → 原比例
        frame: BGR帧（检测模块已画好框）
        detections: [(x1, y1, x2, y2, label, conf), ...]（帧原始坐标）
        返回 (显示帧, 平移到裁剪坐标的 detections, 是否放大中)
        是否放大中 = 是否在预览框右下角显示原画面小窗
        """
        h, w = frame.shape[:2]
        now = time.monotonic()

        if not detections:
            if self._rect is None:
                return frame, detections, False  # 未跟踪：原比例、无小窗
            if now - self._last_seen > self.lost_hold:
                self.reset()  # 真正丢失 → 立即回到原比例
                return frame, detections, False
            # 短暂丢失：冻结窗口继续裁剪（防止一闪一闪）
            return self._crop(frame, self._rect, detections, self._pip_on)

        self._last_seen = now
        x1, y1, x2, y2 = self._pick_primary(detections)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

        # 期望窗口尺寸：保持画面宽高比、包住目标×边距、限制最大倍数
        aspect = w / h
        want_w = max(bw * self.margin, bh * self.margin * aspect,
                     w / self.max_zoom)
        want_w = min(want_w, float(w))
        want_h = want_w / aspect

        if self._size is None:
            # 首次识别：从全画面尺寸开始，形成平滑推近动画
            self._size = (float(w), float(h))
        sw, sh = self._size
        sw += (want_w - sw) * self.alpha
        sh += (want_h - sh) * self.alpha
        # 硬下限：窗口任何时刻都不小于目标框（带10%余量），
        # 目标突然变大（靠近镜头）时窗口立即扩大，放大过程中目标完整可见
        sw = max(sw, min(bw * 1.1, w))
        sh = max(sh, min(bh * 1.1, h))
        self._size = (sw, sh)

        # 窗口中心始终对准主目标 → 放大过程中目标保持在画面中间
        # （只平滑尺寸不平滑中心；贴画面边缘时窗口贴边）
        x = min(max(cx - sw / 2, 0.0), w - sw)
        y = min(max(cy - sh / 2, 0.0), h - sh)
        self._rect = (x, y, sw, sh)

        # 小窗显示状态带滞回：防止在阈值附近抖动导致小窗闪烁
        if sw < w * PIP_ON:
            self._pip_on = True
        elif sw > w * PIP_OFF:
            self._pip_on = False

        return self._crop(frame, self._rect, detections, self._pip_on)

    def _pick_primary(self, detections):
        """
        主目标选择（稳定防闪烁）：
        已跟踪过 → 取离上次主目标中心最近的框（多目标不会来回切换）；
        首次识别 → 取面积最大框（与锁定红框同一规则）
        """
        if self._primary_center is None:
            d = max(detections, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))
        else:
            px, py = self._primary_center
            d = min(detections, key=lambda t:
                    (((t[0] + t[2]) / 2 - px) ** 2
                     + ((t[1] + t[3]) / 2 - py) ** 2))
        self._primary_center = ((d[0] + d[2]) / 2, (d[1] + d[3]) / 2)
        return d[0], d[1], d[2], d[3]

    @staticmethod
    def _crop(frame, rect, detections, zoomed):
        """按窗口裁剪画面并平移检测坐标（窗口=全画面时裁剪结果与原帧一致）"""
        h, w = frame.shape[:2]
        x, y, cw, ch = rect
        iw = min(max(1, int(round(cw))), w)
        ih = min(max(1, int(round(ch))), h)
        ix = max(0, min(int(round(x)), w - iw))
        iy = max(0, min(int(round(y)), h - ih))
        crop = frame[iy:iy + ih, ix:ix + iw]
        shifted = [(d[0] - ix, d[1] - iy, d[2] - ix, d[3] - iy, d[4], d[5])
                   for d in detections]
        return crop, shifted, zoomed
