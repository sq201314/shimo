# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

"""
加载动画组件
LoadingPage：多终端并行执行的黑客风格初始化界面
  - 每个任务一个终端窗口面板，滚动日志+进度条
  - 矩阵代码雨背景
  - 所有终端执行完（关闭）= 初始化完成
"""
import math
import random
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QFont, QPainter, QColor, QPen

# 各任务的假日志台词
TASK_LOGS = {
    "default": [
        "alloc memory block ... ok",
        "read file header ... ok",
        "decode weights stream ...",
        "verify checksum ... pass",
        "build tensor graph ...",
        "fuse conv+bn layers ...",
        "upload to gpu:0 ...",
        "warmup inference ...",
        "optimize kernels ...",
        "done.",
    ],
}


class _TaskPanel:
    """单个终端面板的状态"""

    def __init__(self, task_id, title):
        self.id = task_id
        self.title = title
        self.done = False
        self.pct = 0
        self.logs = []
        self._logs_pool = list(TASK_LOGS["default"])
        random.shuffle(self._logs_pool)

    def tick(self):
        if self.done:
            return
        # 进度模拟推进（真实完成由 mark_done 切断）
        self.pct = min(95, self.pct + random.randint(1, 4))
        if random.random() < 0.4 and self._logs_pool:
            line = self._logs_pool.pop(0)
            self.logs.append(line)
            if len(self.logs) > 6:
                self.logs.pop(0)

    def mark_done(self):
        self.done = True
        self.pct = 100
        self.logs.append("ALL TASKS COMPLETE -> WINDOW CLOSED")
        if len(self.logs) > 6:
            self.logs.pop(0)


class LoadingPage(QWidget):
    """多终端并行初始化界面"""

    RAIN_CHARS = "01アイウエオカキクケコサシスセソ01ABCDEF"

    def __init__(self, tasks, text="正在初始化......", parent=None):
        """
        tasks: [(task_id, title), ...]
        """
        super().__init__(parent)
        self._text = text
        self._angle = 0
        self._panels = [_TaskPanel(tid, title) for tid, title in tasks]
        self._rain = []
        self._layout = {}   # panel.id -> (x, y, w, h, rot)
        self._layout_size = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(70)

    def _gen_layout(self):
        """随机散布终端位置：错落有致，但必须完整显示在窗口内"""
        w, h = self.width(), self.height()
        n = len(self._panels)
        if n == 0:
            self._layout = {}
            self._layout_size = (w, h)
            return

        # 可用区域（避开标题和底部状态栏），四周留安全边距
        margin = 16
        top = int(h * 0.15)
        bottom = int(h * 0.93)
        area_w = w - margin * 2
        area_h = bottom - top

        cols = max(1, round(n ** 0.5))
        rows = (n + cols - 1) // cols
        slot_w = area_w // cols
        slot_h = area_h // rows

        # 打乱面板和槽位的对应，视觉上更随机
        slots = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(slots)

        self._layout = {}
        for i, panel in enumerate(self._panels):
            r, c = slots[i % len(slots)]
            # 面板尺寸：小于槽位（给旋转留余量），槽内随机
            max_pw = int(slot_w * 0.95)
            max_ph = int(slot_h * 0.92)
            pw = random.randint(int(max_pw * 0.75), max_pw) if max_pw > 60 else max_pw
            ph = random.randint(int(max_ph * 0.75), max_ph) if max_ph > 60 else max_ph
            rot = random.uniform(-5, 5)

            # 槽内随机偏移
            slot_x = margin + c * slot_w
            slot_y = top + r * slot_h
            dx = random.randint(0, max(0, slot_w - pw))
            dy = random.randint(0, max(0, slot_h - ph))
            x, y = slot_x + dx, slot_y + dy

            # 计算旋转后的包围盒，确保整体在窗口内
            rad = abs(rot) * math.pi / 180
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            bw = (pw * cos_a + ph * sin_a) / 2
            bh = (pw * sin_a + ph * cos_a) / 2
            cx = x + pw / 2
            cy = y + ph / 2
            cx = min(max(cx, margin + bw), w - margin - bw)
            cy = min(max(cy, top + bh), bottom - bh)
            x = cx - pw / 2
            y = cy - ph / 2

            self._layout[panel.id] = (int(x), int(y), pw, ph, rot)

        self._layout_size = (w, h)

    def resizeEvent(self, event):
        self._layout_size = None  # 尺寸变化后重新生成布局
        super().resizeEvent(event)

    # ---- 外部接口 ----
    def mark_done(self, task_id):
        """某任务完成（主线程调用）"""
        for panel in self._panels:
            if panel.id == task_id:
                panel.mark_done()
                break
        self.update()

    def all_done(self):
        return all(p.done for p in self._panels)

    # ---- 更新 ----
    def _tick(self):
        self._angle = (self._angle + 4) % 360
        for panel in self._panels:
            panel.tick()
        self._update_rain()
        self.update()

    def _update_rain(self):
        w = self.width()
        col_w = 18
        n_cols = max(1, w // col_w)
        if not self._rain:
            self._rain = [[random.randint(-20, 0), random.uniform(0.4, 1.6)]
                          for _ in range(n_cols)]
        while len(self._rain) < n_cols:
            self._rain.append([random.randint(-20, 0), random.uniform(0.4, 1.6)])
        for col in self._rain:
            col[0] += col[1] * 2
            if col[0] > self.height() / col_w + 5:
                col[0] = random.randint(-30, -5)

    # ---- 绘制 ----
    def paintEvent(self, event):
        # 异常保护：PyQt5 槽函数异常会直接闪退
        try:
            self._paint(event)
        except Exception:
            import traceback
            traceback.print_exc()

    def _paint(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        p.fillRect(self.rect(), QColor(6, 9, 20))

        # 1. 矩阵雨
        p.setFont(QFont("Consolas", 11))
        col_w, line_h = 18, 16
        for i, (y, speed) in enumerate(self._rain):
            x = i * col_w
            yy = int(y * line_h)
            for j in range(8):
                cy = yy - j * line_h
                if -line_h < cy < h:
                    if j == 0:
                        p.setPen(QColor(170, 210, 255))
                    else:
                        p.setPen(QColor(40, 110, 220, max(10, 140 - j * 18)))
                    p.drawText(x, cy, self.RAIN_CHARS[(i * 7 + j + int(y)) % len(self.RAIN_CHARS)])

        # 2. 标题（白色）
        p.setPen(QColor(240, 240, 240))
        p.setFont(QFont("Consolas", 22, QFont.Bold))
        p.drawText(0, int(h * 0.09), w, 45, Qt.AlignCenter, "AI RECOGNITION SYSTEM // INIT")

        # 3. 终端面板随机散布（重叠、倾斜）
        if self._layout_size != (w, h):
            self._gen_layout()

        done_count = 0
        # 固定随机绘制顺序，让面板自然地互相压叠
        draw_order = list(range(len(self._panels)))
        random.Random(42).shuffle(draw_order)
        for idx in draw_order:
            panel = self._panels[idx]
            if panel.done:
                done_count += 1
            x, y, pw, ph, rot = self._layout[panel.id]
            p.save()
            p.translate(x + pw / 2, y + ph / 2)
            p.rotate(rot)
            self._draw_panel(p, panel, -pw / 2, -ph / 2, pw, ph)
            p.restore()

        # 4. 底部状态栏
        n = len(self._panels)
        p.setFont(QFont("Consolas", 13, QFont.Bold))
        if done_count == n:
            p.setPen(QColor(80, 220, 120))  # 全部完成 → 绿色
            p.drawText(0, h - int(h * 0.05), w, 30, Qt.AlignCenter,
                       "[ ALL TERMINALS CLOSED - INIT COMPLETE ]")
        else:
            p.setPen(QColor(240, 240, 240))  # 白色
            blinking = int(self._angle / 8) % 2 == 0
            cursor = "_" if blinking else " "
            p.drawText(0, h - int(h * 0.05), w, 30, Qt.AlignCenter,
                       f"{self._text} [{done_count}/{n} terminals closed]{cursor}")

    def _draw_panel(self, p, panel, x, y, w, h):
        x, y, w, h = int(x), int(y), int(w), int(h)
        # 面板背景
        p.setPen(QPen(QColor(30, 80, 180), 1))
        p.setBrush(QColor(10, 16, 36, 235))
        p.drawRoundedRect(x, y, w, h, 6, 6)

        # 标题栏
        title_h = 24
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(16, 40, 90, 235))
        p.drawRoundedRect(x, y, w, title_h, 6, 6)
        p.fillRect(x, y + title_h - 6, w, 6, QColor(16, 40, 90, 235))

        # 标题栏三个窗口按钮
        dot_colors = [QColor(255, 95, 86), QColor(255, 189, 46), QColor(39, 201, 63)]
        for i, dc in enumerate(dot_colors):
            p.setBrush(dc)
            p.setPen(Qt.NoPen)
            p.drawEllipse(x + 8 + i * 14, y + 7, 10, 10)

        # 任务标题 + 状态（白色，正常窗口标题）
        p.setFont(QFont("Consolas", 10, QFont.Bold))
        p.setPen(QColor(240, 240, 240))
        status = " [DONE]" if panel.done else " [RUNNING...]"
        p.drawText(x + 55, y + 17, panel.title + status)

        # 日志行（正常终端配色：普通白、成功绿、警告黄、错误红）
        p.setFont(QFont("Consolas", 9))
        log_top = y + title_h + 16
        line_step = max(14, (h - title_h - 46) // 7)
        for i, line in enumerate(panel.logs):
            ly = log_top + i * line_step
            if ly > y + h - 40:
                break
            p.setPen(self._log_color(line))
            p.drawText(x + 10, ly, "> " + line)

        # 进度条
        bar_y = y + h - 26
        bar_h = 12
        p.setPen(QPen(QColor(30, 80, 180), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(x + 10, bar_y, w - 20, bar_h, 4, 4)
        p.setPen(Qt.NoPen)
        color = QColor(80, 160, 255) if panel.done else QColor(60, 140, 235)
        p.setBrush(color)
        fill_w = (w - 22) * panel.pct / 100
        if fill_w > 2:
            p.drawRoundedRect(QRectF(x + 11, bar_y + 1, fill_w, bar_h - 2), 3, 3)
        # 百分比（白色）
        p.setFont(QFont("Consolas", 9, QFont.Bold))
        p.setPen(QColor(240, 240, 240))
        p.drawText(x + w - 50, bar_y + 10, f"{panel.pct}%")

    @staticmethod
    def _log_color(line):
        """按日志内容返回终端常规配色"""
        low = line.lower()
        if any(k in low for k in ("err", "fail", "traceback")):
            return QColor(255, 90, 90)      # 错误红
        if "warn" in low:
            return QColor(255, 200, 60)     # 警告黄
        if any(k in low for k in ("ok", "pass", "done", "complete", "closed")):
            return QColor(80, 220, 120)     # 成功绿
        return QColor(210, 210, 210)        # 普通浅灰白

    # ---- 控制 ----
    def start(self):
        if self.parent():
            self.setGeometry(0, 0, self.parent().width(), self.parent().height())
        self._timer.start(70)
        self.show()
        self.raise_()

    def stop(self):
        self._timer.stop()
        self.hide()