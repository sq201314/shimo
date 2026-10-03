# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class CaptureDialog(QWidget):
    """采集进度弹窗"""
    cancel = pyqtSignal()

    def __init__(self, total=4, parent=None):
        super().__init__(parent)
        self.total = total
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setFixedSize(400, 160)
        self.setStyleSheet("background: #2E3440; border-radius: 12px;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(14)

        self.title = QLabel("正在采集目标图像中...", self)
        self.title.setFont(QFont("微软雅黑", 14, QFont.Bold))
        self.title.setStyleSheet("color: #D8DEE9; background: transparent;")
        self.title.setAlignment(Qt.AlignCenter)

        self.count_label = QLabel(f"0 / {total}", self)
        self.count_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.count_label.setStyleSheet("color: #88C0D0; background: transparent;")
        self.count_label.setAlignment(Qt.AlignCenter)

        self.bar = QProgressBar(self)
        self.bar.setRange(0, total)
        self.bar.setValue(0)
        self.bar.setFixedHeight(16)
        self.bar.setStyleSheet("""
            QProgressBar { border: 2px solid #4C566A; border-radius: 8px; text-align: center;
                           color: white; font-size: 12px; background: #434C5E; }
            QProgressBar::chunk { background: #88C0D0; border-radius: 8px; }
        """)

        lay.addWidget(self.title)
        lay.addWidget(self.count_label)
        lay.addWidget(self.bar)

    def update_progress(self, current):
        self.count_label.setText(f"{current} / {self.total}")
        self.bar.setValue(current)
        if current >= self.total:
            self.title.setText("采集完成！")

    def center_on(self, parent):
        if parent:
            px = parent.x() + (parent.width() - self.width()) // 2
            py = parent.y() + (parent.height() - self.height()) // 2
            self.move(px, py)