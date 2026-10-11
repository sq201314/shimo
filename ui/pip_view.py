# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

"""
放大跟踪右下角原画面小窗标签
可点击：点击后主画面与小窗内容互换（再点换回），发出 clicked 信号
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QLabel


class PipLabel(QLabel):
    """右下角小窗（PipLabel）：左键点击发出 clicked 信号"""

    clicked = pyqtSignal()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            e.accept()
            self.clicked.emit()
        # 不调用 QLabel 默认实现（默认会忽略事件冒泡到父窗口）
