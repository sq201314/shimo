# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtGui import QPainter, QColor


class ToggleButton(QWidget):
    toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 36)
        self._on = False
        self._circle_x = 4
        self._bg_color = "#555"
        self.setCursor(Qt.PointingHandCursor)

    def is_on(self):
        return self._on

    def mousePressEvent(self, event):
        self._on = not self._on
        target = 44 if self._on else 4
        anim = QPropertyAnimation(self, b"circle_x")
        anim.setDuration(200)
        anim.setStartValue(self._circle_x)
        anim.setEndValue(target)
        anim.setEasingCurve(QEasingCurve.InOutCubic)
        anim.start()
        self._anim = anim
        self._bg_color = "#00CC66" if self._on else "#555"
        self.update()
        self.toggled.emit(self._on)

    def _get_x(self):
        return self._circle_x

    def _set_x(self, v):
        self._circle_x = v
        self.update()

    circle_x = pyqtProperty(float, fget=_get_x, fset=_set_x)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(self._bg_color))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, 80, 36, 18, 18)
        p.setBrush(QColor("#fff"))
        p.drawEllipse(int(self._circle_x), 4, 28, 28)
        p.end()