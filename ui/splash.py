# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

import os
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QGraphicsOpacityEffect
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtGui import QFont, QPainter, QColor


class AnimLetter(QLabel):
    def __init__(self, text="", color="#ffffff", parent=None):
        super().__init__(text, parent)
        self._angle = 0.0
        self._sc = 0.1
        self._color = color

    def _setA(self, v): self._angle = v; self.update()
    def _getA(self): return self._angle
    def _setS(self, v): self._sc = v; self.update()
    def _getS(self): return self._sc

    angle = pyqtProperty(float, fget=_getA, fset=_setA)
    sc = pyqtProperty(float, fget=_getS, fset=_setS)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.translate(self.width() // 2, self.height() // 2)
        p.rotate(self._angle)
        p.scale(self._sc, self._sc)
        p.translate(-self.width() // 2, -self.height() // 2)
        p.setFont(self.font())
        rect = self.rect()
        base = QColor(self._color)
        r, g, b = base.red(), base.green(), base.blue()
        depth = 12
        for i in range(depth, 0, -1):
            t = i / depth
            dr = int(r * (0.15 + 0.85 * (1 - t)))
            dg = int(g * (0.15 + 0.85 * (1 - t)))
            db = int(b * (0.15 + 0.85 * (1 - t)))
            p.setPen(QColor(dr, dg, db))
            p.drawText(rect.adjusted(0, i, 0, i), Qt.AlignCenter, self.text())
        p.setPen(QColor(min(255, r + 40), min(255, g + 40), min(255, b + 40)))
        p.drawText(rect.adjusted(0, -1, 0, -1), Qt.AlignCenter, self.text())
        p.setPen(QColor(255, 255, 255, 90))
        p.drawText(rect.adjusted(0, -2, 0, -2), Qt.AlignCenter, self.text())
        p.end()


class SplashScreen(QWidget):
    LETTERS = list("Quan")
    COLORS = ["#FF3366", "#FF9933", "#33CC33", "#3399FF"]

    def __init__(self, on_done=None):
        super().__init__()
        self.on_done = on_done
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background: #000000;")
        scr = QApplication.primaryScreen().geometry()
        self.setGeometry(scr)
        w, h = scr.width(), scr.height()

        fs = max(70, int(h * 0.16))
        letter_w = max(120, int(w * 0.15))
        letter_h = int(fs * 2.0)
        total_w = letter_w * len(self.LETTERS)
        start_x = (w - total_w) // 2
        y = int(h * 0.30)

        self.letters = []
        for i, ch in enumerate(self.LETTERS):
            lbl = AnimLetter(ch, self.COLORS[i], self)
            lbl.setFont(QFont("Segoe Print", fs, QFont.Bold))
            lbl.setStyleSheet("background: transparent;")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setGeometry(start_x + i * letter_w, y, letter_w, letter_h)
            lbl.setGraphicsEffect(self._eff(0))
            self.letters.append(lbl)

        self.sub = QLabel("AI智能识别系统", self)
        sfs = max(16, int(h * 0.028))
        sub_y = y + letter_h + int(h * 0.05)
        self.sub.setFont(QFont("微软雅黑", sfs))
        self.sub.setStyleSheet("color: #888; background: transparent;")
        self.sub.setAlignment(Qt.AlignCenter)
        self.sub.setGeometry(0, sub_y, w, int(sfs * 2))
        self.sub.setGraphicsEffect(self._eff(0))

        self.show()
        self._step = -1
        self._t = QTimer(self)
        self._t.timeout.connect(self._next)
        self._t.start(300)

    def _eff(self, v):
        e = QGraphicsOpacityEffect(self)
        e.setOpacity(v)
        return e

    def _fade(self, w, dur=600):
        e = w.graphicsEffect()
        if not e:
            e = QGraphicsOpacityEffect(w)
            w.setGraphicsEffect(e)
        a = QPropertyAnimation(e, b"opacity", self)
        a.setDuration(dur)
        a.setStartValue(e.opacity())
        a.setEndValue(1.0)
        a.start()
        if not hasattr(self, '_a'): self._a = []
        self._a.append(a)

    def _anim(self, obj, prop, start, end, dur):
        a = QPropertyAnimation(obj, prop, self)
        a.setDuration(dur)
        a.setStartValue(start)
        a.setEndValue(end)
        a.start()
        if not hasattr(self, '_a'): self._a = []
        self._a.append(a)

    def _next(self):
        self._step += 1
        n = len(self.LETTERS)
        if 0 <= self._step < n:
            lbl = self.letters[self._step]
            self._anim(lbl, b"angle", -90, 0, 800)
            self._anim(lbl, b"sc", 0.1, 1.0, 800)
            self._fade(lbl, 800)
        elif self._step == n + 1:
            self._fade(self.sub, 600)
        elif self._step == n + 5:
            self._t.stop()
            for lbl in self.letters: lbl.hide()
            self.sub.hide()
            e = QGraphicsOpacityEffect(self)
            self.setGraphicsEffect(e)
            a = QPropertyAnimation(e, b"opacity", self)
            a.setDuration(500)
            a.setStartValue(1.0)
            a.setEndValue(0.0)
            a.finished.connect(self._done)
            a.start()
            self._fa = a

    def _done(self):
        self.close()
        if self.on_done: self.on_done()