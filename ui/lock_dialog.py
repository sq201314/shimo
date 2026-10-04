# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

"""
锁定确认弹窗：语音播报文本 + 已框选目标截图 + Yes/No
Yes → yes_clicked 信号（主程序标红框）；No / 关闭 → 仅消失
"""
import cv2
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage

# 弹窗显示与语音播报共用的文本（tts.ALERT_TEXT 与之保持一致）
ALERT_TEXT = "已经发现并锁定目标是否进行打击。"

# 弹窗尺寸：小而宽（宽扁形）
DLG_W, DLG_H = 660, 430
IMG_MAX_W, IMG_MAX_H = 600, 250


class LockDialog(QDialog):
    yes_clicked = pyqtSignal()   # 点击 Yes（主程序据此标红框）
    answered = pyqtSignal()      # 任意方式关闭（Yes/No/标题栏X），用于停语音、清引用

    def __init__(self, text, img_bgr, parent=None):
        super().__init__(parent)
        self._done = False
        self._answered = False
        # finished 覆盖所有关闭路径（Yes/No/标题栏X/Esc），answered 只发一次
        self.finished.connect(self._on_finished)
        self.setWindowTitle("目标锁定确认")
        self.setWindowFlags(
            Qt.Dialog | Qt.WindowStaysOnTopHint | Qt.MSWindowsFixedSizeDialogHint
        )
        self.setFixedSize(DLG_W, DLG_H)
        self.setStyleSheet("""
            QDialog { background: #2E3440; border-radius: 12px; }
            QLabel#text { color: #ECEFF4; font-size: 20px; font-weight: bold; }
            QLabel#hint { color: #88C0D0; font-size: 13px; }
            QLabel#img { background: #1B1E26; border: 2px solid #4C566A;
                         border-radius: 6px; }
            QPushButton#yes {
                background: #A3BE8C; color: #1B1E26; border: none;
                border-radius: 6px; padding: 10px 40px;
                font-size: 16px; font-weight: bold;
            }
            QPushButton#yes:hover { background: #B5CC9C; }
            QPushButton#no {
                background: #BF616A; color: #ECEFF4; border: none;
                border-radius: 6px; padding: 10px 40px;
                font-size: 16px; font-weight: bold;
            }
            QPushButton#no:hover { background: #D08770; }
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 22, 28, 22)
        lay.setSpacing(12)

        lbl = QLabel(text, self)
        lbl.setObjectName("text")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        hint = QLabel("已框选目标", self)
        hint.setObjectName("hint")
        hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(hint)

        self.img_lbl = QLabel(self)
        self.img_lbl.setObjectName("img")
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setFixedSize(IMG_MAX_W, IMG_MAX_H)
        self.img_lbl.setPixmap(self._to_pixmap(img_bgr))
        lay.addWidget(self.img_lbl, alignment=Qt.AlignCenter)

        bl = QHBoxLayout()
        bl.setSpacing(40)
        bl.addStretch()
        self.btn_yes = QPushButton("Yes", self)
        self.btn_yes.setObjectName("yes")
        self.btn_yes.setCursor(Qt.PointingHandCursor)
        self.btn_yes.clicked.connect(self._on_yes)
        bl.addWidget(self.btn_yes)
        self.btn_no = QPushButton("No", self)
        self.btn_no.setObjectName("no")
        self.btn_no.setCursor(Qt.PointingHandCursor)
        self.btn_no.clicked.connect(self._on_no)
        bl.addWidget(self.btn_no)
        bl.addStretch()
        lay.addLayout(bl)

    @staticmethod
    def _to_pixmap(img_bgr):
        """BGR截图 → QPixmap（等比缩放到图片区）"""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pm = QPixmap.fromImage(qimg)
        return pm.scaled(IMG_MAX_W, IMG_MAX_H,
                         Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _on_finished(self, _result):
        if not self._answered:
            self._answered = True
            self.answered.emit()

    def _on_yes(self):
        if self._done:
            return
        self._done = True
        self.yes_clicked.emit()
        self.accept()

    def _on_no(self):
        if self._done:
            return
        self._done = True
        self.accept()
