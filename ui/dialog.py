# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt


class MsgDialog(QDialog):
    def __init__(self, title, msg, ask=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint | Qt.MSWindowsFixedSizeDialogHint)
        self.setFixedSize(400, 200)
        self.setStyleSheet("""
            QDialog { background: #2E3440; border-radius: 12px; }
            QLabel { color: #D8DEE9; font-size: 16px; }
            QPushButton {
                background: #81A1C1; border: none; border-radius: 6px;
                padding: 8px 20px; color: white; font-size: 14px; min-width: 80px;
            }
            QPushButton:hover { background: #88C0D0; }
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(20)
        lbl = QLabel(msg)
        lbl.setWordWrap(True)
        lay.addWidget(lbl, alignment=Qt.AlignCenter)
        bl = QHBoxLayout()
        bl.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.ok = QPushButton("确定")
        self.ok.clicked.connect(self.accept)
        bl.addWidget(self.ok)
        self._result = False
        if ask:
            self.cancel = QPushButton("取消")
            self.cancel.clicked.connect(self.reject)
            bl.addWidget(self.cancel)
        bl.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        lay.addLayout(bl)

    def accept(self):
        self._result = True
        super().accept()

    @staticmethod
    def ask(parent, title, msg):
        d = MsgDialog(title, msg, ask=True, parent=parent)
        d.exec_()
        return d._result

    @staticmethod
    def info(parent, title, msg):
        MsgDialog(title, msg, parent=parent).exec_()