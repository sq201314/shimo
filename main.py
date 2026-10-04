# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

import sys
import os
import re
import cv2
import threading
import time
import subprocess

# 离线模式：跳过ultralytics导入时的DNS在线检测（可能卡10~20秒）
os.environ.setdefault("YOLO_OFFLINE", "1")

# 预加载torch和ultralytics，避免点击训练时才导入（省8~10秒）
import torch  # noqa: F401
import ultralytics  # noqa: F401

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QApplication,
    QLabel, QProgressBar, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QSize
from PyQt5.QtGui import QFont, QMovie, QPixmap, QImage

from ui.dialog import MsgDialog
from ui.splash import SplashScreen
from ui.progress import CaptureDialog
from ui.toggle import ToggleButton
from ui.loading import LoadingPage
from ui.lock_dialog import LockDialog, ALERT_TEXT

camera = 0  # 摄像头索引，0为默认摄像头

# 添加 core 目录到模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GIF_PATH = os.path.join(BASE_DIR, "8.gif")
CAPTURE_COUNT = 4


class Sig(QObject):
    img = pyqtSignal(QImage)
    prog = pyqtSignal(int)
    cap_done = pyqtSignal()
    train_done = pyqtSignal()
    log = pyqtSignal(str)
    det_stopped = pyqtSignal(str, int)  # 检测停止信号（参数：mode, 会话序号）
    models_ready = pyqtSignal()    # 模型加载完成信号
    task_done = pyqtSignal(str)    # 单个加载任务完成（参数：task_id）
    lock_trigger = pyqtSignal()    # Ctrl+鼠标左键 触发锁定确认（钩子线程 → 主线程）


class _TrainLogWriter:
    """捕获训练日志：按行发射到UI，剥离ANSI控制符，\r视为进度条刷新"""
    _ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07")

    def __init__(self, emit):
        self._emit = emit
        self._buf = ""

    def write(self, s):
        if not s:
            return
        if not isinstance(s, str):
            s = str(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit_line(line)
        # 缓冲中未换行的 \r 内容是进度条刷新，丢弃
        if "\r" in self._buf:
            self._buf = self._buf.rsplit("\r", 1)[-1]

    def _emit_line(self, line):
        line = line.rstrip("\r")
        if "\r" in line:
            line = line.split("\r")[-1]
        line = self._ANSI.sub("", line)  # 剥离ANSI控制符
        line = line.strip()
        if line:
            try:
                self._emit(line)
            except Exception:
                pass

    def flush(self):
        if self._buf.strip():
            self._emit_line(self._buf)
        self._buf = ""

    def isatty(self):
        return False


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI智能识别系统")
        self.setStyleSheet("background: #000;")
        self._current_frame = None
        self.resize(1920, 1080)
        self._init_ui()
        self.showFullScreen()

    def _init_ui(self):
        w, h = self.width(), self.height()
        self.s = Sig()

        # GIF背景
        self.bg = QLabel(self)
        self.bg.setGeometry(0, 0, w, h)
        self.movie = QMovie(GIF_PATH)
        self.bg.setMovie(self.movie)
        self.movie.setScaledSize(QSize(w, h))
        self.movie.start()
        self.bg.lower()

        # 标题
        self.title = QLabel("AI智能识别系统", self)
        ts = max(20, int(h * 0.06))
        self.title.setFont(QFont("黑体", ts, QFont.Bold))
        self.title.setStyleSheet("color: #eee; background: transparent;")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setGeometry(0, int(h * 0.03), w, int(ts * 2))

        # 按钮行
        self.btn_row = QWidget(self)
        self.btn_row.setStyleSheet("background: transparent;")
        bw = max(200, int(w * 0.22))
        bh = max(40, int(h * 0.07))
        row_w = bw + 100
        self.btn_row.setGeometry((w - row_w) // 2, int(h * 0.16), row_w, bh)
        row_lay = QHBoxLayout(self.btn_row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(15)

        self.btn = QPushButton("接入无人机画面", self.btn_row)
        self.btn.setCursor(Qt.PointingHandCursor)
        bfs = max(12, int(bh * 0.35))
        self.btn.setFont(QFont("微软雅黑", bfs))
        self.btn.setFixedSize(bw, bh)
        self.btn.setStyleSheet("""
            QPushButton { background: #CCFF00; color: #222; border: 2px solid #AACC00; border-radius: 8px; }
            QPushButton:hover { background: #EEFF66; }
            QPushButton:pressed { background: #99CC00; }
            QPushButton:disabled { background: #888; color: #555; border-color: #666; }
        """)
        self.btn.clicked.connect(self._on_btn)

        # 滑动开关（功能选择后显示）
        self.toggle = ToggleButton(self.btn_row)
        self.toggle.toggled.connect(self._on_toggle)
        self.toggle.hide()

        row_lay.addWidget(self.btn)
        row_lay.addWidget(self.toggle)

        # 识别功能按钮行（初始隐藏）
        self.det_btn_row = QWidget(self)
        self.det_btn_row.setStyleSheet("background: transparent;")
        self.det_btn_row.setGeometry(0, int(h * 0.16) + 60, w, bh)
        det_lay = QHBoxLayout(self.det_btn_row)
        det_lay.setContentsMargins(0, 0, 0, 0)
        det_lay.setSpacing(75)  # 约5个字宽

        det_btn_style = """
            QPushButton { background: transparent; color: #D8DEE9; border: none; }
            QPushButton:hover { color: #88C0D0; }
            QPushButton:pressed { color: #5E81AC; }
        """
        det_btn_active = """
            QPushButton { background: transparent; color: #88C0D0; border: none; font-weight: bold; }
        """

        det_font = QFont("微软雅黑", 15)
        self.btn_obj = QPushButton("目标检测", self.det_btn_row)
        self.btn_obj.setCursor(Qt.PointingHandCursor)
        self.btn_obj.setFont(det_font)
        self.btn_obj.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.btn_obj.setStyleSheet(det_btn_style)
        self.btn_obj.clicked.connect(self._on_obj_toggle)

        self.btn_person = QPushButton("人体检测", self.det_btn_row)
        self.btn_person.setCursor(Qt.PointingHandCursor)
        self.btn_person.setFont(det_font)
        self.btn_person.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.btn_person.setStyleSheet(det_btn_style)
        self.btn_person.clicked.connect(self._on_person_toggle)

        self.btn_pose = QPushButton("姿态骨骼", self.det_btn_row)
        self.btn_pose.setCursor(Qt.PointingHandCursor)
        self.btn_pose.setFont(det_font)
        self.btn_pose.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.btn_pose.setStyleSheet(det_btn_style)
        self.btn_pose.clicked.connect(self._on_pose_toggle)

        self.btn_back = QPushButton("返回", self.det_btn_row)
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.setFont(det_font)
        self.btn_back.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.btn_back.setStyleSheet("""
            QPushButton { background: transparent; color: #D8DEE9; border: none; }
            QPushButton:hover { color: #BF616A; }
            QPushButton:pressed { color: #D08770; }
        """)
        self.btn_back.clicked.connect(self._on_back)

        det_lay.addWidget(self.btn_obj)
        det_lay.addWidget(self.btn_person)
        det_lay.addWidget(self.btn_pose)
        det_lay.addWidget(self.btn_back)
        self.det_btn_row.hide()

        self._det_btn_style = det_btn_style
        self._det_btn_active = det_btn_active
        self._active_det = None  # 当前激活的检测功能
        self._models_loaded = False  # 模型是否加载完成

        # 画面区
        vw = max(320, int(w * 0.65))
        vy = int(h * 0.28)
        vh = max(200, h - vy - int(h * 0.08))
        self.view = QLabel(self)
        self.view.setFixedSize(vw, vh)
        self.view.move((w - vw) // 2, vy)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet("border: 3px solid #555; background: transparent;")

        # 进度条
        pw = max(300, int(w * 0.4))
        self.pbar = QProgressBar(self)
        self.pbar.setRange(0, 100)
        self.pbar.setValue(0)
        self.pbar.setTextVisible(True)
        self.pbar.setFixedWidth(pw)
        self.pbar.move((w - pw) // 2, int(h * 0.90))
        self.pbar.hide()

        # 信号
        self.s.img.connect(self._show)
        self.s.prog.connect(self._on_prog)
        self.s.cap_done.connect(self._on_cap_done)
        self.s.train_done.connect(self._on_train_done)
        self.s.log.connect(self._on_log)
        self.s.det_stopped.connect(self._on_det_stopped)
        self.s.models_ready.connect(self._on_models_ready)
        self.s.task_done.connect(self._on_task_done)

        # 状态
        self.cap = None
        self.running = False
        self.step = 0
        self.toggle_mode = False
        self.train_sub = "label"
        self.annotated_count = 0  # 已标注图片数量，用于区分新旧图片
        self.pose_running = False  # 姿态检测是否运行中
        # 多线程帧缓冲（生产者-消费者）
        self._frame = None
        self._frame_lock = threading.Lock()
        self._capture_running = False
        self._det_seq = 0  # 检测会话序号：每次切换+1，旧线程检测到变化立即自行退出（不阻塞UI）
        self._switch_lock = threading.Lock()  # 切换锁，防止频繁切换冲突

        # ---- 锁定确认（Ctrl+鼠标左键 语音播报+弹窗，姿态骨骼除外） ----
        self._last_det = None      # 最新检测结果 (frame, detections)，检测线程写
        self._det_lock = threading.Lock()  # 保护 _last_det
        self._lock_on = False      # Yes锁定后：画面标注红框
        self._lock_dlg = None      # 当前锁定确认弹窗（打开期间忽略新触发）
        self.s.lock_trigger.connect(self._on_lock_trigger)
        import hotkey
        hotkey.start(lambda: self.s.lock_trigger.emit())

    def _cleanup(self):
        """退出时清理所有资源"""
        self.running = False
        # 关闭摄像头
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            self.cap = None
        # 清理torch模型和显存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 清除所有模型引用
            import gc
            gc.collect()
        except:
            pass
        # 停止全局热键钩子与语音播报
        try:
            import hotkey
            hotkey.stop()
            import tts
            tts.stop()
        except Exception:
            pass
        # 关闭弹窗和子控件
        for attr in ['cap_dialog', 'log_area', 'glass_bar', '_lock_dlg']:
            if hasattr(self, attr):
                try:
                    obj = getattr(self, attr)
                    if obj:
                        obj.close()
                        obj.deleteLater()
                except:
                    pass
        # 删除临时文件
        for f in ['train_progress.txt']:
            try:
                fp = os.path.join(BASE_DIR, f)
                if os.path.exists(fp):
                    os.remove(fp)
            except:
                pass
        # 强制退出所有进程和线程
        os._exit(0)

    def _layout(self):
        w, h = self.width(), self.height()
        self.bg.setGeometry(0, 0, w, h)
        self.movie.setScaledSize(QSize(w, h))
        ts = max(20, int(h * 0.06))
        self.title.setFont(QFont("黑体", ts, QFont.Bold))
        self.title.setGeometry(0, int(h * 0.03), w, int(ts * 2))
        bw = max(200, int(w * 0.22))
        bh = max(40, int(h * 0.07))
        bfs = max(12, int(bh * 0.35))
        row_w = bw + 100
        self.btn_row.setGeometry((w - row_w) // 2, int(h * 0.16), row_w, bh)
        self.btn.setFixedSize(bw, bh)
        self.btn.setFont(QFont("微软雅黑", bfs))
        # 识别按钮行（自动宽度）
        self.det_btn_row.adjustSize()
        row_rect = self.det_btn_row.geometry()
        self.det_btn_row.move((w - row_rect.width()) // 2, int(h * 0.16) + 60)
        vw = max(320, int(w * 0.65))
        vy = int(h * 0.28)
        vh = max(200, h - vy - int(h * 0.08))
        self.view.setFixedSize(vw, vh)
        self.view.move((w - vw) // 2, vy)
        pw = max(300, int(w * 0.4))
        self.pbar.setFixedWidth(pw)
        self.pbar.move((w - pw) // 2, int(h * 0.90))

    def resizeEvent(self, e):
        if hasattr(self, 'bg'): self._layout()
        super().resizeEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self._cleanup()
            QApplication.quit()
        elif e.key() == Qt.Key_1 and e.modifiers() & Qt.ControlModifier:
            if self.isFullScreen():
                sw = QApplication.primaryScreen().geometry()
                w, h = int(sw.width() * 0.8), int(sw.height() * 0.8)
                self.showNormal()
                self.resize(w, h)
                self.move((sw.width() - w) // 2, (sw.height() - h) // 2)
            else:
                self.showFullScreen()

    # ---- 按钮 ----
    def _on_btn(self):
        if self.step == 0:
            self._do_connect()
        elif self.step == 1:
            self._do_func_select()
        elif self.step == 2:
            if not self.toggle_mode:
                self._do_capture()
            else:
                if self.train_sub == "label":
                    self._do_label()
                else:
                    self._do_train()

    def _on_toggle(self, on):
        self.toggle_mode = on
        if on:
            # 切到模型训练：重置为标注状态
            self.train_sub = "label"
            self.btn.setText("开始进行标注")
        else:
            self.btn.setText("采集图像")

    def _do_func_select(self):
        """功能选择弹窗"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton as QPB
        dlg = QDialog(self)
        dlg.setWindowTitle("功能选择")
        dlg.setFixedSize(350, 180)
        dlg.setStyleSheet("""
            QDialog { background: #2E3440; border-radius: 12px; }
            QLabel { color: #D8DEE9; font-size: 16px; }
            QPushButton {
                background: #81A1C1; border: none; border-radius: 6px;
                padding: 10px 24px; color: white; font-size: 15px; min-width: 100px;
            }
            QPushButton:hover { background: #88C0D0; }
        """)
        from PyQt5.QtWidgets import QLabel as QL
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(16)
        lay.addWidget(QL("请选择功能：", dlg), alignment=Qt.AlignCenter)
        bl = QHBoxLayout()
        bl.setSpacing(20)
        btn_train = QPB("模型训练", dlg)
        btn_det = QPB("智能识别", dlg)
        bl.addWidget(btn_train)
        bl.addWidget(btn_det)
        lay.addLayout(bl)
        result = {"choice": None}
        btn_train.clicked.connect(lambda: (result.update(choice="train"), dlg.accept()))
        btn_det.clicked.connect(lambda: (result.update(choice="detect"), dlg.accept()))
        dlg.exec_()
        if result["choice"] == "train":
            # 选模型训练 → 显示滑动开关，进入采集流程
            self.btn.setText("采集图像")
            self.toggle.show()
            self.toggle_mode = False
            self.train_sub = "label"
            self.step = 2
        elif result["choice"] == "detect":
            # 进入识别模式：显示多终端加载界面，后台并行加载模型
            # 注意：不在主线程 import ultralytics（很慢会卡界面），直接扫目录
            self.step = 3
            self._models_loaded = False
            tasks = []
            model_dir = os.path.join(BASE_DIR, "models")
            if os.path.exists(model_dir):
                for fname in sorted(os.listdir(model_dir)):
                    if fname.endswith(".pt") and not fname.startswith("yolo"):
                        name = fname[:-3]
                        tasks.append((name, f"model/{fname}"))
            tasks += [
                ("yolo11n", "model/yolo11n.pt [person]"),
                ("pose", "mediapipe/pose_landmarker"),
            ]
            self.loading_page = LoadingPage(tasks, "正在初始化......", self)
            self.loading_page.start()
            threading.Thread(target=self._preload_models, daemon=True).start()

    # ---- 接入摄像头 ----
    def _do_connect(self):
        ok = MsgDialog.ask(self, "确认", "是否接入摄像头画面？")
        if not ok: return
        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            MsgDialog.info(self, "错误", "无法打开摄像头")
            return
        MsgDialog.info(self, "成功", "摄像头已接入")
        self.running = True
        self._capture_running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._video_loop, daemon=True).start()
        self.btn.setText("功能选择")
        self.step = 1

    def _capture_loop(self):
        """采集线程：持续读取摄像头帧到缓冲区（生产者）"""
        while self._capture_running and self.cap and self.cap.isOpened():
            try:
                ret, f = self.cap.read()
                if not ret:
                    break
                self._current_frame = f.copy()
                with self._frame_lock:
                    self._frame = f.copy()
                time.sleep(0.01)
            except Exception as e:
                print(f"采集线程错误: {e}")
                break

    def _video_loop(self):
        """显示线程：从缓冲区取帧显示（消费者）"""
        try:
            while self.running and self._capture_running:
                with self._frame_lock:
                    f = self._frame.copy() if self._frame is not None else None
                if f is not None:
                    qimg = self._qimg(f)
                    if qimg:
                        self.s.img.emit(qimg)
                time.sleep(0.03)
        except Exception as e:
            print(f"显示线程错误: {e}")

    # ---- 采集图像 ----
    def _do_capture(self):
        if not self.cap or not self.cap.isOpened():
            MsgDialog.info(self, "错误", "请先接入摄像头")
            return
        self.out = os.path.join(BASE_DIR, "data", "images")
        os.makedirs(self.out, exist_ok=True)
        self.btn.setEnabled(False)
        # 弹出采集进度窗口
        self.cap_dialog = CaptureDialog(CAPTURE_COUNT, self)
        self.cap_dialog.center_on(self)
        self.cap_dialog.show()
        threading.Thread(target=self._cap_thread, daemon=True).start()

    def _cap_thread(self):
        # 累加：从已有图片数量开始编号
        existing = len([f for f in os.listdir(self.out) if f.endswith('.jpg')]) if os.path.exists(self.out) else 0
        cnt, last = 0, 0
        while self.running and cnt < CAPTURE_COUNT:
            now = time.time()
            if now - last < 0.5:
                time.sleep(0.01)
                continue
            last = now
            if self._current_frame is not None:
                filename = os.path.join(self.out, f"{existing + cnt}.jpg")
                cv2.imwrite(filename, self._current_frame)
                cnt += 1
                self.s.prog.emit(cnt)
        self.s.cap_done.emit()

    def _on_log(self, text):
        if hasattr(self, 'log_area') and self.log_area.isVisible():
            self.log_area.append(text)
            # 自动滚动到底部
            sb = self.log_area.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _on_prog(self, val):
        if hasattr(self, 'cap_dialog') and self.cap_dialog:
            self.cap_dialog.update_progress(val)
        # 更新玻璃进度条
        if hasattr(self, 'glass_bar') and self.glass_bar.isVisible():
            self.glass_bar.setValue(val)
        # 更新原进度条
        if self.pbar.isVisible():
            self.pbar.setValue(val)

    def _on_cap_done(self):
        if hasattr(self, 'cap_dialog') and self.cap_dialog:
            self.cap_dialog.update_progress(CAPTURE_COUNT)
        self.btn.setEnabled(True)
        self.title.setText("AI智能识别系统")
        MsgDialog.info(self, "完成", f"采集{CAPTURE_COUNT}张图像完成")
        if hasattr(self, 'cap_dialog') and self.cap_dialog:
            self.cap_dialog.close()
        # 回到采集模式，新采集后重置标注状态
        self.btn.setText("采集图像")
        self.toggle_mode = False
        self.train_sub = "label"  # 重置：下次切到训练时需重新标注
        self.toggle.show()
        self.step = 2

    # ---- 标注（只显示未标注的新图片） ----
    def _do_label(self):
        self.title.setText("正在进行模型标注...")
        self.title.setStyleSheet("color: #eee; background: transparent;")
        self.btn.setEnabled(False)
        # 关闭摄像头，显示标注提示
        self.running = False
        self.view.setText("正在进行模型标注")
        self.view.setStyleSheet("border: 3px solid #555; background: transparent; color: #888; font-size: 36px; font-weight: bold;")
        img_dir = os.path.join(BASE_DIR, "data", "images")
        cls_file = os.path.join(BASE_DIR, "data", "classes.txt")

        # 取出未标注的新图片到临时目录
        new_dir = os.path.join(BASE_DIR, "data", "new_images")
        os.makedirs(new_dir, exist_ok=True)
        # 只删除图片和标签，保留classes.txt
        for f in os.listdir(new_dir):
            if f.endswith('.jpg') or f.endswith('.txt'):
                os.remove(os.path.join(new_dir, f))

        # 复制新图片（index >= annotated_count）
        all_imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')],
                          key=lambda x: int(os.path.splitext(x)[0]))
        new_imgs = all_imgs[self.annotated_count:]
        for f in new_imgs:
            import shutil
            shutil.copy2(os.path.join(img_dir, f), os.path.join(new_dir, f))

        if not new_imgs:
            MsgDialog.info(self, "提示", "没有新图片需要标注")
            self.btn.setEnabled(True)
            return

        try:
            labelimg_exe = os.path.join(BASE_DIR, "venv", "Scripts", "labelImg.exe")
            subprocess.run([labelimg_exe, new_dir, cls_file, new_dir], cwd=BASE_DIR)
        except Exception as e:
            print(f"LabelImg error: {e}")

        # 标注完成后，把图片、标签、classes.txt移到labeled_images目录
        import shutil
        labeled_dir = os.path.join(BASE_DIR, "data", "labeled_images")
        os.makedirs(labeled_dir, exist_ok=True)

        for f in new_imgs:
            # 复制图片
            src_img = os.path.join(new_dir, f)
            dst_img = os.path.join(labeled_dir, f)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            # 复制标签
            txt = os.path.splitext(f)[0] + ".txt"
            src_txt = os.path.join(new_dir, txt)
            dst_txt = os.path.join(labeled_dir, txt)
            if os.path.exists(src_txt):
                shutil.copy2(src_txt, dst_txt)

        # 合并classes.txt（去重，不覆盖）
        new_cls = os.path.join(new_dir, "classes.txt")
        if os.path.exists(new_cls):
            labeled_cls = os.path.join(labeled_dir, "classes.txt")
            main_cls = os.path.join(BASE_DIR, "data", "classes.txt")
            # 读取已有的类别
            existing = set()
            if os.path.exists(labeled_cls):
                with open(labeled_cls, "r", encoding="utf-8", errors="replace") as f:
                    existing = {line.strip() for line in f if line.strip()}
            # 读取新标注的类别
            with open(new_cls, "r", encoding="utf-8", errors="replace") as f:
                new_classes = {line.strip() for line in f if line.strip()}
            # 合并
            merged = existing | new_classes
            with open(labeled_cls, "w", encoding="utf-8") as f:
                for cls in sorted(merged):
                    f.write(cls + "\n")
            # 同步到主目录
            shutil.copy2(labeled_cls, main_cls)

        # 清空临时目录
        for f in os.listdir(new_dir):
            os.remove(os.path.join(new_dir, f))

        self.annotated_count = len(all_imgs)
        self.title.setText("AI智能识别系统")
        self.btn.setEnabled(True)
        self.train_sub = "train"
        self.btn.setText("开始进行模型训练")
        # 恢复摄像头画面
        self.view.clear()
        self.view.setStyleSheet("border: 3px solid #555; background: transparent;")
        self.running = True
        threading.Thread(target=self._video_loop, daemon=True).start()
        MsgDialog.info(self, "完成", "标注完成")

    # ---- 模型训练 ----
    def _do_train(self):
        self.title.setText("正在进行模型训练")
        self.title.setStyleSheet("color: #eee; background: transparent;")
        self.btn.setEnabled(False)
        self.running = False
        self.view.hide()
        # 日志区
        from PyQt5.QtWidgets import QTextEdit
        if not hasattr(self, 'log_area'):
            self.log_area = QTextEdit(self)
            self.log_area.setReadOnly(True)
            self.log_area.setStyleSheet("background: #111; color: #0f0; font-family: Consolas; border: 3px solid #555;")
            self.log_area.setFont(QFont("Consolas", 10))
        self.log_area.clear()
        self.log_area.show()
        # 拟态玻璃进度条（日志上方）
        self.glass_bar = QProgressBar(self)
        self.glass_bar.setRange(0, 100)
        self.glass_bar.setValue(0)
        self.glass_bar.setTextVisible(True)
        self.glass_bar.setFixedHeight(28)
        self.glass_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(255, 255, 255, 0.15);
                border: 1px solid rgba(255, 255, 255, 0.25);
                border-radius: 14px;
                text-align: center;
                color: #fff;
                font-size: 13px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(100, 180, 255, 0.7),
                    stop:1 rgba(180, 100, 255, 0.7));
                border-radius: 14px;
            }
        """)
        # 布局：进度条在预览框下方（不重叠）
        geo = self.view.geometry()
        self.log_area.setGeometry(geo.x(), geo.y(), geo.width(), geo.height())
        self.glass_bar.setGeometry(geo.x(), geo.y() + geo.height() + 10, geo.width(), 28)
        self.glass_bar.show()
        self.pbar.hide()
        self.view.hide()
        # 调整日志区域位置和大小与view一致
        self.log_area.setGeometry(self.view.geometry())
        self.pbar.setValue(0)
        self.pbar.show()
        threading.Thread(target=self._train_thread, daemon=True).start()

    def _train_thread(self):
        progress_file = os.path.join(BASE_DIR, "train_progress.txt")
        with open(progress_file, "w") as f:
            f.write("0")
        self._watch_stop = threading.Event()
        threading.Thread(target=self._watch_progress, args=(progress_file,), daemon=True).start()

        # 进程内训练：复用已加载的 torch，省去子进程 8~10 秒导入时间
        orig_out, orig_err = sys.stdout, sys.stderr
        done_flag = {"v": False}

        def _emit(text):
            self.s.log.emit(text)
            # 日志出现"训练完成" → 停掉进度监听，进度条立即到100%
            if "训练完成" in text and not done_flag["v"]:
                done_flag["v"] = True
                self._watch_stop.set()   # 先停监听，防止98%覆盖100%
                self.s.prog.emit(100)

        writer = _TrainLogWriter(_emit)
        sys.stdout = writer
        sys.stderr = writer

        # ultralytics logger 在导入时已绑定原流，必须改 handler.stream 才能捕获
        import logging
        saved_streams = []
        for handler in logging.getLogger().handlers:
            saved_streams.append((handler, getattr(handler, "stream", None)))
            handler.stream = writer
        for name in list(logging.root.manager.loggerDict):
            lg = logging.getLogger(name)
            for handler in lg.handlers:
                saved_streams.append((handler, getattr(handler, "stream", None)))
                handler.stream = writer

        try:
            import train as train_mod
            train_mod.main()
            writer.flush()
            if not done_flag["v"]:
                self.s.prog.emit(100)
            time.sleep(0.6)         # 让100%显示一下
            self.s.train_done.emit()
        except Exception as e:
            writer.flush()
            self.s.log.emit(f"训练出错: {e}")
            self.s.prog.emit(100)
            self.s.train_done.emit()
        finally:
            for handler, old_stream in saved_streams:
                if old_stream is not None:
                    handler.stream = old_stream
            sys.stdout, sys.stderr = orig_out, orig_err
            self._watch_stop.set()

    def _watch_progress(self, progress_file):
        last = 0
        while not self._watch_stop.is_set():
            time.sleep(0.5)
            try:
                if os.path.exists(progress_file):
                    with open(progress_file, "r") as f:
                        val = int(f.read().strip())
                    if val > last:  # 严格递增，防止98%覆盖100%
                        last = val
                        self.s.prog.emit(val)
            except:
                pass

    def _on_train_done(self):
        self.pbar.hide()
        if hasattr(self, 'glass_bar') and self.glass_bar.isVisible():
            self.glass_bar.hide()
        self.title.setText("模型训练完成")
        self.title.setStyleSheet("color: #eee; background: transparent;")
        # 弹窗提示，点确定后回到功能选择
        MsgDialog.info(self, "完成", "模型训练完成")
        # 回到功能选择，恢复摄像头画面
        if hasattr(self, 'log_area'):
            self.log_area.hide()
        self.view.show()
        self.view.clear()
        self.view.setStyleSheet("border: 3px solid #555; background: transparent;")
        self.title.setText("AI智能识别系统")
        self.btn.setText("功能选择")
        self.btn.setEnabled(True)
        self.step = 1
        self.running = True
        threading.Thread(target=self._video_loop, daemon=True).start()

    # ---- 识别功能切换 ----
    def _stop_current_det(self):
        """停止当前检测（非阻塞：序号+1，旧线程下一帧自行退出，UI零等待）"""
        with self._switch_lock:
            self._det_seq += 1          # 旧线程回调发现序号变化立即返回False
            self.pose_running = False
            self.running = False
            # 重置锁定确认状态（红框、最新结果、未关闭的弹窗）
            self._lock_on = False
            self._last_det = None
            if self._lock_dlg:
                try:
                    self._lock_dlg.close()
                except Exception:
                    pass
                self._lock_dlg = None
            # 重置所有按钮样式
            self.btn_obj.setStyleSheet(self._det_btn_style)
            self.btn_person.setStyleSheet(self._det_btn_style)
            self.btn_pose.setStyleSheet(self._det_btn_style)
            self._active_det = None

    def _start_det_mode(self, btn, mode):
        """启动某个检测模式"""
        # 检查摄像头
        if not self.cap or not self.cap.isOpened():
            MsgDialog.info(self, "错误", "请先接入摄像头")
            return
        # 检查模型是否加载完成
        if not self._models_loaded:
            MsgDialog.info(self, "提示", "模型正在加载中，请稍候...")
            return
        # 如果点击的是当前正在运行的功能，则关闭
        if self._active_det == mode:
            self._stop_current_det()
            self.running = True
            threading.Thread(target=self._video_loop, daemon=True).start()
            return
        # 停止之前的检测（内部有锁保护，非阻塞）
        self._stop_current_det()
        # 激活新功能（新线程绑定当前会话序号）
        with self._switch_lock:
            self._active_det = mode
            btn.setStyleSheet(self._det_btn_active)
            self.running = False
            seq = self._det_seq
            if mode == "pose":
                self.pose_running = True
                threading.Thread(target=self._pose_thread, args=(seq,), daemon=True).start()
            elif mode == "obj":
                threading.Thread(target=self._obj_thread, args=(seq,), daemon=True).start()
            elif mode == "person":
                threading.Thread(target=self._person_thread, args=(seq,), daemon=True).start()

    def _on_pose_toggle(self):
        self._start_det_mode(self.btn_pose, "pose")

    def _on_obj_toggle(self):
        """点击目标检测：向下弹出模型选择菜单"""
        if not self.cap or not self.cap.isOpened():
            MsgDialog.info(self, "错误", "请先接入摄像头")
            return
        if not self._models_loaded:
            MsgDialog.info(self, "提示", "模型正在加载中，请稍候...")
            return
        # 如果正在运行，则停止
        if self._active_det == "obj":
            self._stop_current_det()
            self.running = True
            threading.Thread(target=self._video_loop, daemon=True).start()
            return
        # 弹出向下菜单选择模型
        sys.path.insert(0, os.path.join(BASE_DIR, "core"))
        import object_detect
        model_names = object_detect.get_model_names()
        if not model_names:
            MsgDialog.info(self, "提示", "没有可用的目标检测模型")
            return
        from PyQt5.QtWidgets import QMenu
        menu = QMenu(self.btn_obj)
        menu.setStyleSheet("""
            QMenu { background: #2E3440; color: #D8DEE9; border: 1px solid #4C566A;
                    border-radius: 6px; padding: 6px; }
            QMenu::item { padding: 8px 20px; border-radius: 4px; }
            QMenu::item:selected { background: #5E81AC; color: #ECEFF4; }
        """)
        for name in model_names:
            action = menu.addAction(name)
            action.triggered.connect(lambda checked, n=name: self._start_obj_model(n))
        # 在按钮下方弹出
        menu.exec_(self.btn_obj.mapToGlobal(self.btn_obj.rect().bottomLeft()))

    def _start_obj_model(self, model_name):
        """启动指定模型的目标检测"""
        self._stop_current_det()
        with self._switch_lock:
            self._active_det = "obj"
            self._active_model = model_name
            self.btn_obj.setStyleSheet(self._det_btn_active)
            self.running = False
            seq = self._det_seq
            threading.Thread(target=self._obj_thread, args=(seq,), daemon=True).start()

    def _on_person_toggle(self):
        self._start_det_mode(self.btn_person, "person")

    def _on_back(self):
        """从识别模式返回功能选择"""
        self._stop_current_det()
        self.det_btn_row.hide()
        self.btn.show()
        self.btn.setText("功能选择")
        self.btn.setEnabled(True)
        self.step = 1
        # 恢复摄像头画面
        if self.cap and self.cap.isOpened():
            self.running = True
            threading.Thread(target=self._video_loop, daemon=True).start()

    def _preload_models(self):
        """并行加载所有识别模型（每个任务一个线程）"""
        sys.path.insert(0, os.path.join(BASE_DIR, "core"))

        threads = []

        def _run_task(task_id, fn):
            """单个任务：执行 + 异常保护 + 完成信号"""
            try:
                fn()
            except Exception as e:
                print(f"[加载任务 {task_id}] 失败: {e}")
            finally:
                try:
                    self.s.task_done.emit(task_id)
                except Exception:
                    pass

        def _start(task_id, fn):
            t = threading.Thread(target=_run_task, args=(task_id, fn), daemon=True)
            threads.append(t)
            t.start()

        # 1. 姿态骨骼最先启动（MediaPipe初始化慢，且torch.load会长时间霸占GIL，
        #    让它抢在YOLO权重加载前跑完）
        import pose
        _start("pose", pose.preload)

        # 2. 每个目标检测模型一个线程
        import object_detect
        for name, path in object_detect.list_model_files():
            _start(name, lambda p=path: object_detect.preload_one(p))

        # 3. 人体检测模型
        import person_detect
        _start("yolo11n", person_detect.preload)

        # 等待全部完成（每个线程内部已有异常保护）
        for t in threads:
            t.join()

        # 所有终端关闭 → 初始化完成
        self.s.models_ready.emit()

    def _on_task_done(self, task_id):
        """单个加载任务完成（主线程）"""
        try:
            if getattr(self, 'loading_page', None):
                self.loading_page.mark_done(task_id)
        except Exception as e:
            print(f"task_done错误: {e}")

    def _on_models_ready(self):
        """模型加载完成信号处理（主线程）"""
        try:
            # 隐藏加载界面，回到主界面
            if getattr(self, 'loading_page', None):
                self.loading_page.stop()
                self.loading_page = None
            # 显示功能按钮
            self.btn.hide()
            self.toggle.hide()
            self.det_btn_row.show()
            self._models_loaded = True
            self.btn_obj.setEnabled(True)
            self.btn_person.setEnabled(True)
            self.btn_pose.setEnabled(True)
        except Exception as e:
            print(f"models_ready错误: {e}")

    def _on_det_stopped(self, mode, seq):
        """检测停止信号处理（主线程中执行）
        只处理当前会话的自然退出；切换时旧会话的通知直接忽略（样式已在停止时重置）"""
        if seq != self._det_seq:
            return
        if self._active_det == mode:
            self._active_det = None
        if mode == "obj":
            self.btn_obj.setStyleSheet(self._det_btn_style)
        elif mode == "person":
            self.btn_person.setStyleSheet(self._det_btn_style)
        elif mode == "pose":
            self.btn_pose.setStyleSheet(self._det_btn_style)

    def _get_frame(self):
        """从帧缓冲获取最新帧（供检测线程使用）"""
        with self._frame_lock:
            if self._frame is not None:
                return True, self._frame.copy()
        return False, None

    def _pose_thread(self, seq):
        try:
            sys.path.insert(0, os.path.join(BASE_DIR, "core"))
            from pose import run_pose_detect

            def cb(frame, people):
                if self._det_seq != seq:  # 会话已切换 → 立即退出
                    return False
                qimg = self._qimg(frame)
                if qimg:
                    self.s.img.emit(qimg)
                return True

            run_pose_detect(get_frame=self._get_frame, callback=cb)
        except Exception as e:
            print(f"姿态骨骼线程错误: {e}")
        finally:
            self.s.det_stopped.emit("pose", seq)

    def _obj_thread(self, seq):
        """目标检测：使用选定的模型检测"""
        try:
            sys.path.insert(0, os.path.join(BASE_DIR, "core"))
            import object_detect

            model_name = getattr(self, '_active_model', None)
            if not model_name:
                return

            object_detect.run_single(get_frame=self._get_frame, model_name=model_name,
                                     callback=self._make_det_cb(seq))
        except Exception as e:
            print(f"目标检测线程错误: {e}")
        finally:
            self.s.det_stopped.emit("obj", seq)

    def _person_thread(self, seq):
        """人体检测：检测画面中的所有人"""
        try:
            sys.path.insert(0, os.path.join(BASE_DIR, "core"))
            import person_detect

            person_detect.run(get_frame=self._get_frame,
                              callback=self._make_det_cb(seq))
        except Exception as e:
            print(f"人体检测线程错误: {e}")
        finally:
            self.s.det_stopped.emit("person", seq)

    # ---- 锁定确认（Ctrl+鼠标左键 → 语音播报+弹窗，姿态骨骼除外） ----
    def _make_det_cb(self, seq):
        """目标/人体检测共用回调：缓存最新结果 + 锁定后叠加红框 + 显示画面"""
        def cb(frame, dets):
            if self._det_seq != seq:  # 会话已切换 → 立即退出
                return False
            with self._det_lock:
                self._last_det = (frame.copy(), list(dets)) if dets else None
            if not dets:
                # 目标丢失 → 自动解除红框锁定，重新识别后需再走 Ctrl+左键 确认流程
                self._lock_on = False
            elif self._lock_on:
                # Yes锁定后：主目标标注红色框
                x1, y1, x2, y2, label, _ = self._primary_det(dets)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"LOCKED {label}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            qimg = self._qimg(frame)
            if qimg:
                self.s.img.emit(qimg)
            return True
        return cb

    @staticmethod
    def _primary_det(dets):
        """主目标 = 面积最大的检测框（触发截取与红框标注用同一规则）"""
        return max(dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))

    def _on_lock_trigger(self):
        """Ctrl+鼠标左键（钩子线程经信号转到主线程）"""
        # 姿态骨骼、非识别模式、无目标、弹窗已打开时均不响应
        if self._active_det not in ("obj", "person"):
            return
        if self._lock_dlg is not None:
            return
        with self._det_lock:
            det = self._last_det
        if not det:
            return
        frame, dets = det
        x1, y1, x2, y2, _, _ = self._primary_det(dets)
        crop = frame[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
        if crop is None or crop.size == 0:
            return
        crop = crop.copy()
        ch, cw = crop.shape[:2]
        cv2.rectangle(crop, (2, 2), (cw - 3, ch - 3), (0, 0, 255), 3)  # 框选目标描红
        dlg = LockDialog(ALERT_TEXT, crop, self)
        dlg.yes_clicked.connect(self._on_lock_yes)
        dlg.answered.connect(self._on_lock_answered)
        self._lock_dlg = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        import tts
        tts.speak(ALERT_TEXT)  # 语音播报弹窗文本

    def _on_lock_yes(self):
        """选择Yes：弹窗消失后画面持续标注红框"""
        self._lock_on = True

    def _on_lock_answered(self):
        """弹窗以任意方式关闭：停语音、清引用（No不改变画面）"""
        import tts
        tts.stop()
        dlg = self.sender()
        if isinstance(dlg, LockDialog):
            if self._lock_dlg is dlg:
                self._lock_dlg = None
            dlg.deleteLater()
        else:
            self._lock_dlg = None

    # ---- 显示 ----
    def _show(self, qimg):
        vw, vh = self.view.width(), self.view.height()
        iw, ih = qimg.width(), qimg.height()
        if vw <= 0 or vh <= 0 or iw <= 0 or ih <= 0:
            return
        # 上下与预显框精确对齐（高度撑满），左右居中处理（不强制对齐）
        scale = vh / ih
        scaled_w = max(1, int(iw * scale))
        scaled = qimg.scaled(scaled_w, vh,
                             Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.view.setPixmap(QPixmap.fromImage(scaled))

    def _qimg(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # .copy() 确保 QImage 拥有自己的内存，防止数据被回收后花屏
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


if __name__ == '__main__':
    # 全局异常保护：打印错误而不是闪退
    def _excepthook(exc_type, exc_value, exc_tb):
        import traceback
        traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.excepthook = _excepthook

    from multiprocessing import freeze_support
    freeze_support()
    app = QApplication(sys.argv)

    state = {"win": None}

    def init_main():
        state["win"] = MainWindow()

    def show_main():
        if state["win"]:
            state["win"].showFullScreen()

    splash = SplashScreen(on_done=show_main)
    QTimer.singleShot(1000, init_main)

    sys.exit(app.exec_())