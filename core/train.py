#!/usr/bin/env python3

# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件
"""
模型训练脚本 - 参考 YOLO 训练流程
训练日志实时输出到 stdout，供主程序读取显示
进度写入 train_progress.txt 供主程序进度条读取
"""

import os
import sys
import time

# 必须在导入ultralytics之前设置：跳过DNS在线检测（可能卡10~20秒）
os.environ.setdefault("YOLO_OFFLINE", "1")

import torch
from ultralytics import YOLO

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CORE_DIR)  # 项目根目录
LABELED_DIR = os.path.join(BASE_DIR, "data", "labeled_images")
DATA_YAML = os.path.join(LABELED_DIR, "data.yaml")
CLASSES_FILE = os.path.join(LABELED_DIR, "classes.txt")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL = os.path.join(MODELS_DIR, "yolo11n.pt")

# 设置模型下载目录为项目内 models 文件夹
os.environ['YOLO_CONFIG_DIR'] = MODELS_DIR
EPOCHS = 5
IMGSZ = 320      # 416比640快约2倍
BATCH = 32       # 416尺寸下可跑更大batch
WORKERS = 12      # i9-13900多核并行加载
PROGRESS_FILE = os.path.join(BASE_DIR, "train_progress.txt")


def log(msg):
    print(msg, flush=True)


def write_progress(val):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(val))


def check_env():
    log("=" * 60)
    log("环境检查")
    log("=" * 60)
    log(f"Python     : {sys.version.split()[0]}")
    log(f"PyTorch    : {torch.__version__}")
    log(f"CUDA       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU        : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        log(f"VRAM       : {props.total_memory / (1024**3):.2f} GB")
    log(f"训练轮数    : {EPOCHS}")
    log(f"图像尺寸    : {IMGSZ}")
    log(f"Batch Size : {BATCH}")
    log(f"Workers    : {WORKERS}")
    # GPU优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log("cuDNN benchmark: True")
    log("TF32: True")
    log("=" * 60)


def check_dataset():
    # 自动生成 data.yaml
    if not os.path.exists(CLASSES_FILE):
        log(f"错误: 未找到类别文件 {CLASSES_FILE}")
        log("请先完成标注")
        return False

    with open(CLASSES_FILE, "r", encoding="utf-8", errors="replace") as f:
        classes = [c.strip() for c in f.readlines() if c.strip()]

    if not classes:
        log("错误: classes.txt 为空")
        return False

    # 写入 data.yaml
    yaml_content = f"""path: {LABELED_DIR.replace(os.sep, '/')}
train: .
val: .
nc: {len(classes)}
names: {classes}
"""
    with open(DATA_YAML, "w") as f:
        f.write(yaml_content)

    log(f"数据集配置: {DATA_YAML}")
    log(f"类别数量: {len(classes)}")
    log(f"类别名称: {classes}")
    return True


def _patch_final_eval():
    """
    跳过训练后的 final_eval 二次验证（卡顿根源）。
    保留 strip_optimizer 清理，删除 Validating 验证步骤。
    """
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.utils.torch_utils import strip_optimizer
    from ultralytics.utils import RANK

    def fast_final_eval(self):
        if RANK in {-1, 0}:
            if self.last.exists():
                strip_optimizer(self.last)
            if self.best.exists():
                strip_optimizer(self.best)

    BaseTrainer.final_eval = fast_final_eval


def train(model=None):
    log("\n" + "=" * 60)
    log("开始训练")
    log("=" * 60)
    log(f"预训练模型 : {MODEL}")
    log(f"数据集     : {DATA_YAML}")
    log(f"总 Epoch   : {EPOCHS}")
    log(f"图像尺寸   : {IMGSZ}")
    log(f"Batch Size : {BATCH}")
    log(f"Workers    : {WORKERS}")
    log(f"设备       : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        log(f"GPU        : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        log(f"显存       : {props.total_memory / (1024**3):.2f} GB")
    log("=" * 60)
    write_progress(14)

    if model is None:
        model = YOLO(MODEL)
    write_progress(16)

    progress = {"current": 0, "start_time": time.time()}
    t_train_call = time.time()  # model.train() 调用时刻

    # 统计数据集 Images / Instances（供日志显示）
    try:
        n_img = n_inst = 0
        for fn in os.listdir(LABELED_DIR):
            if fn.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                n_img += 1
            elif fn.endswith(".txt") and fn != "classes.txt":
                with open(os.path.join(LABELED_DIR, fn), "r", encoding="utf-8", errors="replace") as fh:
                    n_inst += sum(1 for line in fh if line.strip())
        progress["val_counts"] = (n_img, n_inst)
    except Exception:
        progress["val_counts"] = (0, 0)

    def on_pre_routine_start(trainer):
        log(f"[timing] trainer init: {time.time() - t_train_call:.1f}s")

    def on_pre_routine_end(trainer):
        log(f"[timing] dataset & model ready: {time.time() - t_train_call:.1f}s")

    def on_train_timing(trainer):
        log(f"[timing] first batch starts: {time.time() - t_train_call:.1f}s")

    model.add_callback("on_pretrain_routine_start", on_pre_routine_start)
    model.add_callback("on_pretrain_routine_end", on_pre_routine_end)
    model.add_callback("on_train_start", on_train_timing)

    def on_train_epoch_end(trainer):
        progress["current"] = trainer.epoch + 1
        # 训练阶段占 20%~98%，最后2%留给收尾日志
        pct = 20 + int(progress["current"] / EPOCHS * 78)
        progress["pct"] = min(pct, 98)
        write_progress(min(pct, 98))

    def on_fit_epoch_end(trainer):
        # 此时本轮验证已完成，metrics 是当轮最新数据（Ultralytics 原生表格样式）
        try:
            ep = getattr(trainer, "epoch", 0) + 1
            items = {}
            if getattr(trainer, "tloss", None) is not None:
                items = trainer.label_loss_items(trainer.tloss)
            m = getattr(trainer, "metrics", None) or {}

            def _g(key):
                for k in m:
                    if key in str(k):
                        try:
                            return float(m[k])
                        except Exception:
                            return 0.0
                return 0.0

            lr = trainer.optimizer.param_groups[0]["lr"] if getattr(trainer, "optimizer", None) else 0.0
            elapsed = time.time() - progress["start_time"]
            vram = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0

            # 第一行：表头
            log(f"{'Epoch':>7} {'GPU_mem':>8} {'box_loss':>9} {'cls_loss':>9} "
                f"{'dfl_loss':>9} {'lr':>10} {'progress':>9}")
            # 第二行：训练数值
            log(f"{ep:>3}/{EPOCHS:<3} {vram:>6.2f}G {items.get('box_loss', 0):>9.4f} "
                f"{items.get('cls_loss', 0):>9.4f} {items.get('dfl_loss', 0):>9.4f} "
                f"{lr:>10.6f} {progress.get('pct', 20):>8}%")
            # 第三行：验证表头
            log(f"{'':>11} {'Class':>7} {'Images':>7} {'Instances':>9} "
                f"{'Box(P':>8} {'R':>7} {'mAP50':>7} {'mAP50-95':>9}")
            # 第四行：验证数值
            n_img, n_inst = progress.get("val_counts", (0, 0))
            log(f"{'':>11} {'all':>7} {n_img:>7} {n_inst:>9} "
                f"{_g('precision'):>8.4f} {_g('recall'):>7.4f} "
                f"{_g('mAP50('):>7.4f} {_g('mAP50-95'):>9.4f}   "
                f"elapsed: {elapsed:.0f}s")
        except Exception as e:
            log(f"[Epoch] log error: {e}")

    def on_train_start(trainer):
        write_progress(20)  # 训练正式开始

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=WORKERS,
        project=os.path.join(BASE_DIR, "runs"),
        name="train",
        exist_ok=True,
        save=True,
        save_period=0,      # 只保存最后一次，不保存中间checkpoint
        patience=0,
        amp=False,          #True
        verbose=True,
        cache=True,         # 小数据集缓存到内存，省去磁盘读取
        multi_scale=False,  # 关闭多尺度，加速
        plots=False,        # 关闭绘图，加速
        val=False,          # 跳过训练后验证，跑完直接完成
    )

    write_progress(98)  # 训练结束，进入最后2%收尾阶段
    log("\n" + "=" * 60)
    log("训练完成！")
    log("=" * 60)

    # 输出模型路径
    best = os.path.join(BASE_DIR, "runs", "train", "weights", "best.pt")
    last = os.path.join(BASE_DIR, "runs", "train", "weights", "last.pt")
    if os.path.exists(best):
        log(f"最佳模型: {best}")
    if os.path.exists(last):
        log(f"最终权重: {last}")

    return results


def main():
    import threading

    write_progress(2)

    # 模型加载与环境检查并行，缩短前期加载时间
    model_box = {}

    def _load_model():
        try:
            model_box["model"] = YOLO(MODEL)
            model_box["ok"] = True
        except Exception as e:
            model_box["err"] = e

    t = threading.Thread(target=_load_model, daemon=True)
    t.start()

    check_env()
    write_progress(8)

    if not check_dataset():
        write_progress(100)
        return

    write_progress(12)

    t.join()  # 等待模型加载完成（通常已并行完成）
    if "err" in model_box:
        raise model_box["err"]

    _patch_final_eval()  # 跳过训练后二次验证，消除卡顿

    try:
        train(model=model_box.get("model"))
    except Exception as e:
        log(f"\n训练出错: {e}")
        write_progress(100)
        raise


if __name__ == "__main__":
    main()