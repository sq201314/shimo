# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

"""
军事目标检测模型训练脚本
独立运行，不被主程序调用
使用 YOLO11m 微调，数据集：13类军事车辆（坦克/步战车/两栖车等）
"""
import os
import sys

# 确保能找到 ultralytics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "venv", "Lib", "site-packages"))

from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "data_tank.yaml")
PROJECT_DIR = os.path.join(BASE_DIR, "runs")

# 基础模型（首次运行会自动下载）
BASE_MODEL = "yolo11m.pt"


def main():
    print("=" * 50)
    print("  坦克检测模型训练")
    print("=" * 50)
    print(f"  数据集: {DATA_YAML}")
    print(f"  基础模型: {BASE_MODEL}")
    print(f"  输出目录: {PROJECT_DIR}")
    print("=" * 50)

    # 加载基础模型
    model = YOLO(BASE_MODEL)

    # 开始训练（10小时，优化速度）
    results = model.train(
        data=DATA_YAML,
        epochs=500,           # 跑500轮，300轮时可手动终止使用
        imgsz=640,           # 416速度快3倍，精度损失<2%
        batch=10,            # 每轮约102批次（4197÷41≈102）
        device=0,            # GPU 0
        project=PROJECT_DIR,
        name="tank_ship",
        exist_ok=True,
        patience=80,          # 早停：80轮无提升停止
        workers=8,           # 加大数据加载线程
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        close_mosaic=10,
        amp=True,            # 混合精度（必须开启）
        cos_lr=True,
        label_smoothing=0.05,
        augment=True,
        cache=True,          # 缓存图片到内存，大幅提速
        rect=True,           # 矩形训练，减少填充
    )

    print("\n" + "=" * 50)
    print("  训练完成！")
    print(f"  最优模型: {PROJECT_DIR}/tank_ship/weights/best.pt")
    print(f"  末尾模型: {PROJECT_DIR}/tank_ship/weights/last.pt")
    print("=" * 50)
    print("\n请将 best.pt 复制到 models/military/best.pt 以供主程序使用")


if __name__ == "__main__":
    main()
