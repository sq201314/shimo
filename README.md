<div align="center">

# 🤖 AI 智能识别系统

**全屏实时 · 多模型 · 离线可用** —— 目标检测 / 人体检测 / 姿态骨骼，附完整「采集 → 标注 → 训练」工具链

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11%20CUDA%2012.8-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![PyQt5](https://img.shields.io/badge/UI-PyQt5-41CD52?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Author](https://img.shields.io/badge/Author-An%20Xiaoquan-orange?style=flat-square)

[功能特性](#-功能特性) · [快速开始](#-快速开始) · [使用流程](#-使用流程) · [许可证](#-许可证)

</div>

---

## 🌟 为什么选择这个项目

> 🎯 **多模型目标检测** —— 自动扫描 `models/*.pt`，下拉菜单切换坦克 / 轮船 / 气球等模型，关键词过滤互不串扰
> 🦴 **33 关键点姿态骨骼** —— MediaPipe full 精度模型，按部位分色绘制骨架
> 🎓 **端到端训练链** —— 采集 → labelImg 标注 → YOLO 训练，日志实时显示 + 进度条
> 🖥️ **沉浸式体验** —— 启动动画、GIF 全屏背景、黑客风多终端初始化界面
> 📴 **完全离线部署** —— 所有模型均为本地文件，拷贝即用，无需联网

---

> 📚 **学习声明**：本项目主要用于 **AI / 计算机视觉方向的学习与研究**，旨在实践目标检测、人体姿态估计与模型训练的完整流程。代码与文档按个人学习项目组织，不保证生产级稳定性与精度；如需商用或二次开发，请自行充分测试与评估。欢迎以此为参考互相学习交流。

---

## ✨ 功能特性

<table>
<tr>
<td width="50%">

### 🎯 目标检测
- 扫描 `models/` 下所有 `.pt` 模型
- 点击按钮弹出**下拉菜单**选择模型
- 每个模型**只识别自己的目标**，互不串扰
- 置信度 ≥ **0.7** 才显示框，目标专属配色

</td>
<td width="50%">

### 🧍 人体检测
- YOLO COCO 预训练模型，开箱即用
- **绿色边框**实时标注画面中的人员
- 置信度 ≥ 0.7 过滤误检
- 只画框不画骨骼，轻量高速

</td>
</tr>
<tr>
<td width="50%">

### 🦴 姿态骨骼
- MediaPipe Pose **full 精度**模型
- **33 个关键点** + 骨架连线
- 按部位分色：面部 ⚪ / 躯干 🩵 / 左臂 🟦 / 右臂 🟩 / 左腿 🟪 / 右腿 🟧
- 每 2 帧检测一次，全局缓存实例

</td>
<td width="50%">

### 🎓 模型训练
- 采集图像 → **labelImg** 标注 → YOLO 训练
- 训练日志**实时显示**在画面区
- 进度条同步更新
- 支持大规模数据集独立训练

</td>
</tr>
</table>

<div align="center">

### 🖥️ 界面亮点

 **GIF 全屏动态背景** 　·　 **黑客风多终端初始化**（矩阵代码雨 + 并行加载）

</div>

---

## 🛠️ 技术栈

| 领域 | 技术 |
|:----:|------|
| 🐍 语言 | **Python 3.10** |
| 🎨 界面 | **PyQt5** — 全屏 GIF 背景 + 透明控件，响应式自适应任意分辨率 |
| 🧠 推理 | **PyTorch 2.11 (CUDA 12.8)** + **ultralytics** — YOLO GPU 训练与推理 |
| 🦴 骨骼 | **MediaPipe 1.0 Tasks API** — 33 关键点姿态检测 |
| 📷 采集 | **OpenCV** — 多线程采集（生产者-消费者帧缓冲） |
| 🏷️ 标注 | **labelImg** — YOLO 格式标注工具 |

> 💻 **推荐配置**：NVIDIA RTX 5060 8GB（训练必需）· 32GB 内存 · Windows 10/11

---

## 🚀 快速开始

**1️⃣ 获取代码**

```bash
git clone https://github.com/sq201314/shimo.git
cd shimo
```

**2️⃣ 创建环境**

```bash
python -m venv venv
venv\Scripts\activate
```

**3️⃣ 安装依赖**

```bash
# ① PyTorch GPU 版（CUDA 12.8，需单独指定官方源）
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# ② 其余依赖
pip install -r requirements.txt
```

**4️⃣ 运行**

```bash
python main.py
```

<div align="center">

📖 **驱动安装 / 模型下载 / 离线部署 → [安装说明书.md](安装说明书.md)**

</div>

<br>

<details>
<summary>🔍 <b>验证环境是否就绪</b>（点击展开）</summary>

```bash
# 应输出 CUDA: True + 你的显卡型号
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 应显示 CUDA Version ≥ 12.8
nvidia-smi
```

</details>

---

## 📖 使用流程

```
🚀 启动画面
   │
   ▼
📷 接入无人机画面 ──→ 🎛️ 功能选择
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   🧠 智能识别                      🎓 模型训练
   黑客风初始化加载                   采集图像
          │                             ▼
          ▼                        🏷️ labelImg 标注
   ┌──────┼──────┐                     │
   ▼      ▼      ▼                     ▼
 🎯目标 🧍人体 🦴骨骼            🔥 YOLO 训练
  检测   检测   骨骼              日志 + 进度条
```

---

## 📄 许可证

本项目采用 **MIT 许可证** —— 可自由使用、修改、商用，只需保留版权声明。

```
Copyright (c) 2026 An Xiaoquan
```

完整条款见 [LICENSE](LICENSE) ｜ [中文译本](LICENSE.zh-CN.md)

---

<div align="center">

**如果这个项目对你有帮助，欢迎 ⭐ Star 支持一下！**

Made with ❤️ by **An Xiaoquan**

</div>
