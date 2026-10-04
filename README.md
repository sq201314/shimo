<div align="center">

# AI Intelligent Recognition System

**Real-time · Multi-model · Fully offline** — Object detection / human detection / pose skeleton, with a complete capture → annotate → train pipeline

**English** | [简体中文](README.zh-CN.md)

[Features](#features) · [Quick Start](#quick-start) · [Workflow](#workflow) · [License](#license)

</div>

---

## Highlights

> **Multi-model object detection** — Auto-scans `models/*.pt`, switch models (tank / warship / balloon) from a dropdown; keyword filtering keeps models isolated
> **33-keypoint pose skeleton** — MediaPipe full-accuracy model, skeleton color-coded by body part
> **End-to-end training chain** — Capture → labelImg annotation → YOLO training, with live logs and a progress bar
> **Immersive interface** — Startup animation, full-screen GIF background, hacker-style multi-terminal loading screen
> **Fully offline deployment** — All models are local files; copy the folder and run, no internet required

---

> **Notice**: This project is intended for **learning and research in AI / computer vision**. It is organized as a personal study project and is not guaranteed to be production-grade; test and evaluate thoroughly before any commercial or downstream use.

---

## Features

<table>
<tr>
<td width="50%">

### Object Detection
- Auto-scans all `.pt` models under `models/`
- Click the button to open a **dropdown menu** and pick a model
- Each model **only detects its own targets**, no cross-talk
- Bounding boxes shown only at confidence ≥ **0.7**, target-specific colors

</td>
<td width="50%">

### Human Detection
- YOLO COCO pretrained model, works out of the box
- **Green boxes** mark people in real time
- Confidence ≥ 0.7 filters false positives
- Boxes only, no skeleton — light and fast

</td>
</tr>
<tr>
<td width="50%">

### Pose Skeleton
- MediaPipe Pose **full-accuracy** model
- **33 keypoints** with skeleton connections
- Color-coded by body part: face / torso / left arm / right arm / left leg / right leg
- Detected every 2 frames, model instance globally cached

</td>
<td width="50%">

### Model Training
- Capture images → annotate with **labelImg** → YOLO training
- Training logs displayed **live** in the view area
- Progress bar updates in sync
- Supports standalone training on large-scale datasets

</td>
</tr>
</table>

---

## Tech Stack

| Area | Technology |
|:----:|------|
| Language | **Python 3.10** |
| UI | **PyQt5** — full-screen GIF background + transparent widgets, responsive to any resolution |
| Inference | **PyTorch 2.11 (CUDA 12.8)** + **ultralytics** — YOLO GPU training and inference |
| Skeleton | **MediaPipe 1.0 Tasks API** — 33-keypoint pose detection |
| Capture | **OpenCV** — multithreaded capture (producer-consumer frame buffer) |
| Annotation | **labelImg** — YOLO-format annotation tool |

> **Recommended hardware**: NVIDIA RTX 5060 8GB (required for training) · 32GB RAM · Windows 10/11

---

## Quick Start

**1. Get the code**

```bash
git clone https://github.com/sq201314/shimo.git
cd shimo
```

**2. Create the environment**

```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
# (1) PyTorch GPU build (CUDA 12.8, must use the official index)
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# (2) Everything else
pip install -r requirements.txt
```

**4. Run**

```bash
python main.py
```

**Driver setup / model downloads / offline deployment → [安装说明书.md](安装说明书.md)** (Chinese)

<details>
<summary><b>Verify your environment</b> (click to expand)</summary>

```bash
# Should print CUDA: True plus your GPU name
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Should show CUDA Version >= 12.8
nvidia-smi
```

</details>

---

## Workflow

```
Startup screen
   |
   v
Connect camera ----------> Function select
                              |
               +--------------+--------------+
               v                             v
        Smart recognition              Model training
        multi-terminal loading          capture images
               |                             |
               v                        labelImg annotate
     +---------+---------+                   |
     v         v         v                   v
  Object     Human      Pose            YOLO training
  detection detection  skeleton         logs + progress
```

---

## License

This project is released under the **MIT License** — free to use, modify, and commercialize, as long as the copyright notice is retained.

```
Copyright (c) 2026 An Xiaoquan
```

Full text: [LICENSE](LICENSE) ｜ [Chinese translation](LICENSE.zh-CN.md)

---

<div align="center">

If this project helps you, a **Star** is appreciated!

Made by **An Xiaoquan**

</div>
