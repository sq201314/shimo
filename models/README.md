# 目标检测模型目录

每个子文件夹为一个独立模型，包含：
- `best.pt` — YOLO 模型权重
- `classes.txt` — 类别名称（每行一个）

## 已有模型

| 文件夹 | 目标 | 状态 |
|--------|------|------|
| military/ | 坦克 | ✅ 已训练 |
| balloon/ | 气球 | ❌ 需下载 |
| ship/ | 轮船 | ❌ 需下载 |

## 添加新模型

1. 创建文件夹 `models/你的模型名/`
2. 放入 `best.pt`（YOLO格式）
3. 放入 `classes.txt`（类别名称）
4. 重启程序，自动加载

## 气球模型下载

在终端执行：
```bash
d:\shimo\venv\Scripts\python.exe -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='keremberke/yolov8m-balloon-detection', filename='best.pt', local_dir='d:/shimo/models/balloon', local_dir_use_symlinks=False)
print(f'下载完成: {path}')
"
```

## 轮船模型下载

在终端执行：
```bash
d:\shimo\venv\Scripts\python.exe -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='keremberke/yolov8n-vessel-detection', filename='best.pt', local_dir='d:/shimo/models/ship', local_dir_use_symlinks=False)
print(f'下载完成: {path}')
"
```

## 自己训练的模型

将训练好的 `best.pt` 复制到对应文件夹即可：
```bash
copy 训练输出\weights\best.pt models\你的模型名\best.pt
```