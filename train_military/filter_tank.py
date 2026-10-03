# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

"""
过滤数据集：只保留坦克类别
原始13类 → 只保留4类坦克（PLA/RUS/TW/US Tank）
"""
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 原始类别（data.yaml中的顺序）
ORIGINAL_NAMES = [
    'PLA Amphibious assault vehicle',   # 0
    'PLA Infantry fighting vehicle',     # 1
    'PLA Tank',                          # 2  ← 保留
    'PLA Vehicle',                       # 3
    'RUS Infantry fighting vehicle',     # 4
    'RUS TANK',                          # 5  ← 保留
    'RUS Vehicle',                       # 6
    'TW Infantry fighting vehicle',      # 7
    'TW Tank',                           # 8  ← 保留
    'US Amphibious assault vehicle',     # 9
    'US Infantry fighting vehicle',      # 10
    'US Tank',                           # 11  ← 保留
    'US Vehicle',                        # 12
]

# 只保留的坦克类别ID
TANK_IDS = {2, 5, 8, 11}

# 新类别映射（旧ID → 新ID）
TANK_MAP = {2: 0, 5: 1, 8: 2, 11: 3}
TANK_NAMES = ['PLA Tank', 'RUS TANK', 'TW Tank', 'US Tank']


def filter_split(split):
    """过滤一个split（train/valid）的标签"""
    img_dir = os.path.join(DATA_DIR, split, "images")
    lbl_dir = os.path.join(DATA_DIR, split, "labels")

    # 输出目录
    out_dir = os.path.join(BASE_DIR, "data_tank")
    out_img = os.path.join(out_dir, split, "images")
    out_lbl = os.path.join(out_dir, split, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    total, kept = 0, 0
    for lbl_file in os.listdir(lbl_dir):
        if not lbl_file.endswith(".txt"):
            continue
        total += 1
        lbl_path = os.path.join(lbl_dir, lbl_file)

        # 读取标注，只保留坦克类别
        new_lines = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id in TANK_IDS:
                    new_cls = TANK_MAP[cls_id]
                    new_lines.append(f"{new_cls} {' '.join(parts[1:])}")

        # 只保留有坦克标注的图片
        if new_lines:
            kept += 1
            # 写新标签
            with open(os.path.join(out_lbl, lbl_file), "w") as f:
                f.write("\n".join(new_lines) + "\n")
            # 复制图片
            img_name = lbl_file.replace(".txt", ".jpg")
            src_img = os.path.join(img_dir, img_name)
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(out_img, img_name))

    print(f"  {split}: {kept}/{total} 张图片含有坦克标注")


def main():
    print("正在过滤数据集（只保留坦克）...")
    filter_split("train")
    filter_split("valid")

    # 创建新的 data.yaml
    yaml_content = f"""train: {BASE_DIR}/data_tank/train/images
val: {BASE_DIR}/data_tank/valid/images

nc: {len(TANK_NAMES)}
names: {TANK_NAMES}
"""
    yaml_path = os.path.join(BASE_DIR, "data_tank.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n过滤完成！新数据集: {BASE_DIR}/data_tank/")
    print(f"新配置文件: {yaml_path}")
    print(f"类别: {TANK_NAMES}")


if __name__ == "__main__":
    main()
