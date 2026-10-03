# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

"""
GPU 预热锁：多线程加载模型时，防止并发 CUDA 推理导致崩溃
只预热第一个模型（YOLO 结构相同，cudnn 算法缓存全局共享），其余跳过
"""
import threading

WARMUP_LOCK = threading.Lock()
_WARMED = False  # 是否已完成过一次GPU预热


def warmup_once(fn):
    """串行执行预热 fn；只允许第一个调用真正执行，其余直接跳过"""
    global _WARMED
    with WARMUP_LOCK:
        if _WARMED:
            return False
        fn()
        _WARMED = True
        return True