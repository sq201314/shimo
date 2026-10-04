# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

"""
全局热键模块：Ctrl + 鼠标左键 触发回调
使用 Windows 低级鼠标钩子（WH_MOUSE_LL，ctypes 实现，无第三方依赖）
钩子安装在独立线程的消息循环上，回调在该线程触发，由调用方自行转 Qt 信号
"""
import ctypes
import threading
from ctypes import wintypes

_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32

WH_MOUSE_LL = 14  # 注意：13 是 WH_KEYBOARD_LL，14 才是 WH_MOUSE_LL
WM_LBUTTONDOWN = 0x0201
WM_QUIT = 0x0012
VK_CONTROL = 0x11

# 钩子过程：返回值是 LONG_PTR，64位下用 c_ssize_t
_HOOKPROC = ctypes.WINFUNCTYPE(
    ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
)

# 必须声明参数/返回类型：默认按32位截断，64位下 HHOOK/HMODULE 句柄会丢高位导致钩子安装失败
_user32.SetWindowsHookExW.argtypes = [
    ctypes.c_int, _HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
_user32.SetWindowsHookExW.restype = wintypes.HANDLE
_user32.CallNextHookEx.argtypes = [
    wintypes.HANDLE, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
_user32.CallNextHookEx.restype = ctypes.c_ssize_t
_user32.UnhookWindowsHookEx.argtypes = [wintypes.HANDLE]
_user32.UnhookWindowsHookEx.restype = wintypes.BOOL
_user32.GetMessageW.argtypes = [
    ctypes.c_void_p, wintypes.HWND, ctypes.c_uint, ctypes.c_uint]
_user32.GetMessageW.restype = ctypes.c_int
_kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
_kernel32.GetModuleHandleW.restype = wintypes.HMODULE
_kernel32.GetCurrentThreadId.argtypes = []
_kernel32.GetCurrentThreadId.restype = wintypes.DWORD

_callback = None       # 触发回调（无参数）
_proc_ref = None       # 保持回调对象引用，防止被GC后钩子崩溃
_hook = 0              # 当前钩子句柄
_thread = None
_thread_id = 0
_stop = threading.Event()


def _proc(n_code, w_param, l_param):
    """低级鼠标钩子过程：按下鼠标左键且 Ctrl 按住时触发回调"""
    try:
        if n_code >= 0 and w_param == WM_LBUTTONDOWN and _callback:
            if _user32.GetAsyncKeyState(VK_CONTROL) & 0x8000:
                _callback()
    except Exception as e:
        print(f"[热键] 回调执行错误: {e}")
    return _user32.CallNextHookEx(_hook, n_code, w_param, l_param)


def _loop():
    """钩子线程：安装钩子 + 消息循环（低级钩子要求安装线程有消息泵）"""
    global _hook, _thread_id
    _thread_id = _kernel32.GetCurrentThreadId()
    _hook = _user32.SetWindowsHookExW(
        WH_MOUSE_LL, _proc_ref, _kernel32.GetModuleHandleW(None), 0
    )
    if not _hook:
        print("[热键] 鼠标钩子安装失败")
        return
    print("[热键] Ctrl+鼠标左键 钩子已启动")
    msg = wintypes.MSG()
    while not _stop.is_set():
        ret = _user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
        if ret <= 0:  # 0=WM_QUIT，-1=错误
            break
        _user32.TranslateMessage(ctypes.byref(msg))
        _user32.DispatchMessageW(ctypes.byref(msg))
    if _hook:
        _user32.UnhookWindowsHookEx(_hook)
        _hook = 0


def start(callback):
    """启动钩子（重复调用只替换回调）
    callback: 无参数函数，在钩子线程中被调用
    """
    global _callback, _proc_ref, _thread, _stop
    _callback = callback
    if _proc_ref is None:
        _proc_ref = _HOOKPROC(_proc)  # 必须持有引用，否则钩子调用时函数对象已被回收
    if _thread and _thread.is_alive():
        return
    _stop = threading.Event()
    _thread = threading.Thread(target=_loop, daemon=True)
    _thread.start()


def stop():
    """停止钩子（投递 WM_QUIT 退出消息循环）"""
    global _thread, _thread_id, _callback
    _callback = None
    _stop.set()
    if _thread and _thread.is_alive() and _thread_id:
        _user32.PostThreadMessageW(_thread_id, WM_QUIT, 0, 0)
        _thread.join(timeout=1)
    _thread = None
    _thread_id = 0
