# Copyright (c) 2026 An Xiaoquan
# SPDX-License-Identifier: MIT
# 本项目采用 MIT 许可证，完整声明见 LICENSE 文件

"""
语音播报模块：Windows 自带 System.Speech（PowerShell 调用，离线零依赖）
优先选择中文语音（zh-CN，如 Huihui），后台线程播报不阻塞 UI
"""
import base64
import subprocess
import threading

_lock = threading.RLock()   # 可重入：speak 内部先 stop 上一条
_proc = None                 # 当前播报的 PowerShell 进程

# 锁定确认播报文本（弹窗显示与语音播报同一份内容）
ALERT_TEXT = "已经发现并锁定目标是否进行打击。"


def speak(text=ALERT_TEXT):
    """异步播报（新播报会打断上一条），立即返回"""
    threading.Thread(target=_speak_sync, args=(text,), daemon=True).start()


def stop():
    """停止当前播报（异步，立即返回）"""
    threading.Thread(target=_stop_sync, daemon=True).start()


def _speak_sync(text):
    global _proc
    safe = text.replace("'", "''")  # PowerShell 单引号转义
    # -EncodedCommand 用 UTF-16LE base64 传参，避免中文命令行编码问题
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$v=$s.GetInstalledVoices()|Where-Object"
        "{$_.VoiceInfo.Culture.Name -like 'zh*' -and $_.Enabled}"
        "|Select-Object -First 1;"
        "if($v){$s.SelectVoice($v.VoiceInfo.Name)};"
        f"$s.Speak('{safe}')"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    with _lock:
        _stop_sync()  # 打断上一条未完成的播报
        try:
            _proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive",
                 "-EncodedCommand", encoded],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,  # 不弹黑窗
            )
            _proc.wait()
        except Exception as e:
            print(f"[语音] 播报失败: {e}")
        finally:
            _proc = None


def _stop_sync():
    global _proc
    with _lock:
        if _proc and _proc.poll() is None:
            try:
                _proc.terminate()
            except Exception:
                pass
            _proc = None
