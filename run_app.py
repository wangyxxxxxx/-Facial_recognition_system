import os
import sys
import time
import socket
import subprocess
import threading
import requests
from datetime import datetime


# ==========================
# 基础配置
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

API_HOST = "127.0.0.1"
API_PORT = 8000
API_BASE = f"http://{API_HOST}:{API_PORT}"

LOG_DIR = os.path.join(BASE_DIR, "api_logs")
os.makedirs(LOG_DIR, exist_ok=True)

LAUNCHER_LOG = os.path.join(LOG_DIR, "launcher.log")
BACKEND_LOG = os.path.join(LOG_DIR, "backend.log")


# ==========================
# 日志工具
# ==========================

def log(msg):
    text = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(text)

    try:
        with open(LAUNCHER_LOG, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


def show_error(title, message):
    """
    Windows 下弹窗提示；其它环境直接打印。
    """
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
    except Exception:
        print(title)
        print(message)


def pipe_backend_output(proc, log_path):
    """
    将后端 uvicorn 输出同时写到：
    1. PyCharm 运行窗口
    2. api_logs/backend.log
    """

    def reader():
        try:
            with open(log_path, "a", encoding="utf-8", errors="replace") as f:
                f.write("\n\n")
                f.write("========== Backend process output ==========\n")
                f.write(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("============================================\n")
                f.flush()

                for line in iter(proc.stdout.readline, ""):
                    if not line:
                        break

                    # 输出到 PyCharm 运行窗口
                    print(line, end="")

                    # 同时写入 backend.log
                    f.write(line)
                    f.flush()

        except Exception as e:
            log(f"读取后端输出时发生异常：{e}")

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    return t


# ==========================
# 端口和 API 检查
# ==========================

def is_port_free(host, port):
    """
    检查端口是否空闲。
    True: 端口空闲
    False: 端口已被占用
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.bind((host, port))
        s.close()
        return True
    except OSError:
        s.close()
        return False


def is_api_alive():
    """
    检查当前 8000 端口上是否已经有本项目 API 在运行。
    要求 api_server_lab.py 里有 /health 接口，并返回 {"status": "ok"}。
    """
    try:
        r = requests.get(API_BASE + "/health", timeout=2)
        if r.status_code == 200:
            data = r.json()
            return data.get("status") == "ok"
    except Exception:
        pass

    return False


def wait_for_api(timeout=120):
    """
    等待 API 后端启动成功。
    """
    start = time.time()

    while time.time() - start < timeout:
        if is_api_alive():
            return True
        time.sleep(1)

    return False


# ==========================
# 后端进程管理
# ==========================

def start_api_server():
    """
    启动 FastAPI 后端。
    """

    if not is_port_free(API_HOST, API_PORT):
        if is_api_alive():
            msg = (
                f"检测到 {API_BASE} 已经有本项目 API 服务在运行。\n\n"
                "为避免误关闭其它进程，本程序不会复用或关闭该服务。\n\n"
                "请先手动关闭已有的 API 服务，然后重新启动本程序。"
            )
            log(msg)
            show_error("端口已被占用", msg)
            raise RuntimeError("API 服务已经在运行")
        else:
            msg = (
                f"检测到端口 {API_PORT} 已经被其它程序占用。\n\n"
                "为避免误关闭无关进程，本程序不会自动关闭占用该端口的程序。\n\n"
                "请先手动释放该端口，然后重新启动本程序。"
            )
            log(msg)
            show_error("端口已被占用", msg)
            raise RuntimeError(f"端口 {API_PORT} 已被占用")

    log("正在启动 API 后端...")

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "uvicorn",
        "api_server_lab:app",
        "--host",
        API_HOST,
        "--port",
        str(API_PORT),
    ]

    creationflags = 0

    # Windows 下不额外弹出黑色 cmd 窗口
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )

        log(f"API 后端进程已启动，PID={proc.pid}")
        log(f"后端日志文件：{BACKEND_LOG}")

        # 同时输出到 PyCharm 和 backend.log
        pipe_backend_output(proc, BACKEND_LOG)

    except Exception as e:
        msg = f"API 后端启动失败：\n{e}"
        log(msg)
        show_error("API 后端启动失败", msg)
        raise

    ok = wait_for_api(timeout=120)

    if not ok:
        log("API 后端启动失败或超时，准备关闭本次启动的 API 进程树。")
        stop_api_server(proc)

        msg = (
            "API 后端启动失败或超时。\n\n"
            "常见原因：\n"
            "1. api_server_lab.py 报错\n"
            "2. 权重、gallery 或 watermark key 路径不正确\n"
            "3. 当前环境缺少依赖\n"
            "4. 端口被临时占用\n\n"
            f"请查看后端日志：\n{BACKEND_LOG}"
        )

        show_error("API 后端启动失败", msg)
        raise RuntimeError("API 后端启动失败或超时")

    log("API 后端启动成功。")
    return proc


def stop_api_server(proc):
    """
    只关闭本次 run_app.py 启动的 API 后端进程。
    不根据端口乱杀其它程序。
    """

    if proc is None:
        return

    try:
        if proc.poll() is not None:
            return

        log(f"正在关闭 API 后端进程，PID={proc.pid}")

        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            proc.terminate()

            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

        log("API 后端进程已关闭。")

    except Exception as e:
        log(f"关闭 API 后端进程时发生异常：{e}")


# ==========================
# 主流程
# ==========================

def main():
    log("========== ArcFace 系统启动 ==========")

    api_proc = None

    try:
        api_proc = start_api_server()

        log("正在启动 Qt GUI...")

        from qt_gui import main as qt_main

        qt_main()

    except SystemExit:
        # Qt 正常退出时可能触发 SystemExit，这里直接放行
        raise

    except Exception as e:
        msg = f"程序运行异常：\n{e}"
        log(msg)
        show_error("ArcFace 运行异常", msg)

    finally:
        stop_api_server(api_proc)
        log("========== ArcFace 系统退出 ==========")


if __name__ == "__main__":
    main()