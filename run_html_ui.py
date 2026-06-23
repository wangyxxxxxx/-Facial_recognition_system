import os
import sys
import time
import socket
import threading
import subprocess
import webbrowser
from datetime import datetime
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlsplit

import requests


# ==========================
# 基础配置
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_DIR = os.path.join(BASE_DIR, "html_ui")

#API_HOST = "0.0.0.0"
API_HOST = "127.0.0.1"
API_PORT = 8000
API_BASE = f"http://{API_HOST}:{API_PORT}"

#UI_HOST = "0.0.0.0"
UI_HOST = "127.0.0.1"
UI_PORT = 8080
UI_BASE = f"http://{UI_HOST}:{UI_PORT}"

MAIN_PAGE_URL = f"{UI_BASE}/index.html"
ADMIN_LOGIN_PAGE_URL = f"{UI_BASE}/admin_login.html"

LOG_DIR = os.path.join(BASE_DIR, "api_logs")
os.makedirs(LOG_DIR, exist_ok=True)

LAUNCHER_LOG = os.path.join(LOG_DIR, "html_launcher.log")
BACKEND_LOG = os.path.join(LOG_DIR, "backend.log")


# ==========================
# 日志工具
# ==========================

def log(msg):
    text = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(text, flush=True)
    try:
        with open(LAUNCHER_LOG, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


def pipe_backend_output(proc, log_path):
    def reader():
        try:
            with open(log_path, "a", encoding="utf-8", errors="replace") as f:
                f.write("\n\n========== Backend process output from HTML UI launcher ==========" + "\n")
                f.write(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("===============================================================\n")
                f.flush()
                for line in iter(proc.stdout.readline, ""):
                    if not line:
                        break
                    print(line, end="", flush=True)
                    f.write(line)
                    f.flush()
        except Exception as e:
            log(f"读取后端输出时发生异常：{e}")

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    return t


# ==========================
# 端口和后端管理
# ==========================

def is_port_free(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        s.close()
        return True
    except OSError:
        s.close()
        return False


def is_api_alive():
    try:
        r = requests.get(API_BASE + "/health", timeout=2)
        if r.status_code == 200:
            data = r.json()
            return data.get("status") == "ok"
    except Exception:
        pass
    return False


def wait_for_api(timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        if is_api_alive():
            return True
        time.sleep(1)
    return False


def start_api_server_if_needed():
    if not is_port_free(API_HOST, API_PORT):
        if is_api_alive():
            log(f"检测到已有后端 API 正在运行：{API_BASE}，HTML UI 将直接复用它。")
            return None
        raise RuntimeError(f"端口 {API_PORT} 已被占用，但不是本项目 API。请先释放该端口。")

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
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW

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
    pipe_backend_output(proc, BACKEND_LOG)

    if not wait_for_api(timeout=120):
        stop_api_server(proc)
        raise RuntimeError(f"API 后端启动失败或超时，请查看：{BACKEND_LOG}")

    log("API 后端启动成功。")
    return proc


def stop_api_server(proc):
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
# HTML 静态服务 + API 反向代理
# ==========================

class HtmlUiHandler(SimpleHTTPRequestHandler):
    server_version = "ArcFaceHtmlUI/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HTML_DIR, **kwargs)

    def log_message(self, fmt, *args):
        log("[HTML UI] " + fmt % args)

    def do_GET(self):
        if self.path.startswith("/api/"):
            self.proxy_to_api()
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/"):
            self.proxy_to_api()
            return
        self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def proxy_to_api(self):
        parsed = urlsplit(self.path)
        api_path = parsed.path[len("/api"):]
        if not api_path:
            api_path = "/"
        target_url = API_BASE + api_path
        if parsed.query:
            target_url += "?" + parsed.query

        body = None
        content_length = self.headers.get("Content-Length")
        if content_length:
            body = self.rfile.read(int(content_length))

        headers = {}
        for k, v in self.headers.items():
            lk = k.lower()
            if lk in {"host", "connection", "accept-encoding", "content-length"}:
                continue
            headers[k] = v

        try:
            resp = requests.request(
                method=self.command,
                url=target_url,
                data=body,
                headers=headers,
                timeout=240,
            )
        except Exception as e:
            msg = f"API 代理请求失败：{e}".encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return

        self.send_response(resp.status_code)
        for k, v in resp.headers.items():
            lk = k.lower()
            if lk in {"content-encoding", "transfer-encoding", "connection", "keep-alive"}:
                continue
            self.send_header(k, v)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(resp.content)


def open_browser_pages():
    """
    启动 HTML UI 后自动打开两个页面：
    1. 主页面 index.html
    2. 管理员登录页面 admin_login.html
    """
    try:
        log(f"正在打开主页面：{MAIN_PAGE_URL}")
        webbrowser.open(MAIN_PAGE_URL, new=2)

        # 给浏览器一点时间创建第一个标签页，避免两个页面抢占。
        time.sleep(0.5)

        log(f"正在打开管理员登录页面：{ADMIN_LOGIN_PAGE_URL}")
        webbrowser.open(ADMIN_LOGIN_PAGE_URL, new=2)
    except Exception as e:
        log(f"自动打开浏览器页面失败：{e}")


def start_html_server():
    if not os.path.isdir(HTML_DIR):
        raise RuntimeError(f"HTML UI 目录不存在：{HTML_DIR}")

    if not is_port_free(UI_HOST, UI_PORT):
        raise RuntimeError(f"HTML UI 端口 {UI_PORT} 已被占用，请关闭旧的 HTML UI 或修改端口。")

    server = ThreadingHTTPServer((UI_HOST, UI_PORT), HtmlUiHandler)
    log(f"HTML UI 已启动：{UI_BASE}")
    log("浏览器将自动打开主页面和管理员登录页面。")

    # 让 HTTP 服务先开始监听，再异步打开浏览器。
    threading.Timer(0.8, open_browser_pages).start()

    server.serve_forever()


# ==========================
# 主流程
# ==========================

def main():
    log("========== ArcFace HTML UI 启动 ==========")
    api_proc = None
    try:
        api_proc = start_api_server_if_needed()
        start_html_server()
    except KeyboardInterrupt:
        log("收到退出信号。")
    except Exception as e:
        log(f"HTML UI 运行异常：{e}")
        raise
    finally:
        stop_api_server(api_proc)
        log("========== ArcFace HTML UI 退出 ==========")


if __name__ == "__main__":
    main()
