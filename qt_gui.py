import os
import sys
import json
import tempfile
import requests

from PyQt5.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QMessageBox,
)


API_BASE = "http://127.0.0.1:8000"

# gallery 模式由后端管理员配置控制；普通前端只传 auto，不能自行切换 clean/protected
GALLERY_MODE = "auto"

# 你前面根据 clean 负样本统计得到的经验阈值
WATERMARK_THRESHOLD = 0.0643

TOPK = 5


def capture_image_from_camera(parent, title="摄像头采集"):
    """
    使用本机摄像头采集一张图片，并保存为 jpg 文件。

    v5 更新：
    - 在摄像头预览画面中增加人脸轮廓引导框；
    - 提示用户把脸放进绿色椭圆框内；
    - 按 Space/Enter 后不再保存整张摄像头画面，而是保存引导框附近的中心裁剪区域；
    - 这样可以减少天花板、背景、身体区域对 ArcFace embedding 的干扰。
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        QMessageBox.warning(
            parent,
            "缺少摄像头依赖",
            "当前环境未安装 opencv-python，无法使用摄像头采集。\n\n请先运行：\npip install opencv-python",
        )
        return None

    def open_camera():
        backend_candidates = []
        if os.name == "nt":
            backend_candidates.append((getattr(cv2, "CAP_DSHOW", 700), "DirectShow"))
            backend_candidates.append((getattr(cv2, "CAP_MSMF", 1400), "MSMF"))
        backend_candidates.append((None, "Default"))

        tried = []
        for camera_index in (0, 1, 2):
            for backend, backend_name in backend_candidates:
                tried.append(f"index={camera_index}, backend={backend_name}")
                try:
                    if backend is None:
                        cap_obj = cv2.VideoCapture(camera_index)
                    else:
                        cap_obj = cv2.VideoCapture(camera_index, backend)

                    if not cap_obj.isOpened():
                        cap_obj.release()
                        continue

                    cap_obj.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                    ok = False
                    for _ in range(20):
                        ret, frame = cap_obj.read()
                        if ret and frame is not None and frame.size > 0:
                            ok = True
                            break

                    if ok:
                        return cap_obj, camera_index, backend_name, tried

                    cap_obj.release()
                except Exception:
                    try:
                        cap_obj.release()
                    except Exception:
                        pass

        return None, None, None, tried

    def get_face_guide_box(frame):
        """
        返回一个适合人脸录入的中心裁剪框。
        这里不用做人脸检测，只提供固定位置引导，保证采集图里脸占主要区域。
        """
        h, w = frame.shape[:2]

        # 中心略偏上，避免脸在画面下方时额头被截掉
        cx = w // 2
        cy = int(h * 0.48)

        # 人脸引导椭圆大小
        oval_w = int(w * 0.34)
        oval_h = int(h * 0.55)

        # 实际保存的裁剪框比椭圆稍大，保留少量上下文
        crop_w = int(oval_w * 1.55)
        crop_h = int(oval_h * 1.35)

        crop_side = max(crop_w, crop_h)
        crop_side = min(crop_side, h, w)

        x1 = max(0, cx - crop_side // 2)
        y1 = max(0, cy - crop_side // 2)
        x2 = min(w, x1 + crop_side)
        y2 = min(h, y1 + crop_side)

        # 边界修正，保证裁剪框尽量保持正方形
        x1 = max(0, x2 - crop_side)
        y1 = max(0, y2 - crop_side)

        return {
            "cx": cx,
            "cy": cy,
            "oval_w": oval_w,
            "oval_h": oval_h,
            "crop": (x1, y1, x2, y2),
        }

    def draw_face_guide(frame, camera_index, backend_name):
        guide = get_face_guide_box(frame)
        x1, y1, x2, y2 = guide["crop"]
        cx, cy = guide["cx"], guide["cy"]
        oval_w, oval_h = guide["oval_w"], guide["oval_h"]

        show = frame.copy()

        # 让裁剪框外区域稍微变暗，用户更容易把脸放到中间
        overlay = show.copy()
        overlay[:] = (0, 0, 0)
        mask = np.zeros(show.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        dark = cv2.addWeighted(show, 0.35, overlay, 0.65, 0)
        show = np.where(mask[:, :, None] == 255, show, dark)

        # 绿色人脸轮廓椭圆
        cv2.ellipse(
            show,
            (cx, cy),
            (oval_w // 2, oval_h // 2),
            0,
            0,
            360,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

        # 裁剪区域边框，用于提示最终保存范围
        cv2.rectangle(show, (x1, y1), (x2, y2), (0, 180, 255), 2)

        # 中心辅助线
        cv2.line(show, (cx - 18, cy), (cx + 18, cy), (0, 255, 0), 2)
        cv2.line(show, (cx, cy - 18), (cx, cy + 18), (0, 255, 0), 2)

        cv2.putText(
            show,
            "Keep face inside the green oval",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            show,
            "SPACE/ENTER: capture    ESC: cancel",
            (16, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            show,
            f"Camera index={camera_index}, backend={backend_name}",
            (16, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return show, guide

    cap, camera_index, backend_name, tried = open_camera()
    if cap is None:
        QMessageBox.warning(
            parent,
            "摄像头错误",
            "无法打开摄像头。\n\n"
            "请检查：\n"
            "1. 摄像头是否被微信、腾讯会议、浏览器等程序占用；\n"
            "2. Windows 隐私设置是否允许桌面应用访问摄像头；\n"
            "3. PyCharm 是否使用本机解释器，而不是远程解释器。\n\n"
            "已尝试：\n" + "\n".join(tried),
        )
        return None

    QMessageBox.information(
        parent,
        title,
        "即将打开摄像头窗口。\n\n"
        "请把人脸放入绿色椭圆框内，尽量正脸、居中、靠近一些。\n"
        "按 Space 空格键或 Enter 回车键采集图片；\n"
        "按 Esc 取消采集。",
    )

    window_name = "ArcFace Camera Capture"
    captured_frame = None
    captured_crop = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue

            show, guide = draw_face_guide(frame, camera_index, backend_name)
            cv2.imshow(window_name, show)
            key = cv2.waitKey(20)

            if key != -1:
                key_low = key & 0xFF
                if key_low in (13, 10, 32):  # Enter / LF / Space
                    x1, y1, x2, y2 = guide["crop"]
                    captured_frame = frame.copy()
                    captured_crop = captured_frame[y1:y2, x1:x2].copy()
                    break
                if key_low == 27:  # Esc
                    break

    finally:
        cap.release()
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            cv2.destroyAllWindows()

    if captured_crop is None or captured_crop.size == 0:
        QMessageBox.warning(parent, "采集取消", "没有采集到图片。请让人脸位于绿色椭圆框内，并按 Space 或 Enter。")
        return None

    save_dir = os.path.join(os.getcwd(), "api_logs", "camera_images")
    os.makedirs(save_dir, exist_ok=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=save_dir)
    tmp_path = tmp.name
    tmp.close()

    ok, encoded = cv2.imencode(".jpg", captured_crop)
    if not ok:
        QMessageBox.warning(parent, "保存失败", "摄像头图片编码失败。")
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return None

    try:
        encoded.tofile(tmp_path)
    except Exception as e:
        QMessageBox.warning(parent, "保存失败", f"摄像头图片写入失败：\n{e}")
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return None

    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) <= 0:
        QMessageBox.warning(parent, "保存失败", f"摄像头图片保存失败：\n{tmp_path}")
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return None

    return tmp_path


class ApiWorker(QThread):
    finished = Signal(str, dict)
    failed = Signal(str)

    def __init__(self, task, image_path):
        super().__init__()
        self.task = task
        self.image_path = image_path

    def post_image(self, endpoint, data):
        if not self.image_path or not os.path.exists(self.image_path):
            raise FileNotFoundError("请先选择有效图片")

        url = API_BASE + endpoint

        with open(self.image_path, "rb") as f:
            files = {
                "image": (
                    os.path.basename(self.image_path),
                    f,
                    "application/octet-stream",
                )
            }

            r = requests.post(
                url,
                files=files,
                data=data,
                timeout=180,
            )

        try:
            return r.json()
        except Exception:
            return {
                "success": False,
                "error": r.text,
            }

    def run(self):
        try:
            if self.task == "predict":
                result = self.post_image(
                    endpoint="/predict",
                    data={
                        "gallery_mode": GALLERY_MODE,
                        "topk": str(TOPK),
                    },
                )
                self.finished.emit("predict", result)
                return

            if self.task == "watermark":
                # 使用 /both：先用 protected gallery 识别身份，
                # 再自动用识别出的身份做水印检测。
                result = self.post_image(
                    endpoint="/both",
                    data={
                        "gallery_mode": GALLERY_MODE,
                        "threshold": str(WATERMARK_THRESHOLD),
                        "topk": str(TOPK),
                    },
                )
                self.finished.emit("watermark", result)
                return

            if self.task == "admin_verify":
                # 管理员验证只上传当前选择的图片，真正的管理员判断在后端 /admin_verify 中完成。
                result = self.post_image(
                    endpoint="/admin_verify",
                    data={},
                )
                self.finished.emit("admin_verify", result)
                return

            raise ValueError(f"未知任务: {self.task}")

        except Exception as e:
            self.failed.emit(str(e))


class ArcFaceSimpleGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("人脸识别与水印确权系统")
        self.resize(720, 620)

        self.image_path = None
        self.worker = None
        self.auth_window = None
        self.temp_camera_paths = []

        self.build_ui()

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(18)
        main_layout.setContentsMargins(28, 24, 28, 24)

        title = QLabel("人脸识别与水印确权系统")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        main_layout.addWidget(title)

        subtitle = QLabel("请选择一张人脸图片，然后进行身份识别、水印检测或管理员授权验证。识别图库由管理员在授权页面设置。")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Microsoft YaHei", 11))
        subtitle.setStyleSheet("color: #555;")
        main_layout.addWidget(subtitle)

        main_layout.addWidget(self.build_image_group())
        main_layout.addWidget(self.build_action_group())
        main_layout.addWidget(self.build_result_group())

        self.status_label = QLabel("系统已就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; padding: 6px;")
        main_layout.addWidget(self.status_label)

    def build_image_group(self):
        group = QGroupBox("输入图片")
        group.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))

        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        self.preview_label = QLabel("未选择图片")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedHeight(260)
        self.preview_label.setStyleSheet(
            """
            QLabel {
                border: 1px solid #bbb;
                border-radius: 8px;
                background: #f7f7f7;
                color: #777;
                font-size: 16px;
            }
            """
        )

        self.image_path_label = QLabel("图片路径：未选择")
        self.image_path_label.setWordWrap(True)
        self.image_path_label.setStyleSheet("color: #555;")

        select_btn = QPushButton("选择图片")
        select_btn.setFixedHeight(42)
        select_btn.clicked.connect(self.select_image)

        camera_btn = QPushButton("摄像头采集图片")
        camera_btn.setFixedHeight(42)
        camera_btn.clicked.connect(self.capture_input_image)

        image_button_style = """
            QPushButton {
                font-size: 16px;
                background-color: #2563eb;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
        """
        select_btn.setStyleSheet(image_button_style)
        camera_btn.setStyleSheet(image_button_style)

        button_row = QHBoxLayout()
        button_row.setSpacing(12)
        button_row.addWidget(select_btn)
        button_row.addWidget(camera_btn)

        layout.addWidget(self.preview_label)
        layout.addWidget(self.image_path_label)
        layout.addLayout(button_row)

        return group

    def build_action_group(self):
        group = QGroupBox("功能")
        group.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))

        layout = QHBoxLayout(group)
        layout.setSpacing(20)

        self.predict_btn = QPushButton("身份识别")
        self.predict_btn.setFixedHeight(56)
        self.predict_btn.clicked.connect(self.call_predict)

        self.watermark_btn = QPushButton("水印检测")
        self.watermark_btn.setFixedHeight(56)
        self.watermark_btn.clicked.connect(self.call_watermark)

        self.auth_btn = QPushButton("进入授权页面")
        self.auth_btn.setFixedHeight(56)
        self.auth_btn.clicked.connect(self.call_admin_verify)

        button_style = """
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                background-color: #16a34a;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #15803d;
            }
            QPushButton:disabled {
                background-color: #aaa;
            }
        """

        self.predict_btn.setStyleSheet(button_style)
        self.watermark_btn.setStyleSheet(button_style)
        self.auth_btn.setStyleSheet(button_style)

        layout.addWidget(self.predict_btn)
        layout.addWidget(self.watermark_btn)
        layout.addWidget(self.auth_btn)

        return group

    def build_result_group(self):
        group = QGroupBox("结果")
        group.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))

        layout = QVBoxLayout(group)
        layout.setSpacing(16)

        self.identity_result = QLabel("身份识别结果：未检测")
        self.identity_result.setAlignment(Qt.AlignCenter)
        self.identity_result.setFixedHeight(70)
        self.identity_result.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        self.identity_result.setStyleSheet(
            """
            QLabel {
                border-radius: 10px;
                background-color: #eef2ff;
                color: #1e3a8a;
            }
            """
        )

        self.watermark_result = QLabel("水印检测结果：未检测")
        self.watermark_result.setAlignment(Qt.AlignCenter)
        self.watermark_result.setFixedHeight(70)
        self.watermark_result.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        self.watermark_result.setStyleSheet(
            """
            QLabel {
                border-radius: 10px;
                background-color: #f1f5f9;
                color: #334155;
            }
            """
        )

        layout.addWidget(self.identity_result)
        layout.addWidget(self.watermark_result)

        return group

    def set_input_image(self, path, source="图片"):
        if not path:
            return

        self.image_path = path
        self.image_path_label.setText(f"{source}路径：{path}")

        pix = QPixmap(path)
        if not pix.isNull():
            pix = pix.scaled(
                420,
                250,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.preview_label.setPixmap(pix)
            self.preview_label.setText("")
        else:
            self.preview_label.setText("图片预览失败")

        self.identity_result.setText("身份识别结果：未检测")
        self.watermark_result.setText("水印检测结果：未检测")
        self.set_watermark_neutral()

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )

        if not path:
            return

        self.set_input_image(path, source="图片")

    def capture_input_image(self):
        path = capture_image_from_camera(self, title="摄像头采集图片")
        if not path:
            return

        self.temp_camera_paths.append(path)
        self.set_input_image(path, source="摄像头采集图片")

    def require_image(self):
        if not self.image_path or not os.path.exists(self.image_path):
            QMessageBox.warning(self, "缺少图片", "请先选择图片")
            return False
        return True

    def set_buttons_enabled(self, enabled):
        self.predict_btn.setEnabled(enabled)
        self.watermark_btn.setEnabled(enabled)
        self.auth_btn.setEnabled(enabled)

    def start_worker(self, task):
        self.set_buttons_enabled(False)
        self.status_label.setText("正在处理，请稍候...")

        self.worker = ApiWorker(
            task=task,
            image_path=self.image_path,
        )

        self.worker.finished.connect(self.on_worker_finished)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.start()

    def call_predict(self):
        if not self.require_image():
            return
        self.start_worker("predict")

    def call_watermark(self):
        if not self.require_image():
            return
        self.start_worker("watermark")

    def call_admin_verify(self):
        if not self.require_image():
            return

        reply = QMessageBox.question(
            self,
            "管理员验证",
            "将使用当前选择的图片进行管理员身份验证。\n\n验证通过后才能进入授权页面，是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply != QMessageBox.Yes:
            return

        self.start_worker("admin_verify")

    def on_worker_finished(self, task, result):
        self.set_buttons_enabled(True)
        self.status_label.setText("处理完成")

        if not result.get("success", False):
            error = result.get("error", "未知错误")
            QMessageBox.warning(self, "处理失败", error)
            return

        if task == "predict":
            pred_label = result.get("pred_label", "未知")
            gallery_mode = result.get("gallery_mode", "未知")
            top1 = result.get("top1_cosine", 0.0)
            try:
                self.identity_result.setText(f"身份识别结果：{pred_label}（{gallery_mode}, {float(top1):.4f}）")
            except Exception:
                self.identity_result.setText(f"身份识别结果：{pred_label}（{gallery_mode}）")
            return

        if task == "watermark":
            watermark = result.get("watermark", {})
            detected = bool(watermark.get("detected", False))
            gallery_mode = result.get("gallery_mode", "未知")

            if detected:
                self.watermark_result.setText(f"水印检测结果：检测到水印（识别图库：{gallery_mode}）")
                self.set_watermark_positive()
            else:
                self.watermark_result.setText(f"水印检测结果：未检测到水印（识别图库：{gallery_mode}）")
                self.set_watermark_negative()

            return

        if task == "admin_verify":
            verified = bool(result.get("verified", False))
            score = float(result.get("score", 0.0))
            threshold = float(result.get("threshold", 0.0))
            admin_label = result.get("admin_label", "未知")

            if verified:
                QMessageBox.information(
                    self,
                    "管理员验证通过",
                    f"管理员验证成功。\n\n管理员身份：{admin_label}\nscore={score:.4f}\nthreshold={threshold:.4f}",
                )
                self.open_auth_page(result)
            else:
                QMessageBox.warning(
                    self,
                    "管理员验证失败",
                    f"当前人脸不是管理员，禁止进入授权页面。\n\n管理员身份：{admin_label}\nscore={score:.4f}\nthreshold={threshold:.4f}",
                )

            return

    def open_auth_page(self, admin_result):
        from auth_page import AuthPageWindow

        self.auth_window = AuthPageWindow(
            admin_result=admin_result,
            image_path=self.image_path,
            parent=None,
        )
        self.auth_window.show()
        self.auth_window.raise_()
        self.auth_window.activateWindow()

    def on_worker_failed(self, error):
        self.set_buttons_enabled(True)
        self.status_label.setText("处理失败")
        QMessageBox.warning(self, "请求失败", error)

    def set_watermark_positive(self):
        self.watermark_result.setStyleSheet(
            """
            QLabel {
                border-radius: 10px;
                background-color: #dcfce7;
                color: #166534;
            }
            """
        )

    def set_watermark_negative(self):
        self.watermark_result.setStyleSheet(
            """
            QLabel {
                border-radius: 10px;
                background-color: #fee2e2;
                color: #991b1b;
            }
            """
        )

    def set_watermark_neutral(self):
        self.watermark_result.setStyleSheet(
            """
            QLabel {
                border-radius: 10px;
                background-color: #f1f5f9;
                color: #334155;
            }
            """
        )


def main():
    app = QApplication(sys.argv)
    window = ArcFaceSimpleGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()