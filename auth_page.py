import os
import tempfile
import requests

from PyQt5.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QMessageBox,
    QFileDialog,
    QLineEdit,
    QComboBox,
    QCheckBox,
)


API_BASE = "http://127.0.0.1:8000"


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


class RuntimeConfigWorker(QThread):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, admin_token, recognition_gallery_mode=None, score_gallery_mode=None):
        super().__init__()
        self.admin_token = admin_token
        self.recognition_gallery_mode = recognition_gallery_mode
        self.score_gallery_mode = score_gallery_mode

    def run(self):
        try:
            if not self.admin_token:
                raise ValueError("管理员 token 不存在，请重新进行管理员验证")

            url = API_BASE + "/admin_runtime_config"
            data = {
                "admin_token": self.admin_token,
            }

            # None 表示只读取当前配置；有值表示修改配置。
            if self.recognition_gallery_mode is not None:
                data["recognition_gallery_mode"] = self.recognition_gallery_mode
            if self.score_gallery_mode is not None:
                data["score_gallery_mode"] = self.score_gallery_mode

            r = requests.post(url, data=data, timeout=30)

            try:
                result = r.json()
            except Exception:
                result = {
                    "success": False,
                    "error": r.text,
                }

            self.finished.emit(result)

        except Exception as e:
            self.failed.emit(str(e))


class EnrollWorker(QThread):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, image_path, label, overwrite, admin_token):
        super().__init__()
        self.image_path = image_path
        self.label = label
        self.overwrite = overwrite
        self.admin_token = admin_token

    def run(self):
        try:
            if not self.image_path or not os.path.exists(self.image_path):
                raise FileNotFoundError("请先选择要录入的人脸图片")

            if not self.label or str(self.label).strip() == "":
                raise ValueError("请输入要录入的身份编号 label")

            if not self.admin_token:
                raise ValueError("管理员 token 不存在，请重新进行管理员验证")

            url = API_BASE + "/enroll_face"

            with open(self.image_path, "rb") as f:
                files = {
                    "image": (
                        os.path.basename(self.image_path),
                        f,
                        "application/octet-stream",
                    )
                }
                data = {
                    "label": str(self.label).strip(),
                    "overwrite": "true" if self.overwrite else "false",
                    "admin_token": self.admin_token,
                    "save_request_image": "true",
                }

                r = requests.post(
                    url,
                    files=files,
                    data=data,
                    timeout=240,
                )

            try:
                result = r.json()
            except Exception:
                result = {
                    "success": False,
                    "error": r.text,
                }

            self.finished.emit(result)

        except Exception as e:
            self.failed.emit(str(e))


class DeleteWorker(QThread):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, label, admin_token):
        super().__init__()
        self.label = label
        self.admin_token = admin_token

    def run(self):
        try:
            if not self.label or str(self.label).strip() == "":
                raise ValueError("请输入要删除的身份编号 label")

            if not self.admin_token:
                raise ValueError("管理员 token 不存在，请重新进行管理员验证")

            url = API_BASE + "/delete_face"
            data = {
                "label": str(self.label).strip(),
                "admin_token": self.admin_token,
            }

            r = requests.post(url, data=data, timeout=120)

            try:
                result = r.json()
            except Exception:
                result = {
                    "success": False,
                    "error": r.text,
                }

            self.finished.emit(result)

        except Exception as e:
            self.failed.emit(str(e))


class AuthPageWindow(QWidget):
    """
    独立授权页面窗口。

    这个窗口只在管理员人脸验证通过后由 qt_gui.py 打开。
    当前版本包含：
        1. 录入用户：选择/采集人脸图片 + 输入身份编号 label；
           后端固定写入 clean gallery，并自动生成 protected gallery 和 watermark key；
        2. 删除用户：输入身份编号 label；
           后端固定从 clean gallery 删除，并自动同步 protected gallery 和 watermark key。
    """

    def __init__(self, admin_result=None, image_path=None, parent=None):
        super().__init__(parent)

        self.admin_result = admin_result or {}
        self.image_path = image_path
        self.admin_token = self.admin_result.get("admin_token", "")
        self.enroll_image_path = None
        self.enroll_worker = None
        self.delete_worker = None
        self.config_worker = None
        self.temp_camera_paths = []

        self.setWindowTitle("授权管理页面")
        self.resize(760, 820)

        self.build_ui()
        self.call_load_runtime_config()

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(28, 24, 28, 24)

        title = QLabel("授权管理页面")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        title.setStyleSheet("color: #14532d;")
        main_layout.addWidget(title)

        subtitle = QLabel("管理员身份验证已通过，可以进行授权和用户录入操作。")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Microsoft YaHei", 11))
        subtitle.setStyleSheet("color: #555;")
        main_layout.addWidget(subtitle)

        main_layout.addWidget(self.build_admin_info_group())
        main_layout.addWidget(self.build_system_mode_group())
        main_layout.addWidget(self.build_enroll_group())
        main_layout.addWidget(self.build_delete_group())
        main_layout.addWidget(self.build_log_group())

        close_btn = QPushButton("关闭授权页面")
        close_btn.setFixedHeight(44)
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet(
            """
            QPushButton {
                font-size: 16px;
                background-color: #334155;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #1e293b;
            }
            """
        )
        main_layout.addWidget(close_btn)

    def build_admin_info_group(self):
        group = QGroupBox("管理员验证信息")
        group.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))

        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        admin_label = self.admin_result.get("admin_label", "未知")
        score = float(self.admin_result.get("score", 0.0))
        threshold = float(self.admin_result.get("threshold", 0.0))
        gallery_mode = self.admin_result.get("gallery_mode", "未知")
        token_expires_at = self.admin_result.get("token_expires_at", "")

        info = QLabel(
            f"管理员身份：{admin_label}\n"
            f"验证分数：{score:.4f}\n"
            f"验证阈值：{threshold:.4f}\n"
            f"验证图库：{gallery_mode}\n"
            f"授权 token 过期时间：{token_expires_at or '未知'}"
        )
        info.setFont(QFont("Microsoft YaHei", 12))
        info.setStyleSheet(
            """
            QLabel {
                background-color: #dcfce7;
                color: #166534;
                border-radius: 8px;
                padding: 12px;
            }
            """
        )
        layout.addWidget(info)

        if self.image_path and os.path.exists(self.image_path):
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            pix = QPixmap(self.image_path)
            if not pix.isNull():
                pix = pix.scaled(220, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_label.setPixmap(pix)
                layout.addWidget(image_label)

        return group

    def build_system_mode_group(self):
        group = QGroupBox("系统模式设置（管理员权限）")
        group.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))

        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        note = QLabel(
            "只有管理员可以修改这两个模式。普通主界面和黑盒 API 会使用这里的后端配置，"
            "即使外部请求手动传 gallery_mode，后端也会按这里的设置执行。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #475569;")
        layout.addWidget(note)

        row = QHBoxLayout()
        row.setSpacing(10)

        recognition_label = QLabel("人脸识别图库：")
        recognition_label.setFont(QFont("Microsoft YaHei", 11))

        self.recognition_mode_combo = QComboBox()
        self.recognition_mode_combo.addItems(["protected", "clean"])
        self.recognition_mode_combo.setFixedHeight(36)

        score_label = QLabel("API 分数图库：")
        score_label.setFont(QFont("Microsoft YaHei", 11))

        self.score_mode_combo = QComboBox()
        self.score_mode_combo.addItems(["protected", "clean"])
        self.score_mode_combo.setFixedHeight(36)

        self.refresh_config_btn = QPushButton("读取当前模式")
        self.refresh_config_btn.setFixedHeight(38)
        self.refresh_config_btn.clicked.connect(self.call_load_runtime_config)

        self.apply_config_btn = QPushButton("保存模式设置")
        self.apply_config_btn.setFixedHeight(38)
        self.apply_config_btn.clicked.connect(self.call_update_runtime_config)

        button_style = """
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                background-color: #7c3aed;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #6d28d9;
            }
            QPushButton:disabled {
                background-color: #aaa;
            }
        """
        self.refresh_config_btn.setStyleSheet(button_style)
        self.apply_config_btn.setStyleSheet(button_style)

        row.addWidget(recognition_label)
        row.addWidget(self.recognition_mode_combo)
        row.addWidget(score_label)
        row.addWidget(self.score_mode_combo)
        row.addWidget(self.refresh_config_btn)
        row.addWidget(self.apply_config_btn)
        layout.addLayout(row)

        self.runtime_config_result = QLabel("当前模式：未读取")
        self.runtime_config_result.setAlignment(Qt.AlignCenter)
        self.runtime_config_result.setFixedHeight(48)
        self.runtime_config_result.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.runtime_config_result.setStyleSheet(
            """
            QLabel {
                border-radius: 8px;
                background-color: #f3e8ff;
                color: #581c87;
            }
            """
        )
        layout.addWidget(self.runtime_config_result)

        return group


    def build_enroll_group(self):
        group = QGroupBox("录入用户")
        group.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))

        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        row1 = QHBoxLayout()
        row1.setSpacing(10)

        label_text = QLabel("身份编号：")
        label_text.setFont(QFont("Microsoft YaHei", 11))

        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("例如：201 或 user_001")
        self.label_input.setFixedHeight(36)

        gallery_info = QLabel("录入策略：clean → 自动生成 protected + watermark key")
        gallery_info.setFont(QFont("Microsoft YaHei", 10))
        gallery_info.setStyleSheet("color: #166534;")

        self.overwrite_check = QCheckBox("覆盖已有身份")
        self.overwrite_check.setFont(QFont("Microsoft YaHei", 10))

        row1.addWidget(label_text)
        row1.addWidget(self.label_input, 2)
        row1.addWidget(gallery_info, 2)
        row1.addWidget(self.overwrite_check)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(10)

        self.enroll_path_label = QLabel("录入图片：未选择")
        self.enroll_path_label.setWordWrap(True)
        self.enroll_path_label.setStyleSheet("color: #555;")

        select_btn = QPushButton("选择录入图片")
        select_btn.setFixedHeight(38)
        select_btn.clicked.connect(self.select_enroll_image)

        camera_btn = QPushButton("摄像头采集录入图")
        camera_btn.setFixedHeight(38)
        camera_btn.clicked.connect(self.capture_enroll_image)

        enroll_image_button_style = """
            QPushButton {
                font-size: 15px;
                background-color: #2563eb;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
        """
        select_btn.setStyleSheet(enroll_image_button_style)
        camera_btn.setStyleSheet(enroll_image_button_style)

        row2.addWidget(self.enroll_path_label, 1)
        row2.addWidget(select_btn)
        row2.addWidget(camera_btn)
        layout.addLayout(row2)

        self.enroll_preview = QLabel("未选择录入图片")
        self.enroll_preview.setAlignment(Qt.AlignCenter)
        self.enroll_preview.setFixedHeight(150)
        self.enroll_preview.setStyleSheet(
            """
            QLabel {
                border: 1px solid #bbb;
                border-radius: 8px;
                background: #f8fafc;
                color: #64748b;
                font-size: 14px;
            }
            """
        )
        layout.addWidget(self.enroll_preview)

        self.enroll_btn = QPushButton("确认录入")
        self.enroll_btn.setFixedHeight(46)
        self.enroll_btn.clicked.connect(self.call_enroll)
        self.enroll_btn.setStyleSheet(
            """
            QPushButton {
                font-size: 17px;
                font-weight: bold;
                background-color: #16a34a;
                color: white;
                border-radius: 9px;
            }
            QPushButton:hover {
                background-color: #15803d;
            }
            QPushButton:disabled {
                background-color: #aaa;
            }
            """
        )
        layout.addWidget(self.enroll_btn)

        self.enroll_result = QLabel("录入结果：未录入")
        self.enroll_result.setAlignment(Qt.AlignCenter)
        self.enroll_result.setFixedHeight(54)
        self.enroll_result.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        self.enroll_result.setStyleSheet(
            """
            QLabel {
                border-radius: 8px;
                background-color: #f1f5f9;
                color: #334155;
            }
            """
        )
        layout.addWidget(self.enroll_result)

        note = QLabel(
            "说明：录入不会重新训练模型。系统会先把该图片提取出的 ArcFace embedding 写入 clean gallery，"
            "然后自动调用注册 embedding 水印方法生成 protected gallery 和 watermark key；"
            "录入图片备份统一保存在 api_logs/enrolled_faces/images。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #64748b;")
        layout.addWidget(note)

        return group


    def build_delete_group(self):
        group = QGroupBox("删除用户")
        group.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))

        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        row = QHBoxLayout()
        row.setSpacing(10)

        label_text = QLabel("删除身份编号：")
        label_text.setFont(QFont("Microsoft YaHei", 11))

        self.delete_label_input = QLineEdit()
        self.delete_label_input.setPlaceholderText("例如：201 或 user_001")
        self.delete_label_input.setFixedHeight(36)

        delete_info = QLabel("删除策略：从 clean 删除 → 自动同步 protected + watermark key")
        delete_info.setFont(QFont("Microsoft YaHei", 10))
        delete_info.setStyleSheet("color: #991b1b;")

        self.delete_btn = QPushButton("确认删除")
        self.delete_btn.setFixedHeight(38)
        self.delete_btn.clicked.connect(self.call_delete)
        self.delete_btn.setStyleSheet(
            """
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                background-color: #dc2626;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #b91c1c;
            }
            QPushButton:disabled {
                background-color: #aaa;
            }
            """
        )

        row.addWidget(label_text)
        row.addWidget(self.delete_label_input, 2)
        row.addWidget(delete_info, 2)
        row.addWidget(self.delete_btn)
        layout.addLayout(row)

        self.delete_result = QLabel("删除结果：未删除")
        self.delete_result.setAlignment(Qt.AlignCenter)
        self.delete_result.setFixedHeight(54)
        self.delete_result.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        self.delete_result.setStyleSheet(
            """
            QLabel {
                border-radius: 8px;
                background-color: #f1f5f9;
                color: #334155;
            }
            """
        )
        layout.addWidget(self.delete_result)

        note = QLabel(
            "说明：删除会从 clean gallery 中移除对应 label 和 embedding，然后自动重新生成 protected gallery 和 watermark key；不会删除模型权重、历史日志或图片备份。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #64748b;")
        layout.addWidget(note)

        return group

    def build_log_group(self):
        group = QGroupBox("授权记录")
        group.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))

        layout = QHBoxLayout(group)
        layout.setSpacing(14)

        view_btn = QPushButton("查看最近录入记录")
        view_btn.setFixedHeight(42)
        view_btn.clicked.connect(self.on_view_clicked)
        view_btn.setStyleSheet(
            """
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                background-color: #0f766e;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #115e59;
            }
            """
        )

        delete_log_btn = QPushButton("查看最近删除记录")
        delete_log_btn.setFixedHeight(42)
        delete_log_btn.clicked.connect(self.on_view_delete_clicked)
        delete_log_btn.setStyleSheet(
            """
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                background-color: #991b1b;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #7f1d1d;
            }
            """
        )

        layout.addWidget(view_btn)
        layout.addWidget(delete_log_btn)
        return group

    def set_config_enabled(self, enabled):
        self.refresh_config_btn.setEnabled(enabled)
        self.apply_config_btn.setEnabled(enabled)
        self.recognition_mode_combo.setEnabled(enabled)
        self.score_mode_combo.setEnabled(enabled)

    def set_runtime_config_label(self, text, positive=True):
        self.runtime_config_result.setText(text)
        if positive:
            self.runtime_config_result.setStyleSheet(
                """
                QLabel {
                    border-radius: 8px;
                    background-color: #dcfce7;
                    color: #166534;
                }
                """
            )
        else:
            self.runtime_config_result.setStyleSheet(
                """
                QLabel {
                    border-radius: 8px;
                    background-color: #fee2e2;
                    color: #991b1b;
                }
                """
            )

    def call_load_runtime_config(self):
        if not self.admin_token:
            QMessageBox.warning(self, "缺少管理员权限", "管理员 token 不存在，请关闭授权页面并重新验证管理员")
            return

        self.set_config_enabled(False)
        self.runtime_config_result.setText("正在读取当前模式...")

        self.config_worker = RuntimeConfigWorker(admin_token=self.admin_token)
        self.config_worker.finished.connect(self.on_runtime_config_finished)
        self.config_worker.failed.connect(self.on_runtime_config_failed)
        self.config_worker.start()

    def call_update_runtime_config(self):
        if not self.admin_token:
            QMessageBox.warning(self, "缺少管理员权限", "管理员 token 不存在，请关闭授权页面并重新验证管理员")
            return

        recognition_mode = self.recognition_mode_combo.currentText().strip()
        score_mode = self.score_mode_combo.currentText().strip()

        msg = (
            "即将修改系统运行模式：\n\n"
            f"人脸识别图库：{recognition_mode}\n"
            f"API 分数图库：{score_mode}\n\n"
            "说明：主界面身份识别会使用“人脸识别图库”；/score 和 /score_batch 会使用“API 分数图库”。\n"
            "该操作只有管理员授权页面可以执行。\n\n"
            "是否保存？"
        )
        reply = QMessageBox.question(
            self,
            "确认修改系统模式",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        self.set_config_enabled(False)
        self.runtime_config_result.setText("正在保存系统模式...")

        self.config_worker = RuntimeConfigWorker(
            admin_token=self.admin_token,
            recognition_gallery_mode=recognition_mode,
            score_gallery_mode=score_mode,
        )
        self.config_worker.finished.connect(self.on_runtime_config_finished)
        self.config_worker.failed.connect(self.on_runtime_config_failed)
        self.config_worker.start()

    def on_runtime_config_finished(self, result):
        self.set_config_enabled(True)

        if not result.get("success", False):
            error = result.get("error", "未知错误")
            self.set_runtime_config_label(f"模式设置失败：{error}", positive=False)
            QMessageBox.warning(self, "模式设置失败", error)
            return

        cfg = result.get("config", {})
        recognition_mode = cfg.get("recognition_gallery_mode", "protected")
        score_mode = cfg.get("score_gallery_mode", "protected")
        updated_at = cfg.get("updated_at", "")
        updated_by = cfg.get("updated_by_admin", "")
        changed = bool(result.get("changed", False))

        idx = self.recognition_mode_combo.findText(recognition_mode)
        if idx >= 0:
            self.recognition_mode_combo.setCurrentIndex(idx)

        idx = self.score_mode_combo.findText(score_mode)
        if idx >= 0:
            self.score_mode_combo.setCurrentIndex(idx)

        text = f"当前模式：识别={recognition_mode}，API分数={score_mode}"
        if updated_at:
            text += f"；更新时间={updated_at}"
        if updated_by:
            text += f"；管理员={updated_by}"
        self.set_runtime_config_label(text, positive=True)

        if changed:
            QMessageBox.information(self, "模式已保存", text)

    def on_runtime_config_failed(self, error):
        self.set_config_enabled(True)
        self.set_runtime_config_label(f"模式设置失败：{error}", positive=False)
        QMessageBox.warning(self, "模式设置请求失败", error)


    def set_enroll_image(self, path, source="录入图片"):
        if not path:
            return

        self.enroll_image_path = path
        self.enroll_path_label.setText(f"{source}：{path}")

        pix = QPixmap(path)
        if not pix.isNull():
            pix = pix.scaled(260, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.enroll_preview.setPixmap(pix)
            self.enroll_preview.setText("")
        else:
            self.enroll_preview.setText("图片预览失败")

        self.set_enroll_neutral("录入结果：未录入")

    def select_enroll_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择要录入的人脸图片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )

        if not path:
            return

        self.set_enroll_image(path, source="录入图片")

    def capture_enroll_image(self):
        path = capture_image_from_camera(self, title="摄像头采集录入图")
        if not path:
            return

        self.temp_camera_paths.append(path)
        self.set_enroll_image(path, source="摄像头录入图片")

    def set_enroll_enabled(self, enabled):
        self.enroll_btn.setEnabled(enabled)

    def call_enroll(self):
        label = self.label_input.text().strip()
        overwrite = self.overwrite_check.isChecked()

        if not label:
            QMessageBox.warning(self, "缺少身份编号", "请输入要录入的身份编号 label")
            return

        if not self.enroll_image_path or not os.path.exists(self.enroll_image_path):
            QMessageBox.warning(self, "缺少录入图片", "请先选择要录入的人脸图片")
            return

        if not self.admin_token:
            QMessageBox.warning(self, "缺少管理员权限", "管理员 token 不存在，请关闭授权页面并重新验证管理员")
            return

        msg = (
            f"即将录入身份：{label}\n"
            "录入流程：先写入 clean gallery，再自动生成 protected gallery 和 watermark key\n"
            f"是否覆盖已有身份：{'是' if overwrite else '否'}\n\n"
            "是否继续？"
        )
        reply = QMessageBox.question(
            self,
            "确认录入",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply != QMessageBox.Yes:
            return

        self.set_enroll_enabled(False)
        self.set_enroll_neutral("录入中，请稍候...")

        self.enroll_worker = EnrollWorker(
            image_path=self.enroll_image_path,
            label=label,
            overwrite=overwrite,
            admin_token=self.admin_token,
        )
        self.enroll_worker.finished.connect(self.on_enroll_finished)
        self.enroll_worker.failed.connect(self.on_enroll_failed)
        self.enroll_worker.start()

    def on_enroll_finished(self, result):
        self.set_enroll_enabled(True)

        if not result.get("success", False):
            error = result.get("error", "未知错误")
            self.set_enroll_negative(f"录入失败：{error}")
            QMessageBox.warning(self, "录入失败", error)
            return

        action = result.get("action", "insert")
        label = result.get("label", "未知")
        label_index = result.get("label_index", "未知")
        gallery_mode = result.get("gallery_mode", "clean")
        protected_gallery_path = result.get("protected_gallery_path", "")
        watermark_key_path = result.get("watermark_key_path", "")
        num_identities = result.get("num_identities", "未知")
        backup_path = result.get("backup_path", "")
        warning = result.get("warning", "")

        action_cn = "新增" if action == "insert" else "覆盖更新"
        text = (
            f"录入成功：{action_cn}身份 {label}\n"
            f"label_index={label_index}，clean 身份数={num_identities}\n"
            "已自动生成 protected gallery 和 watermark key"
        )
        self.set_enroll_positive(text)

        detail = text
        if backup_path:
            detail += f"\n\n已自动备份原 clean gallery：\n{backup_path}"
        if protected_gallery_path:
            detail += f"\n\nprotected gallery：\n{protected_gallery_path}"
        if watermark_key_path:
            detail += f"\n\nwatermark key：\n{watermark_key_path}"
        if warning:
            detail += f"\n\n提示：{warning}"

        QMessageBox.information(self, "录入成功", detail)

    def on_enroll_failed(self, error):
        self.set_enroll_enabled(True)
        self.set_enroll_negative(f"录入失败：{error}")
        QMessageBox.warning(self, "录入请求失败", error)


    def set_delete_enabled(self, enabled):
        self.delete_btn.setEnabled(enabled)

    def call_delete(self):
        label = self.delete_label_input.text().strip()

        if not label:
            QMessageBox.warning(self, "缺少身份编号", "请输入要删除的身份编号 label")
            return

        if not self.admin_token:
            QMessageBox.warning(self, "缺少管理员权限", "管理员 token 不存在，请关闭授权页面并重新验证管理员")
            return

        msg = (
            f"即将删除身份：{label}\n"
            "删除流程：从 clean gallery 删除，再自动生成 protected gallery 和 watermark key。\n\n"
            "删除后，该身份将不再参与 clean/protected 的识别。\n"
            "系统会先自动备份原 clean gallery 文件。\n\n"
            "是否确认删除？"
        )
        reply = QMessageBox.question(
            self,
            "确认删除",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        self.set_delete_enabled(False)
        self.set_delete_neutral("删除中，请稍候...")

        self.delete_worker = DeleteWorker(
            label=label,
            admin_token=self.admin_token,
        )
        self.delete_worker.finished.connect(self.on_delete_finished)
        self.delete_worker.failed.connect(self.on_delete_failed)
        self.delete_worker.start()

    def on_delete_finished(self, result):
        self.set_delete_enabled(True)

        if not result.get("success", False):
            error = result.get("error", "未知错误")
            self.set_delete_negative(f"删除失败：{error}")
            QMessageBox.warning(self, "删除失败", error)
            return

        label = result.get("label", "未知")
        deleted_index = result.get("deleted_index", "未知")
        gallery_mode = result.get("gallery_mode", "clean")
        protected_gallery_path = result.get("protected_gallery_path", "")
        watermark_key_path = result.get("watermark_key_path", "")
        num_before = result.get("num_identities_before", "未知")
        num_after = result.get("num_identities_after", "未知")
        backup_path = result.get("backup_path", "")

        text = (
            f"删除成功：身份 {label}\n"
            f"clean 删除前身份数={num_before}，删除后身份数={num_after}\n"
            "已自动同步 protected gallery 和 watermark key"
        )
        self.set_delete_positive(text)

        detail = text
        if backup_path:
            detail += f"\n\n已自动备份原 clean gallery：\n{backup_path}"
        if protected_gallery_path:
            detail += f"\n\nprotected gallery：\n{protected_gallery_path}"
        if watermark_key_path:
            detail += f"\n\nwatermark key：\n{watermark_key_path}"

        QMessageBox.information(self, "删除成功", detail)

    def on_delete_failed(self, error):
        self.set_delete_enabled(True)
        self.set_delete_negative(f"删除失败：{error}")
        QMessageBox.warning(self, "删除请求失败", error)

    def set_delete_positive(self, text):
        self.delete_result.setText(text)
        self.delete_result.setStyleSheet(
            """
            QLabel {
                border-radius: 8px;
                background-color: #dcfce7;
                color: #166534;
            }
            """
        )

    def set_delete_negative(self, text):
        self.delete_result.setText(text)
        self.delete_result.setStyleSheet(
            """
            QLabel {
                border-radius: 8px;
                background-color: #fee2e2;
                color: #991b1b;
            }
            """
        )

    def set_delete_neutral(self, text):
        self.delete_result.setText(text)
        self.delete_result.setStyleSheet(
            """
            QLabel {
                border-radius: 8px;
                background-color: #f1f5f9;
                color: #334155;
            }
            """
        )

    def on_view_clicked(self):
        try:
            r = requests.get(API_BASE + "/logs/enroll_face", timeout=10)
            data = r.json()
        except Exception as e:
            QMessageBox.warning(self, "读取失败", str(e))
            return

        if not data.get("success", False):
            QMessageBox.warning(self, "读取失败", data.get("error", "未知错误"))
            return

        logs = data.get("logs", [])[-10:]
        if not logs:
            QMessageBox.information(self, "录入记录", "暂无录入记录")
            return

        lines = []
        for row in logs:
            lines.append(
                f"{row.get('time', '')} | {row.get('action', '')} | "
                f"label={row.get('label', '')} | gallery={row.get('gallery_mode', '')} | "
                f"index={row.get('label_index', '')}"
            )

        QMessageBox.information(
            self,
            "最近录入记录",
            "\n".join(lines),
        )


    def on_view_delete_clicked(self):
        try:
            r = requests.get(API_BASE + "/logs/delete_face", timeout=10)
            data = r.json()
        except Exception as e:
            QMessageBox.warning(self, "读取失败", str(e))
            return

        if not data.get("success", False):
            QMessageBox.warning(self, "读取失败", data.get("error", "未知错误"))
            return

        logs = data.get("logs", [])[-10:]
        if not logs:
            QMessageBox.information(self, "删除记录", "暂无删除记录")
            return

        lines = []
        for row in logs:
            lines.append(
                f"{row.get('time', '')} | {row.get('action', '')} | "
                f"label={row.get('label', '')} | gallery={row.get('gallery_mode', '')} | "
                f"index={row.get('deleted_index', '')} | "
                f"{row.get('num_identities_before', '')}->{row.get('num_identities_after', '')}"
            )

        QMessageBox.information(
            self,
            "最近删除记录",
            "\n".join(lines),
        )

    def set_enroll_positive(self, text):
        self.enroll_result.setText(text)
        self.enroll_result.setStyleSheet(
            """
            QLabel {
                border-radius: 8px;
                background-color: #dcfce7;
                color: #166534;
            }
            """
        )

    def set_enroll_negative(self, text):
        self.enroll_result.setText(text)
        self.enroll_result.setStyleSheet(
            """
            QLabel {
                border-radius: 8px;
                background-color: #fee2e2;
                color: #991b1b;
            }
            """
        )

    def set_enroll_neutral(self, text):
        self.enroll_result.setText(text)
        self.enroll_result.setStyleSheet(
            """
            QLabel {
                border-radius: 8px;
                background-color: #f1f5f9;
                color: #334155;
            }
            """
        )
