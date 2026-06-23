import json
import os
import sys
import requests

from PyQt5.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QLineEdit,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QMessageBox,
)


DEFAULT_API_BASE = "http://127.0.0.1:8000"


def pretty(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2)


class ApiWorker(QThread):
    finished = Signal(str, str)
    failed = Signal(str, str)

    def __init__(self, target_box, method, api_base, endpoint, image_path=None, data=None):
        super().__init__()
        self.target_box = target_box
        self.method = method
        self.api_base = api_base.rstrip("/")
        self.endpoint = endpoint
        self.image_path = image_path
        self.data = data or {}

    def run(self):
        try:
            url = self.api_base + self.endpoint

            if self.method == "GET":
                r = requests.get(url, timeout=30)
                result = r.json()
                self.finished.emit(self.target_box, pretty(result))
                return

            if self.method == "POST_IMAGE":
                if not self.image_path or not os.path.exists(self.image_path):
                    raise FileNotFoundError("请先选择有效图片")

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
                        data=self.data,
                        timeout=180,
                    )

                try:
                    result = r.json()
                except Exception:
                    result = {
                        "status_code": r.status_code,
                        "text": r.text,
                    }

                self.finished.emit(self.target_box, pretty(result))
                return

            raise ValueError(f"未知请求方法: {self.method}")

        except Exception as e:
            self.failed.emit(self.target_box, str(e))


class ArcFaceQtGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ArcFace 水印确权实验系统")
        self.resize(1200, 800)

        self.image_path = None
        self.worker = None

        self.build_ui()

    def build_ui(self):
        main_layout = QVBoxLayout(self)

        title = QLabel("ArcFace 水印确权实验系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title)

        main_layout.addWidget(self.build_config_group())
        main_layout.addWidget(self.build_image_group())
        main_layout.addWidget(self.build_button_group())
        main_layout.addWidget(self.build_output_group())

    def build_config_group(self):
        group = QGroupBox("基础设置")
        layout = QGridLayout(group)

        self.api_base_edit = QLineEdit(DEFAULT_API_BASE)

        self.gallery_mode_combo = QComboBox()
        self.gallery_mode_combo.addItems(["clean", "prior", "protected"])

        self.target_label_edit = QLineEdit("1")

        self.api_threshold_spin = QDoubleSpinBox()
        self.api_threshold_spin.setRange(-1.0, 1.0)
        self.api_threshold_spin.setSingleStep(0.01)
        self.api_threshold_spin.setValue(0.30)
        self.api_threshold_spin.setDecimals(4)

        self.wm_threshold_spin = QDoubleSpinBox()
        self.wm_threshold_spin.setRange(-1.0, 1.0)
        self.wm_threshold_spin.setSingleStep(0.001)
        self.wm_threshold_spin.setValue(0.085)
        self.wm_threshold_spin.setDecimals(4)

        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 50)
        self.topk_spin.setValue(5)

        self.save_request_check = QCheckBox("保存 /score 请求图片到日志目录")
        self.save_request_check.setChecked(True)

        layout.addWidget(QLabel("后端 API 地址"), 0, 0)
        layout.addWidget(self.api_base_edit, 0, 1, 1, 3)

        layout.addWidget(QLabel("Gallery 模式"), 1, 0)
        layout.addWidget(self.gallery_mode_combo, 1, 1)

        layout.addWidget(QLabel("目标身份 label"), 1, 2)
        layout.addWidget(self.target_label_edit, 1, 3)

        layout.addWidget(QLabel("API 验证阈值"), 2, 0)
        layout.addWidget(self.api_threshold_spin, 2, 1)

        layout.addWidget(QLabel("水印检测阈值"), 2, 2)
        layout.addWidget(self.wm_threshold_spin, 2, 3)

        layout.addWidget(QLabel("Top-K"), 3, 0)
        layout.addWidget(self.topk_spin, 3, 1)

        layout.addWidget(self.save_request_check, 3, 2, 1, 2)

        return group

    def build_image_group(self):
        group = QGroupBox("输入图片")
        layout = QHBoxLayout(group)

        left_layout = QVBoxLayout()

        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)

        select_btn = QPushButton("选择图片")
        select_btn.clicked.connect(self.select_image)

        left_layout.addWidget(QLabel("图片路径"))
        left_layout.addWidget(self.image_path_edit)
        left_layout.addWidget(select_btn)
        left_layout.addStretch()

        self.preview_label = QLabel("未选择图片")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(260, 260)
        self.preview_label.setStyleSheet(
            "border: 1px solid #aaa; background: #f5f5f5;"
        )

        layout.addLayout(left_layout, 2)
        layout.addWidget(self.preview_label, 1)

        return group

    def build_button_group(self):
        group = QGroupBox("功能操作")
        layout = QHBoxLayout(group)

        self.health_btn = QPushButton("检查后端状态")
        self.predict_btn = QPushButton("普通识别 /predict")
        self.score_btn = QPushButton("黑盒分数查询 /score")
        self.detect_btn = QPushButton("水印检测 /detect_watermark")
        self.both_btn = QPushButton("识别 + 水印检测 /both")
        self.logs_btn = QPushButton("查看 score 日志")

        self.health_btn.clicked.connect(self.call_health)
        self.predict_btn.clicked.connect(self.call_predict)
        self.score_btn.clicked.connect(self.call_score)
        self.detect_btn.clicked.connect(self.call_detect)
        self.both_btn.clicked.connect(self.call_both)
        self.logs_btn.clicked.connect(self.call_logs)

        layout.addWidget(self.health_btn)
        layout.addWidget(self.predict_btn)
        layout.addWidget(self.score_btn)
        layout.addWidget(self.detect_btn)
        layout.addWidget(self.both_btn)
        layout.addWidget(self.logs_btn)

        return group

    def build_output_group(self):
        group = QGroupBox("输出结果")
        layout = QGridLayout(group)

        self.predict_output = QTextEdit()
        self.score_output = QTextEdit()
        self.wm_output = QTextEdit()
        self.log_output = QTextEdit()

        for box in [
            self.predict_output,
            self.score_output,
            self.wm_output,
            self.log_output,
        ]:
            box.setReadOnly(True)
            box.setStyleSheet(
                "font-family: Consolas, Microsoft YaHei; font-size: 12px;"
            )

        layout.addWidget(QLabel("普通识别结果"), 0, 0)
        layout.addWidget(QLabel("黑盒 score 查询结果"), 0, 1)
        layout.addWidget(self.predict_output, 1, 0)
        layout.addWidget(self.score_output, 1, 1)

        layout.addWidget(QLabel("水印检测结果"), 2, 0)
        layout.addWidget(QLabel("后端状态 / 日志"), 2, 1)
        layout.addWidget(self.wm_output, 3, 0)
        layout.addWidget(self.log_output, 3, 1)

        return group

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )

        if not path:
            return

        self.image_path = path
        self.image_path_edit.setText(path)

        pix = QPixmap(path)
        if not pix.isNull():
            pix = pix.scaled(
                self.preview_label.width(),
                self.preview_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.preview_label.setPixmap(pix)
        else:
            self.preview_label.setText("图片预览失败")

    def api_base(self):
        return self.api_base_edit.text().strip()

    def gallery_mode(self):
        return self.gallery_mode_combo.currentText()

    def target_label(self):
        return self.target_label_edit.text().strip()

    def api_threshold(self):
        return self.api_threshold_spin.value()

    def wm_threshold(self):
        return self.wm_threshold_spin.value()

    def topk(self):
        return self.topk_spin.value()

    def set_buttons_enabled(self, enabled):
        for btn in [
            self.health_btn,
            self.predict_btn,
            self.score_btn,
            self.detect_btn,
            self.both_btn,
            self.logs_btn,
        ]:
            btn.setEnabled(enabled)

    def start_worker(self, target_box, method, endpoint, image_path=None, data=None):
        self.set_buttons_enabled(False)

        self.worker = ApiWorker(
            target_box=target_box,
            method=method,
            api_base=self.api_base(),
            endpoint=endpoint,
            image_path=image_path,
            data=data,
        )

        self.worker.finished.connect(self.on_worker_finished)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.finished.connect(lambda *_: self.set_buttons_enabled(True))
        self.worker.failed.connect(lambda *_: self.set_buttons_enabled(True))

        self.worker.start()

    def get_output_box(self, target_box):
        mapping = {
            "predict": self.predict_output,
            "score": self.score_output,
            "wm": self.wm_output,
            "log": self.log_output,
            "both": self.predict_output,
        }
        return mapping[target_box]

    def on_worker_finished(self, target_box, text):
        if target_box == "both":
            try:
                obj = json.loads(text)
                self.predict_output.setPlainText(pretty(obj.get("predict", obj)))
                self.wm_output.setPlainText(pretty(obj.get("watermark", obj)))
            except Exception:
                self.predict_output.setPlainText(text)
            return

        self.get_output_box(target_box).setPlainText(text)

    def on_worker_failed(self, target_box, error):
        self.get_output_box(target_box).setPlainText(f"请求失败:\n{error}")
        QMessageBox.warning(self, "请求失败", error)

    def require_image(self):
        if not self.image_path or not os.path.exists(self.image_path):
            QMessageBox.warning(self, "缺少图片", "请先选择图片")
            return False
        return True

    def call_health(self):
        self.start_worker(
            target_box="log",
            method="GET",
            endpoint="/health",
        )

    def call_logs(self):
        self.start_worker(
            target_box="log",
            method="GET",
            endpoint="/logs/score",
        )

    def call_predict(self):
        if not self.require_image():
            return

        data = {
            "gallery_mode": self.gallery_mode(),
            "topk": str(self.topk()),
        }

        self.start_worker(
            target_box="predict",
            method="POST_IMAGE",
            endpoint="/predict",
            image_path=self.image_path,
            data=data,
        )

    def call_score(self):
        if not self.require_image():
            return

        if not self.target_label():
            QMessageBox.warning(self, "缺少 label", "黑盒 score 查询需要目标身份 label")
            return

        data = {
            "target_label": self.target_label(),
            "gallery_mode": self.gallery_mode(),
            "api_threshold": str(self.api_threshold()),
            "save_request_image": str(self.save_request_check.isChecked()).lower(),
        }

        self.start_worker(
            target_box="score",
            method="POST_IMAGE",
            endpoint="/score",
            image_path=self.image_path,
            data=data,
        )

    def call_detect(self):
        if not self.require_image():
            return

        if not self.target_label():
            QMessageBox.warning(self, "缺少 label", "水印检测需要目标身份 label")
            return

        data = {
            "label": self.target_label(),
            "threshold": str(self.wm_threshold()),
        }

        self.start_worker(
            target_box="wm",
            method="POST_IMAGE",
            endpoint="/detect_watermark",
            image_path=self.image_path,
            data=data,
        )

    def call_both(self):
        if not self.require_image():
            return

        data = {
            "gallery_mode": self.gallery_mode(),
            "label": self.target_label(),
            "threshold": str(self.wm_threshold()),
            "topk": str(self.topk()),
        }

        self.start_worker(
            target_box="both",
            method="POST_IMAGE",
            endpoint="/both",
            image_path=self.image_path,
            data=data,
        )


def main():
    app = QApplication(sys.argv)
    window = ArcFaceQtGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()