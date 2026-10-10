from __future__ import annotations

import sys
import traceback
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------
# Configuration: update these paths only if your project is elsewhere.
# ---------------------------------------------------------------------
PROJECT_DIR = Path(r"C:\Projects\adInsightPrediction")
MODEL_BUNDLE_PATH = (
    PROJECT_DIR
    / "outputs"
    / "dinov2_large"
    / "six_attribute_ridge_models.joblib"
)

SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEOS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
TARGET_LABELS = {
    "innovation": "Innovation",
    "premium": "Premium",
    "minimalism": "Minimalism",
    "excitement": "Excitement",
    "social_orientation": "Social Orientation",
    "aspiration": "Aspiration",
}


class PredictionService:
    """Loads the saved Ridge bundle and matching frozen DINOv2 encoder once."""

    def __init__(self):
        self.bundle = None
        self.processor = None
        self.encoder = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.targets = None
        self.frame_count = 4
        self.feature_dim = 1024

    def load(self):
        if self.bundle is not None:
            return

        if not MODEL_BUNDLE_PATH.is_file():
            raise FileNotFoundError(
                "Trained model bundle was not found:\n"
                f"{MODEL_BUNDLE_PATH}\n\n"
                "Make sure you ran the notebook through the model-saving cell."
            )

        self.bundle = joblib.load(MODEL_BUNDLE_PATH)
        model_name = self.bundle.get("model_name", "facebook/dinov2-large")
        self.targets = self.bundle.get(
            "targets",
            list(TARGET_LABELS.keys())
        )
        self.frame_count = int(self.bundle.get("frame_count", 4))
        self.feature_dim = int(self.bundle.get("feature_dim", 1024))

        if self.feature_dim != 1024:
            raise ValueError(
                f"This app expects DINOv2 ViT-L/14 features (1024 dimensions), "
                f"but the bundle says {self.feature_dim}."
            )

        expected_targets = list(TARGET_LABELS.keys())
        if list(self.targets) != expected_targets:
            raise ValueError(
                "The target order/names in the saved bundle do not match the "
                "six attributes expected by this app.\n"
                f"Bundle targets: {self.targets}"
            )

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()

    @torch.inference_mode()
    def extract_frame_feature(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        if self.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.encoder(**inputs)
        else:
            outputs = self.encoder(**inputs)

        feature = outputs.last_hidden_state[:, 0, :]
        return (
            feature.squeeze(0)
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def load_video_frames(self, video_path: Path) -> list[Image.Image]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video:\n{video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"No readable frames found in video:\n{video_path}")

        indices = np.linspace(
            0,
            total_frames - 1,
            num=min(self.frame_count, total_frames),
            dtype=int,
        )

        frames = []
        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))

        cap.release()

        if not frames:
            raise ValueError(f"Could not decode frames from:\n{video_path}")

        return frames

    def extract_asset_feature(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()

        if suffix in SUPPORTED_IMAGES:
            with Image.open(path) as image:
                feature = self.extract_frame_feature(image)

        elif suffix in SUPPORTED_VIDEOS:
            frames = self.load_video_frames(path)
            frame_features = [
                self.extract_frame_feature(frame) for frame in frames
            ]
            feature = np.mean(
                np.stack(frame_features, axis=0),
                axis=0,
            ).astype(np.float32)

        else:
            raise ValueError(
                "Unsupported file type. Please select an image "
                "or a supported video."
            )

        if feature.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected a {self.feature_dim}-dimensional feature vector, "
                f"got {feature.shape}."
            )

        return feature

    def predict(self, file_path: str) -> dict[str, int]:
        self.load()
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found:\n{path}")

        feature = self.extract_asset_feature(path).reshape(1, -1)
        scores = {}

        for target in self.targets:
            raw_score = float(
                self.bundle["models"][target].predict(feature)[0]
            )
            # Convert model's 0–1 output to an inclusive 1–100 display score.
            score_1_to_100 = int(np.clip(np.rint(raw_score * 99 + 1), 1, 100))
            scores[target] = score_1_to_100

        return scores


class PredictionWorker(QThread):
    finished_prediction = pyqtSignal(str, object)
    failed = pyqtSignal(str)

    def __init__(self, service: PredictionService, file_path: str):
        super().__init__()
        self.service = service
        self.file_path = file_path

    def run(self):
        try:
            result = self.service.predict(self.file_path)
            self.finished_prediction.emit(self.file_path, result)
        except Exception as exc:
            details = f"{exc}\n\n{traceback.format_exc()}"
            self.failed.emit(details)


class ScoreCard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("scoreCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")

        self.score_label = QLabel("—")
        self.score_label.setObjectName("scoreValue")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.progress = QProgressBar()
        self.progress.setRange(1, 100)
        self.progress.setValue(1)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(9)

        self.description = QLabel("Waiting for an advertisement")
        self.description.setObjectName("mutedText")

        layout.addWidget(self.title_label)
        layout.addWidget(self.score_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.description)

    def set_score(self, score: int):
        self.score_label.setText(f"{score}/100")
        self.progress.setValue(score)
        self.description.setText(f"Predicted attribute score: {score} out of 100")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ad Insight — Advertisement Analyzer")
        self.resize(1000, 760)
        self.setMinimumSize(780, 650)

        self.service = PredictionService()
        self.worker = None
        self.selected_path = None
        self.cards = {}

        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)

        header = QVBoxLayout()
        title = QLabel("Ad Insight")
        title.setObjectName("appTitle")
        subtitle = QLabel(
            "Analyze an advertisement's visual personality across six attributes."
        )
        subtitle.setObjectName("mutedText")
        header.addWidget(title)
        header.addWidget(subtitle)
        root.addLayout(header)

        file_panel = QFrame()
        file_panel.setObjectName("panel")
        file_layout = QHBoxLayout(file_panel)
        file_layout.setContentsMargins(16, 14, 16, 14)
        file_layout.setSpacing(12)

        self.preview = QLabel("Choose an image or video to get started")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(240, 150)
        self.preview.setMaximumWidth(330)
        self.preview.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        self.file_label = QLabel("No advertisement selected")
        self.file_label.setObjectName("fileLabel")
        self.file_label.setWordWrap(True)

        self.browse_button = QPushButton("Choose Image / Video")
        self.browse_button.clicked.connect(self.choose_file)

        self.analyze_button = QPushButton("Analyze Advertisement")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.analyze)

        self.status_label = QLabel(
            "Model loads when you run the first analysis."
        )
        self.status_label.setObjectName("mutedText")
        self.status_label.setWordWrap(True)

        right_panel.addWidget(self.file_label)
        right_panel.addWidget(self.browse_button)
        right_panel.addWidget(self.analyze_button)
        right_panel.addWidget(self.status_label)
        right_panel.addStretch(1)

        file_layout.addWidget(self.preview, 2)
        file_layout.addLayout(right_panel, 3)
        root.addWidget(file_panel)

        scores_heading = QLabel("Attribute Scores")
        scores_heading.setObjectName("sectionTitle")
        root.addWidget(scores_heading)

        grid = QGridLayout()
        grid.setSpacing(14)

        display_names = [
            ("innovation", "Innovation"),
            ("premium", "Premium"),
            ("minimalism", "Minimalism"),
            ("excitement", "Excitement"),
            ("social_orientation", "Social Orientation"),
            ("aspiration", "Aspiration"),
        ]

        for index, (key, label) in enumerate(display_names):
            card = ScoreCard(label)
            self.cards[key] = card
            grid.addWidget(card, index // 3, index % 3)

        root.addLayout(grid)

        note = QLabel(
            "Scores are model predictions scaled to 1–100. They represent "
            "the six learned visual attributes, not probabilities or a "
            "combined advertisement-quality score."
        )
        note.setObjectName("footnote")
        note.setWordWrap(True)
        root.addWidget(note)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #f5f7fb;
                color: #182230;
                font-family: "Segoe UI";
                font-size: 10pt;
            }
            QLabel#appTitle {
                font-size: 26pt;
                font-weight: 700;
                color: #172b4d;
            }
            QLabel#sectionTitle {
                font-size: 15pt;
                font-weight: 650;
                color: #172b4d;
            }
            QLabel#mutedText, QLabel#footnote {
                color: #64748b;
            }
            QLabel#fileLabel {
                font-weight: 600;
                color: #334155;
            }
            QFrame#panel, QFrame#scoreCard {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
            }
            QLabel#preview {
                background: #f8fafc;
                color: #64748b;
                border: 1px dashed #cbd5e1;
                border-radius: 10px;
                padding: 8px;
            }
            QLabel#metricTitle {
                font-size: 11pt;
                font-weight: 600;
                color: #475569;
            }
            QLabel#scoreValue {
                font-size: 25pt;
                font-weight: 700;
                color: #1d4ed8;
            }
            QPushButton {
                background: #e8eef8;
                color: #1e3a5f;
                border: none;
                border-radius: 9px;
                padding: 11px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #dbe7f8;
            }
            QPushButton:disabled {
                background: #e2e8f0;
                color: #94a3b8;
            }
            QPushButton#primaryButton {
                background: #2457d6;
                color: white;
            }
            QPushButton#primaryButton:hover {
                background: #1d4ed8;
            }
            QProgressBar {
                background: #e8edf5;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: #3b82f6;
                border-radius: 4px;
            }
        """)

    def choose_file(self):
        filters = (
            "Images and videos (*.jpg *.jpeg *.png *.webp *.bmp "
            "*.mp4 *.mov *.avi *.mkv *.webm);;"
            "Images (*.jpg *.jpeg *.png *.webp *.bmp);;"
            "Videos (*.mp4 *.mov *.avi *.mkv *.webm)"
        )

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an advertisement",
            str(PROJECT_DIR / "data") if PROJECT_DIR.exists() else str(Path.home()),
            filters,
        )

        if not file_path:
            return

        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix not in SUPPORTED_IMAGES | SUPPORTED_VIDEOS:
            QMessageBox.warning(
                self,
                "Unsupported file",
                "Please choose a supported image or video file.",
            )
            return

        self.selected_path = str(path)
        self.file_label.setText(path.name)
        self.file_label.setToolTip(str(path))
        self.analyze_button.setEnabled(True)
        self.status_label.setText("Ready to analyze.")

        if suffix in SUPPORTED_IMAGES:
            pixmap = QPixmap(str(path))
        else:
            pixmap = self.video_preview(path)

        if not pixmap.isNull():
            self.preview.setPixmap(
                pixmap.scaled(
                    self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Preview unavailable")

    def video_preview(self, path: Path) -> QPixmap:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return QPixmap()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = max(0, total_frames // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()

        if not ok:
            return QPixmap()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image.save(str(Path.cwd() / "_ad_insight_preview.jpg"), format="JPEG")
        return QPixmap(str(Path.cwd() / "_ad_insight_preview.jpg"))

    def analyze(self):
        if not self.selected_path:
            return

        self.browse_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.status_label.setText(
            "Analyzing… the first run downloads/loads DINOv2 Large. "
            "This can take a little while."
        )

        self.worker = PredictionWorker(self.service, self.selected_path)
        self.worker.finished_prediction.connect(self.show_results)
        self.worker.failed.connect(self.show_error)
        self.worker.start()

    def show_results(self, file_path: str, scores: dict):
        for key, score in scores.items():
            self.cards[key].set_score(score)

        device_label = "GPU" if self.service.device == "cuda" else "CPU"
        self.status_label.setText(
            f"Analysis complete using DINOv2 ViT-L/14 ({device_label})."
        )
        self.browse_button.setEnabled(True)
        self.analyze_button.setEnabled(True)

    def show_error(self, details: str):
        self.status_label.setText("Analysis failed. See the error details.")
        self.browse_button.setEnabled(True)
        self.analyze_button.setEnabled(bool(self.selected_path))

        QMessageBox.critical(
            self,
            "Could not analyze advertisement",
            details,
        )

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Analysis in progress",
                "Please wait for the current analysis to finish before closing."
            )
            event.ignore()
            return
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Ad Insight")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
