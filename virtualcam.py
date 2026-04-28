import sys
import os
import cv2
import numpy as np
import mss
import time
import random
import pyvirtualcam
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QFrame,
                             QTabWidget, QComboBox, QLineEdit, QGroupBox, QCheckBox, 
                             QFormLayout, QStackedWidget)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex

class AppSettings:
    def __init__(self):
        self.mutex = QMutex()
        self._filter = "Normal"
        self._watermark = ""
        self._flip_x = False
        self._flip_y = False

    def update(self, f=None, w=None, fx=None, fy=None):
        self.mutex.lock()
        if f is not None: self._filter = f
        if w is not None: self._watermark = w
        if fx is not None: self._flip_x = fx
        if fy is not None: self._flip_y = fy
        self.mutex.unlock()

    def get_all(self):
        self.mutex.lock()
        f = self._filter
        w = self._watermark
        fx = self._flip_x
        fy = self._flip_y
        self.mutex.unlock()
        return f, w, fx, fy


class CameraThread(QThread):
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    
    def __init__(self, mode, settings, media_path=None, width=1280, height=720, fps=30):
        super().__init__()
        self.mode = mode
        self.media_path = media_path
        self.width = width
        self.height = height
        self.fps = fps
        self.settings = settings
        self.running = True

    def apply_effects(self, frame, f, w, fx, fy):
        if fx: frame = cv2.flip(frame, 1)
        if fy: frame = cv2.flip(frame, 0)

        if f == "Grayscale":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif f == "Invert Colors":
            frame = cv2.bitwise_not(frame)
        elif f == "Matrix Green":
            matrix = frame.copy()
            matrix[:, :, 0] = 0
            matrix[:, :, 2] = 0
            frame = matrix
        elif f == "Retro Glitch":
            h, w_img, _ = frame.shape
            offset_x = np.random.randint(-15, 15)
            offset_y = np.random.randint(-5, 5)
            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            frame = cv2.warpAffine(frame, M, (w_img, h))
        elif f == "Pixelate (Retro 8-bit)":
            h, w_img = frame.shape[:2]
            pixel_size = 20
            small = cv2.resize(frame, (w_img // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
            frame = cv2.resize(small, (w_img, h), interpolation=cv2.INTER_NEAREST)
        elif f == "Edge Detection":
            edges = cv2.Canny(frame, 100, 200)
            frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif f == "Sepia":
            matrix = np.array([[0.272, 0.534, 0.393],
                               [0.349, 0.686, 0.534],
                               [0.393, 0.769, 0.189]])
            frame = cv2.transform(frame, matrix)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        elif f == "Blur":
            frame = cv2.GaussianBlur(frame, (35, 35), 0)

        if w:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.2
            thick = 3
            (tw, th), _ = cv2.getTextSize(w, font, scale, thick)
            h, w_img, _ = frame.shape
            x = w_img - tw - 20
            y = 40 + th
            
            cv2.putText(frame, w, (x, y), font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
            cv2.putText(frame, w, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
            
        return frame

    def fit_frame_to_canvas(self, frame, target_w, target_h):
        """Resizes the frame to fit within target_w x target_h while maintaining aspect ratio, padding with black."""
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Performance check
        if scale == 1.0 and new_w == target_w and new_h == target_h:
            return frame 
            
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        
        pad_y1 = (target_h - new_h) // 2
        pad_y2 = target_h - new_h - pad_y1
        pad_x1 = (target_w - new_w) // 2
        pad_x2 = target_w - new_w - pad_x1
        
        canvas = cv2.copyMakeBorder(resized, pad_y1, pad_y2, pad_x1, pad_x2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return canvas
        
    def stream_media(self, cam):
        cap = cv2.VideoCapture(self.media_path)
        if not cap.isOpened():
            self.error_signal.emit("Error loading media!")
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        is_static_image = (total_frames <= 1)
        
        ret, first_frame = cap.read()
        if not ret:
            self.error_signal.emit("Could not read any frames from this file.")
            cap.release()
            return

        if is_static_image:
            img = self.fit_frame_to_canvas(first_frame, self.width, self.height)
            cap.release()
            while self.running:
                f, w, fx, fy = self.settings.get_all()
                frame = self.apply_effects(img.copy(), f, w, fx, fy)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam.send(frame)
                cam.sleep_until_next_frame()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while self.running:
                f, w, fx, fy = self.settings.get_all()
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret: break

                frame = self.fit_frame_to_canvas(frame, self.width, self.height)
                frame = self.apply_effects(frame, f, w, fx, fy)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam.send(frame)
                cam.sleep_until_next_frame()
                
            cap.release()

    def stream_folder(self, cam):
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        try:
            images = [os.path.join(self.media_path, f) for f in os.listdir(self.media_path) 
                      if os.path.splitext(f)[1].lower() in valid_exts]
        except Exception as e:
            self.error_signal.emit(f"Could not read folder: {e}")
            return
            
        if not images:
            self.error_signal.emit("No valid images (.jpg, .png, .bmp) found in the selected folder!")
            return
            
        last_switch_time = 0
        current_canvas = None
        
        while self.running:
            # 10s wait logic (or grab first image on 0)
            if time.time() - last_switch_time > 10.0:
                img_path = random.choice(images)
                raw_img = cv2.imread(img_path)
                if raw_img is not None:
                    current_canvas = self.fit_frame_to_canvas(raw_img, self.width, self.height)
                last_switch_time = time.time()
                
            if current_canvas is not None:
                f, w, fx, fy = self.settings.get_all()
                
                # Apply effects and convert ONLY on the canvas size
                frame = self.apply_effects(current_canvas.copy(), f, w, fx, fy)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam.send(frame)
            
            cam.sleep_until_next_frame()

    def stream_screen(self, cam):
        with mss.mss() as sct:
            # We explicitly grab the primary monitor (1 in mss is normally the first or primary)
            monitor = sct.monitors[1]
            while self.running:
                f, w, fx, fy = self.settings.get_all()
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)[:, :, :3]
                
                frame = self.fit_frame_to_canvas(frame, self.width, self.height)
                frame = self.apply_effects(frame, f, w, fx, fy)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                cam.send(frame)
                cam.sleep_until_next_frame()

    def run(self):
        try:
            self.status_signal.emit("Starting Virtual Camera...")
            with pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps) as cam:
                self.status_signal.emit(f"Running (Device: {cam.device}) at {self.width}x{self.height} @ {self.fps}fps")
                
                if self.mode == "media":
                    self.stream_media(cam)
                elif self.mode == "folder":
                    self.stream_folder(cam)
                elif self.mode == "screen":
                    self.stream_screen(cam)
                    
            self.status_signal.emit("Stopped")
        except RuntimeError as e:
            self.error_signal.emit(f"Virtual Camera Error:\n{str(e)}\n\nCRITICAL: Is OBS Studio installed?")
            self.status_signal.emit("Error: Driver Missing")
        except Exception as e:
            self.error_signal.emit(f"Unknown Error: {str(e)}")
            self.status_signal.emit("Error")
            
    def stop(self):
        self.running = False
        self.wait()


class VirtualCamAppFixed(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antigravity Virtual Cam - Premium")
        self.setMinimumSize(950, 750)
        self.media_path = None
        self.camera_thread = None
        self.core_settings = AppSettings()
        
        self.setup_ui()
        self.apply_theme()
        
    def setup_ui(self):
        central_widget = QWidget(self)
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(25)
        
        # --- LEFT PANEL ---
        left_panel = QWidget()
        left_panel.setMinimumWidth(380)
        left_panel.setMaximumWidth(450)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(20)
        
        self.title_label = QLabel("VIRTUAL CAM")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        left_layout.addWidget(self.title_label)
        
        self.status_label = QLabel("Status: Ready to Stream")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Segoe UI", 12))
        left_layout.addWidget(self.status_label)
        
        # --- INPUT SOURCE SELECTION ---
        source_group = QGroupBox("Input Source")
        source_layout = QVBoxLayout(source_group)
        source_layout.setContentsMargins(20, 25, 20, 20)
        source_layout.setSpacing(15)
        
        self.combo_source = QComboBox()
        self.combo_source.setMinimumHeight(40)
        self.combo_source.addItems([
            "Local File (Video/Image)", 
            "Local Folder (Random Slideshow)", 
            "Screen Share"
        ])
        self.combo_source.currentIndexChanged.connect(self.on_mode_change)
        source_layout.addWidget(self.combo_source)
        
        self.stack_source = QStackedWidget()
        self.stack_source.setMinimumHeight(100)
        self.stack_source.setMaximumHeight(100)
        
        # Page 1: Media File
        page_media = QWidget()
        media_layout = QVBoxLayout(page_media)
        media_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_select = QPushButton("Select File...")
        self.btn_select.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_select.setMinimumHeight(45)
        self.btn_select.clicked.connect(self.select_media)
        self.media_file_label = QLabel("No file selected.")
        self.media_file_label.setObjectName("hintLabel")
        self.media_file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        media_layout.addWidget(self.btn_select)
        media_layout.addWidget(self.media_file_label)
        self.stack_source.addWidget(page_media)
        
        # Page 2: Folder Slideshow
        page_folder = QWidget()
        folder_layout = QVBoxLayout(page_folder)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_select_folder = QPushButton("Select Folder...")
        self.btn_select_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_select_folder.setMinimumHeight(45)
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.folder_label = QLabel("No folder selected.")
        self.folder_label.setObjectName("hintLabel")
        self.folder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        folder_layout.addWidget(self.btn_select_folder)
        folder_layout.addWidget(self.folder_label)
        self.stack_source.addWidget(page_folder)
        
        # Page 3: Screen Share
        page_screen = QWidget()
        screen_layout = QVBoxLayout(page_screen)
        screen_layout.setContentsMargins(0, 0, 0, 0)
        screen_desc = QLabel("Broadcast your primary computer monitor live.\n(Aspect ratio perfectly letterboxed!)")
        screen_desc.setObjectName("hintLabel")
        screen_desc.setWordWrap(True)
        screen_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        screen_layout.addWidget(screen_desc)
        self.stack_source.addWidget(page_screen)
        
        source_layout.addWidget(self.stack_source)
        left_layout.addWidget(source_group)
        
        # --- TABBED SETTINGS (Filters / Output) ---
        self.settings_tabs = QTabWidget()
        
        # Tab 1: Filters & Effects
        tab_fx = QWidget()
        fx_layout = QFormLayout(tab_fx)
        fx_layout.setContentsMargins(20, 25, 20, 25)
        fx_layout.setSpacing(15)
        
        self.combo_filter = QComboBox()
        self.combo_filter.setMinimumHeight(35)
        self.combo_filter.addItems([
            "Normal", "Pixelate (Retro 8-bit)", "Edge Detection",
            "Sepia", "Grayscale", "Invert Colors", "Matrix Green", 
            "Retro Glitch", "Blur"
        ])
        self.combo_filter.currentTextChanged.connect(self.update_settings)
        fx_layout.addRow("Video Filter:", self.combo_filter)
        
        self.input_watermark = QLineEdit()
        self.input_watermark.setMinimumHeight(35)
        self.input_watermark.setPlaceholderText("Overlay live text...")
        self.input_watermark.textChanged.connect(self.update_settings)
        fx_layout.addRow("Label Text:", self.input_watermark)
        
        # Flips inside FX
        flip_w = QWidget()
        flip_l = QHBoxLayout(flip_w)
        flip_l.setContentsMargins(0, 0, 0, 0)
        self.check_flip_x = QCheckBox("Flip ↔")
        self.check_flip_x.setCursor(Qt.CursorShape.PointingHandCursor)
        self.check_flip_x.stateChanged.connect(self.update_settings)
        self.check_flip_y = QCheckBox("Flip ↕")
        self.check_flip_y.setCursor(Qt.CursorShape.PointingHandCursor)
        self.check_flip_y.stateChanged.connect(self.update_settings)
        flip_l.addWidget(self.check_flip_x)
        flip_l.addWidget(self.check_flip_y)
        fx_layout.addRow("Orientation:", flip_w)
        
        self.settings_tabs.addTab(tab_fx, "Filters & FX")
        
        # Tab 2: Output Quality
        tab_out = QWidget()
        out_layout = QFormLayout(tab_out)
        out_layout.setContentsMargins(20, 25, 20, 25)
        out_layout.setSpacing(25)
        
        self.combo_res = QComboBox()
        self.combo_res.setMinimumHeight(40)
        self.combo_res.addItems([
            "480p (640x480)", 
            "720p (1280x720)", 
            "1080p (1920x1080)"
        ])
        self.combo_res.setCurrentIndex(1) # Default to 720p
        out_layout.addRow("Resolution:", self.combo_res)
        
        self.combo_fps = QComboBox()
        self.combo_fps.setMinimumHeight(40)
        self.combo_fps.addItems(["5", "15", "30", "60"]) 
        self.combo_fps.setCurrentIndex(2) # Default to 30fps
        out_layout.addRow("Stream FPS:", self.combo_fps)
        
        self.settings_tabs.addTab(tab_out, "Output Quality")
        
        left_layout.addWidget(self.settings_tabs)
        left_layout.addStretch()
        
        # --- PERMANENT CONTROL BUTTONS ---
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)
        
        self.btn_start = QPushButton("START STREAM")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setMinimumHeight(55)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.clicked.connect(self.start_camera)
        
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setMinimumHeight(55)
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_stop.setEnabled(False)
        
        control_layout.addWidget(self.btn_start, 2) 
        control_layout.addWidget(self.btn_stop, 1)
        
        left_layout.addLayout(control_layout)
        main_layout.addWidget(left_panel)
        
        # --- RIGHT PANEL (Preview) ---
        right_panel = QFrame()
        right_panel.setObjectName("previewFrame")
        right_layout = QVBoxLayout(right_panel)
        
        self.image_preview = QLabel("PREVIEW AREA")
        self.image_preview.setObjectName("previewLabel")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.image_preview.setWordWrap(True)
        right_layout.addWidget(self.image_preview)
        
        main_layout.addWidget(right_panel, 1)
        
        # Initialize
        self.on_mode_change(0)

    def apply_theme(self):
        dark_red_stylesheet = """
        QMainWindow, #centralWidget {
            background-color: #0d0d0d;
        }
        QLabel {
            color: #ffffff;
            font-family: 'Segoe UI', Arial, sans-serif;
            border: none;
            padding: 0;
            margin: 0;
        }
        QLabel#titleLabel {
            color: #ffffff;
            letter-spacing: 4px;
        }
        QLabel#statusLabel {
            color: #888888;
        }
        QLabel#hintLabel {
            color: #666666;
            font-size: 11px;
            font-style: italic;
        }
        QLabel#previewLabel {
            color: #333333;
            letter-spacing: 2px;
        }

        /* GroupBox Styling */
        QGroupBox {
            border: 1px solid #1a1a1a;
            border-radius: 8px;
            margin-top: 15px;
            padding-top: 20px;
            background-color: #121212;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            color: #c1121f;
            font-size: 13px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* TabWidget Styling */
        QTabWidget::pane {
            border: 1px solid #1a1a1a;
            border-radius: 8px;
            border-top-left-radius: 0;
            background-color: #121212;
            top: -1px;
        }
        QTabBar::tab {
            background: #080808;
            color: #666666;
            padding: 12px 20px;
            border: 1px solid #1a1a1a;
            border-bottom: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            margin-right: 2px;
            font-weight: 800;
            font-size: 13px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QTabBar::tab:selected {
            background: #121212;
            color: #ffffff;
            border-top: 3px solid #c1121f;
        }
        QTabBar::tab:hover:!selected {
            background: #0f0f0f;
            color: #aaaaaa;
        }

        /* Input Elements */
        QComboBox, QLineEdit {
            background-color: #080808;
            border: 1px solid #222222;
            border-radius: 4px;
            padding: 5px 12px;
            color: #ffffff;
            font-size: 13px;
            font-weight: 600;
        }
        QComboBox:disabled, QLineEdit:disabled {
            background-color: #0a0a0a;
            color: #444444;
            border: 1px solid #111111;
        }
        QComboBox:hover:!disabled, QLineEdit:hover:!disabled {
            border: 1px solid #c1121f;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 25px;
            border-left: none;
        }
        QComboBox QAbstractItemView {
            background-color: #080808;
            border: 1px solid #c1121f;
            color: white;
            selection-background-color: #c1121f;
            border-radius: 4px;
            outline: none;
        }

        /* Checkboxes */
        QCheckBox {
            color: #cccccc;
            font-size: 13px;
            font-weight: bold;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #444444;
            border-radius: 4px;
            background: #080808;
        }
        QCheckBox::indicator:hover { border: 1px solid #c1121f; }
        QCheckBox::indicator:checked {
            background: #c1121f;
            border: 1px solid #c1121f;
        }

        /* Buttons Baseline */
        QPushButton {
            background-color: #1a1a1a;
            color: #ffffff;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 800;
            font-family: 'Segoe UI', Arial, sans-serif;
            border: 1px solid #333333;
        }
        QPushButton:hover:!disabled { background-color: #2a2a2a; border: 1px solid #444444; }
        QPushButton:pressed:!disabled { background-color: #111111; }
        
        /* Specific Action Buttons */
        QPushButton#btn_start { 
            background-color: #c1121f; 
            color: #ffffff; 
            border: none; 
            font-size: 15px; 
            letter-spacing: 1px;
        }
        QPushButton#btn_start:hover:!disabled { 
            background-color: #e61a28; 
            margin-top: -1px;
        }
        QPushButton#btn_start:disabled { 
            background-color: #3d0509; 
            color: #8a363a; 
        }
        
        QPushButton#btn_stop { 
            background-color: #111111; 
            color: #666666; 
            border: 1px solid #222222; 
            font-size: 15px; 
            letter-spacing: 1px;
        }
        QPushButton#btn_stop:hover:!disabled { 
            background-color: #ff3333; 
            color: white; 
            border: none; 
        }

        /* Preview Frame */
        #previewFrame {
            border: 2px solid #1a1a1a;
            border-radius: 12px;
            background-color: #080808;
        }
        """
        self.setStyleSheet(dark_red_stylesheet)

    def on_mode_change(self, index):
        self.stack_source.setCurrentIndex(index)
        
        if index == 0:
            self.image_preview.setText("PREVIEW AREA")
            # Only enable if we have a file, not a directory
            has_file = self.media_path is not None and not os.path.isdir(self.media_path)
            self.btn_start.setEnabled(has_file)
            if has_file: self.show_media_preview(self.media_path)
        elif index == 1:
            self.image_preview.setText("SLIDESHOW MODE\n\n(Changes every 10 seconds)")
            has_dir = self.media_path is not None and os.path.isdir(self.media_path)
            self.btn_start.setEnabled(has_dir)
        elif index == 2:
            self.image_preview.setText("SCREEN SHARE MODE\n\n(Capturing Primary Monitor)")
            self.btn_start.setEnabled(True)

    def update_settings(self):
        f = self.combo_filter.currentText()
        w = self.input_watermark.text()
        fx = self.check_flip_x.isChecked()
        fy = self.check_flip_y.isChecked()
        self.core_settings.update(f, w, fx, fy)

    def select_media(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Media", "", "Media Files (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mkv *.mov)"
        )
        if file_path:
            self.media_path = file_path
            self.media_file_label.setText(file_path.split("/")[-1])
            self.folder_label.setText("No folder selected.") # Reset the other
            if self.combo_source.currentIndex() == 0:
                self.btn_start.setEnabled(True)
            self.show_media_preview(file_path)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder for Slideshow")
        if folder_path:
            self.media_path = folder_path
            self.folder_label.setText("..." + folder_path[-20:]) # Show end of path
            self.media_file_label.setText("No file selected.") # Reset the other
            if self.combo_source.currentIndex() == 1:
                self.btn_start.setEnabled(True)
            self.image_preview.setText(f"Folder Selected:\n{os.path.basename(folder_path)}\n\n(Press START to begin slideshow)")

    def show_media_preview(self, filepath):
        cap = cv2.VideoCapture(filepath)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            
            self.preview_frame_data = frame.copy()
            qimg = QImage(self.preview_frame_data.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # This scales perfectly without breaking aspect ratio for the UI itself
            scaled_pixmap = pixmap.scaled(
                self.image_preview.width() - 20, 
                self.image_preview.height() - 20, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_preview.setPixmap(scaled_pixmap)
            self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def start_camera(self):
        idx = self.combo_source.currentIndex()
        if idx == 0: mode = "media"
        elif idx == 1: mode = "folder"
        else: mode = "screen"
        
        if (mode == "media" or mode == "folder") and not self.media_path:
            return
            
        # Parse Resolution
        res_text = self.combo_res.currentText()
        if "480p" in res_text:
            cam_w, cam_h = 640, 480
        elif "1080p" in res_text:
            cam_w, cam_h = 1920, 1080
        else: # Default 720p
            cam_w, cam_h = 1280, 720
            
        # Parse FPS
        cam_fps = int(self.combo_fps.currentText())
            
        self.combo_source.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.btn_select_folder.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        # Only disable Output, Adjustments live Update
        self.combo_res.setEnabled(False)
        self.combo_fps.setEnabled(False)
        
        self.camera_thread = CameraThread(
            mode, 
            self.core_settings, 
            self.media_path,
            cam_w, 
            cam_h, 
            cam_fps
        )
        self.camera_thread.error_signal.connect(self.show_error)
        self.camera_thread.status_signal.connect(self.update_status)
        self.camera_thread.start()

    def stop_camera(self):
        if self.camera_thread and self.camera_thread.running:
            self.camera_thread.stop()
            self.camera_thread = None
            
        self.combo_source.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_select_folder.setEnabled(True)
        
        # Re-enable Start if valid state
        idx = self.combo_source.currentIndex()
        if idx == 2 or (idx == 0 and self.media_path and not os.path.isdir(self.media_path)) or (idx == 1 and self.media_path and os.path.isdir(self.media_path)):
            self.btn_start.setEnabled(True)
            
        self.btn_stop.setEnabled(False)
        self.combo_res.setEnabled(True)
        self.combo_fps.setEnabled(True)
        self.update_status("Stopped")

    def update_status(self, status_text):
        self.status_label.setText(f"Status: {status_text}")
        if "Running" in status_text:
            self.status_label.setStyleSheet("color: #e50914; font-weight: bold; border:none;") 
        elif "Error" in status_text:
            self.status_label.setStyleSheet("color: #ff3333; font-weight: bold; border:none;") 
            self.stop_camera()
        else:
            self.status_label.setStyleSheet("color: #888888; border:none;")

    def show_error(self, error_msg):
        QMessageBox.critical(self, "Virtual Camera Error", error_msg)
        self.stop_camera()

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VirtualCamAppFixed()
    window.show()
    sys.exit(app.exec())
