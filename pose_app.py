import sys, os, time, csv
import numpy as np
import cv2
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets

# ==============================
# 常數（寫死設定）
# ==============================
APP_TITLE = "Pose App - MediaPipe / YOLOv8"
FIXED_W, FIXED_H = 960, 540             # 擷取解析度（推論/CSV 皆用這個）
MP_MODEL_COMPLEXITY = 1                 # MediaPipe 複雜度：0/1/2
YOLO_WEIGHTS = Path(__file__).with_name("yolov8n-pose.pt")  # YOLO 權重（同資料夾）

# ==============================
# 視覺畫布：鋪滿顯示 + HUD（FPS）
# ==============================
class VideoCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix = None
        self._fps_text = "FPS: -"
        self.setStyleSheet("background:#222;")
        self.setMinimumSize(1, 1)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    @QtCore.pyqtSlot(QtGui.QImage)
    def setFrame(self, img: QtGui.QImage):
        if img is None:
            self._pix = None
        else:
            self._pix = QtGui.QPixmap.fromImage(img)
        self.update()

    @QtCore.pyqtSlot(float)
    def setFps(self, fps: float):
        self._fps_text = f"FPS: {fps:.1f}"

    def reset(self):
        self._pix = None
        self._fps_text = "FPS: -"
        self.update()

    def paintEvent(self, e: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

        # 背景
        p.fillRect(self.rect(), QtGui.QColor("#222"))

        # 影像：維持比例且鋪滿（必要時裁切），置中顯示
        if self._pix:
            target_size = self.size()
            src_size = self._pix.size()
            scaled_size = src_size.scaled(target_size, QtCore.Qt.KeepAspectRatioByExpanding)
            x = (target_size.width() - scaled_size.width()) // 2
            y = (target_size.height() - scaled_size.height()) // 2
            p.drawPixmap(QtCore.QRect(QtCore.QPoint(x, y), scaled_size), self._pix)
        else:
            # 初始/停止狀態提示
            p.setPen(QtGui.QColor(180, 180, 180))
            font = p.font(); font.setPointSize(16); p.setFont(font)
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "Stopped – press Start")

        # HUD：FPS（永遠在可視範圍）
        if self._fps_text:
            margin = 10
            font = p.font(); font.setPointSize(14); font.setBold(True); p.setFont(font)
            metrics = QtGui.QFontMetrics(font)
            text_w = metrics.horizontalAdvance(self._fps_text)
            text_h = metrics.height()
            bg_rect = QtCore.QRect(margin-6, margin-6, text_w+12, text_h+12)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(0, 0, 0, 140))
            p.drawRoundedRect(bg_rect, 6, 6)
            p.setPen(QtGui.QColor(0, 255, 0))
            p.drawText(margin, margin + text_h, self._fps_text)
        p.end()

# ==============================
# 後端共用介面
# ==============================
class PoseBackend(QtCore.QObject):
    def start(self): ...
    def stop(self): ...
    def process(self, frame_bgr): ...
    def header(self): ...
    def name(self): ...

# ==============================
# MediaPipe 後端（model_complexity 寫死）
# ==============================
class MediaPipeBackend(PoseBackend):
    def __init__(self):
        super().__init__()
        self.mp = None
        self.pose = None
        self.drawing = None
        self.names = None

    def name(self): return "MediaPipe"

    def start(self):
        import mediapipe as mp
        self.mp = mp
        self.drawing = mp.solutions.drawing_utils
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=MP_MODEL_COMPLEXITY,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.names = [lm.name for lm in mp.solutions.pose.PoseLandmark]

    def stop(self):
        if self.pose:
            self.pose.close()
        self.pose = None

    # 回傳「關鍵點欄位」標頭（不含 frame/timestamp）
    def header(self):
        h = []
        for n in self.names:
            h += [f"{n}_x", f"{n}_y", f"{n}_z", f"{n}_vis"]
        return h

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        row = []
        if res.pose_landmarks:
            self.drawing.draw_landmarks(
                frame_bgr, res.pose_landmarks,
                self.mp.solutions.pose.POSE_CONNECTIONS,
                self.drawing.DrawingSpec(thickness=2, circle_radius=2),
                self.drawing.DrawingSpec(thickness=2)
            )
            lms = res.pose_landmarks.landmark
            for i in range(len(self.names)):
                lm = lms[i]
                row += [lm.x, lm.y, lm.z, lm.visibility]
        else:
            for _ in range(len(self.names)):
                row += [np.nan, np.nan, np.nan, np.nan]
        return frame_bgr, row

# ==============================
# YOLOv8 後端（COCO 17 點；權重寫死）
# ==============================
class YOLOv8Backend(PoseBackend):
    def __init__(self):
        super().__init__()
        self.model = None
        self.names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

    def name(self): return "YOLOv8"

    def start(self):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("未安裝 ultralytics，請先執行：pip install ultralytics") from e
        model_path = str(YOLO_WEIGHTS) if YOLO_WEIGHTS.exists() else "yolov8n-pose.pt"
        self.model = YOLO(model_path)

    def stop(self):
        self.model = None

    # 回傳「關鍵點欄位」標頭（不含 frame/timestamp）
    def header(self):
        h = []
        for n in self.names:
            h += [f"{n}_x", f"{n}_y", f"{n}_z", f"{n}_vis"]
        return h

    def process(self, frame_bgr):
        results = self.model(frame_bgr, verbose=False)
        r = results[0]
        annotated = r.plot()
        row = []
        if r.keypoints is not None and len(r.keypoints) > 0:
            kp_xy = r.keypoints.xy[0].cpu().numpy()        # (17, 2) 第一位人物
            kp_conf = None
            if hasattr(r.keypoints, "conf") and r.keypoints.conf is not None:
                kp_conf = r.keypoints.conf[0].cpu().numpy()  # (17,)
            h, w = annotated.shape[:2]
            for i in range(len(self.names)):
                x, y = kp_xy[i]
                row += [float(x)/w, float(y)/h, np.nan,
                        float(kp_conf[i]) if kp_conf is not None else np.nan]
        else:
            for _ in range(len(self.names)):
                row += [np.nan, np.nan, np.nan, np.nan]
        return annotated, row

# ==============================
# 擷取執行緒（背景執行）
# ==============================
class CaptureWorker(QtCore.QThread):
    frameReady = QtCore.pyqtSignal(QtGui.QImage, float)  # (frame, fps)
    rowReady = QtCore.pyqtSignal(list)                   # CSV row（含 frame_id/timestamp）

    def __init__(self, cam_index, backend: PoseBackend, parent=None):
        super().__init__(parent)
        self.cam_index = int(cam_index)
        self.backend = backend
        self._running = False
        self._frame_id = 0
        self._fps_interval = 0.5
        self._last_update = time.time()
        self._cnt = 0
        self._fps = 0.0
        self.t0 = None  # 起算時間（perf_counter）

    def run(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)  # Windows：DSHOW；跨平台可移除第二參數
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FIXED_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FIXED_H)
        if not cap.isOpened():
            QtCore.qWarning("Cannot open camera")
            return

        self.backend.start()
        self._running = True
        self._frame_id = 0
        self._cnt = 0
        self._fps = 0.0
        self._last_update = time.time()
        self.t0 = time.perf_counter()  # 相對時間起點

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break
                # 保險：部分驅動會忽略 set，強制 resize
                frame = cv2.resize(frame, (FIXED_W, FIXED_H), interpolation=cv2.INTER_LINEAR)

                self._frame_id += 1
                ts_ms = int(time.time() * 1000)                           # 電腦時間（Unix ms）
                elapsed_ms = int((time.perf_counter() - self.t0) * 1000)  # 錄影相對時間（ms）

                frame, row_data = self.backend.process(frame)

                # FPS 計算（支援 Py3.7）
                self._cnt += 1
                now = time.time()
                if (now - self._last_update) >= self._fps_interval:
                    delta = now - self._last_update
                    self._fps = self._cnt / delta if delta > 0 else 0.0
                    self._cnt = 0
                    self._last_update = now

                # CSV
                row = [self._frame_id, ts_ms, elapsed_ms] + row_data
                self.rowReady.emit(row)

                # 送影像到 GUI
                qimg = QtGui.QImage(frame.data, FIXED_W, FIXED_H,
                                    frame.strides[0], QtGui.QImage.Format_BGR888)
                self.frameReady.emit(qimg.copy(), self._fps)

                QtCore.QThread.msleep(1)
        finally:
            cap.release()
            self.backend.stop()

    def stop(self):
        self._running = False
        self.wait(1000)

# ==============================
# 主視窗（Stop 會恢復初始畫面 + 幀過濾 +「經過秒數」顯示）
# ==============================
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)

        # 控制列
        self.cmb_backend = QtWidgets.QComboBox(); self.cmb_backend.addItems(["MediaPipe", "YOLOv8"])
        self.spn_cam = QtWidgets.QSpinBox(); self.spn_cam.setRange(0, 16); self.spn_cam.setValue(0)
        self.edt_csv = QtWidgets.QLineEdit("teamXX_output.csv")
        self.btn_browse = QtWidgets.QPushButton("…")
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop  = QtWidgets.QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.lbl_top_fps = QtWidgets.QLabel("FPS: -")
        self.lbl_elapsed = QtWidgets.QLabel("Elapsed: 0 s")   # <<< 新增：經過秒數顯示（整秒）

        # 影像畫布
        self.canvas = VideoCanvas()

        # 版面
        form = QtWidgets.QFormLayout()
        form.addRow("Backend", self.cmb_backend)
        form.addRow("Camera Index", self.spn_cam)
        csvrow = QtWidgets.QHBoxLayout()
        csvrow.addWidget(self.edt_csv); csvrow.addWidget(self.btn_browse)
        form.addRow("CSV Output", csvrow)

        btnrow = QtWidgets.QHBoxLayout()
        btnrow.addWidget(self.btn_start); btnrow.addWidget(self.btn_stop)
        btnrow.addStretch(1)
        btnrow.addWidget(self.lbl_elapsed)  # 放在右側
        btnrow.addSpacing(12)
        btnrow.addWidget(self.lbl_top_fps)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(form)
        layout.addLayout(btnrow)
        layout.addWidget(self.canvas, 1)

        # 狀態
        self.worker = None
        self.csv_file = None
        self.csv_writer = None
        self.accept_frames = False  # 幀過濾旗標

        # 計時器（以秒為單位）
        self._ui_t0 = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(200)  # 200ms 更新一次，但顯示整秒
        self._timer.timeout.connect(self._tick_elapsed)

        # 事件
        self.btn_browse.clicked.connect(self.onBrowse)
        self.btn_start.clicked.connect(self.onStart)
        self.btn_stop.clicked.connect(self.onStop)
        self.cmb_backend.currentTextChanged.connect(self.onBackendChange)

        self.onBackendChange(self.cmb_backend.currentText())

    # 熱鍵：B 執行中切換後端
    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == QtCore.Qt.Key_B and self.btn_stop.isEnabled():
            self.toggleBackendInRuntime()

    def onBackendChange(self, text):
        self.edt_csv.setText("teamXX_output.csv" if text == "MediaPipe" else "teamXX_output_bonus.csv")

    def onBrowse(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Choose CSV", self.edt_csv.text(), "CSV (*.csv)")
        if path: self.edt_csv.setText(path)

    def makeBackend(self) -> PoseBackend:
        return MediaPipeBackend() if self.cmb_backend.currentText() == "MediaPipe" else YOLOv8Backend()

    def openCsv(self, kp_header):
        # CSV 標頭：加入 frame、兩種 timestamp，再接關鍵點欄位
        path = self.edt_csv.text().strip()
        d = os.path.dirname(path)
        if d: os.makedirs(d, exist_ok=True)
        self.csv_file = open(path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        header = ["frame", "timestamp_ms", "elapsed_ms"] + kp_header
        self.csv_writer.writerow(header)

    def closeCsv(self):
        if self.csv_file:
            try: self.csv_file.flush()
            except: pass
            self.csv_file.close()
        self.csv_file = None
        self.csv_writer = None

    def _start_elapsed_timer(self):
        self._ui_t0 = time.perf_counter()
        self.lbl_elapsed.setText("Elapsed: 0 s")
        self._timer.start()

    def _stop_elapsed_timer(self):
        self._timer.stop()
        self._ui_t0 = None
        self.lbl_elapsed.setText("Elapsed: 0 s")

    def _tick_elapsed(self):
        if self._ui_t0 is None:
            return
        secs = int(time.perf_counter() - self._ui_t0)  # 整秒
        self.lbl_elapsed.setText(f"Elapsed: {secs} s")

    def onStart(self):
        if self.worker and self.worker.isRunning(): return
        try:
            tmp = self.makeBackend(); tmp.start(); kp_header = tmp.header(); tmp.stop()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "後端啟動失敗", str(e)); return
        self.openCsv(kp_header)

        self.worker = CaptureWorker(
            cam_index=int(self.spn_cam.value()),
            backend=self.makeBackend()
        )

        # 先連線，再開放接收幀
        self.worker.frameReady.connect(self.onFrame)
        self.worker.rowReady.connect(self.onRow)
        self.accept_frames = True
        self.worker.start()

        # 啟動視窗計時（以秒為單位）
        self._start_elapsed_timer()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def onStop(self):
        # 立刻停止接收任何延遲訊號
        self.accept_frames = False
        if self.worker:
            try:
                self.worker.frameReady.disconnect(self.onFrame)
                self.worker.rowReady.disconnect(self.onRow)
            except TypeError:
                pass
            self.worker.stop()
            self.worker = None

        self.closeCsv()

        # Stop 後恢復初始狀態
        self.canvas.reset()
        self.lbl_top_fps.setText("FPS: -")
        self._stop_elapsed_timer()  # 歸零秒數

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def toggleBackendInRuntime(self):
        if not self.worker: return

        # 與 onStop 同步處理，避免殘留幀
        self.accept_frames = False
        try:
            self.worker.frameReady.disconnect(self.onFrame)
            self.worker.rowReady.disconnect(self.onRow)
        except TypeError:
            pass
        self.worker.stop()
        self.worker = None
        self.closeCsv()
        self.canvas.reset()
        self.lbl_top_fps.setText("FPS: -")
        self._stop_elapsed_timer()

        # 切換後端
        self.cmb_backend.setCurrentText("YOLOv8" if self.cmb_backend.currentText()=="MediaPipe" else "MediaPipe")
        try:
            tmp = self.makeBackend(); tmp.start(); kp_header = tmp.header(); tmp.stop()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "切換失敗", str(e))
            self.cmb_backend.setCurrentText("MediaPipe")
            tmp = self.makeBackend(); tmp.start(); kp_header = tmp.header(); tmp.stop()

        self.openCsv(kp_header)

        self.worker = CaptureWorker(
            cam_index=int(self.spn_cam.value()),
            backend=self.makeBackend()
        )
        self.worker.frameReady.connect(self.onFrame)
        self.worker.rowReady.connect(self.onRow)
        self.accept_frames = True
        self.worker.start()

        # 重新啟動視窗計時
        self._start_elapsed_timer()

    @QtCore.pyqtSlot(QtGui.QImage, float)
    def onFrame(self, qimg, fps):
        if not self.accept_frames:
            return
        self.lbl_top_fps.setText(f"FPS: {fps:.1f}")
        self.canvas.setFps(fps)
        self.canvas.setFrame(qimg)

    @QtCore.pyqtSlot(list)
    def onRow(self, row):
        if self.csv_writer:
            self.csv_writer.writerow(row)

    def closeEvent(self, e: QtGui.QCloseEvent):
        self.onStop()
        e.accept()

# ==============================
def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    w = MainWindow()
    w.showMaximized()   # 或改成 showFullScreen()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

