# SportDataLab_4-1
## Environment
```
# 1) 建環境（Python 3.10 或 3.11）
conda create -n pose_lab python=3.10 -y
conda activate pose_lab

# 2) 先用 conda 安裝底層科學套件與 OpenCV（from conda-forge）
conda install -c conda-forge numpy opencv -y

# 3) 用 pip 安裝 MediaPipe（conda 沒有官方包）
pip install mediapipe

# 4) 安裝pyQt5/確認其他套件是否安裝
pip install pyqt5 opencv-python mediapipe ultralytics numpy

# 5) 進入你的專案資料夾並執行前面給的 app
python app_pose_cpu.py
```
