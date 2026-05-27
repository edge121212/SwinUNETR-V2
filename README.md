# SwinUNETR-V2 醫學影像分割 (Medical Image Segmentation)

這是一個基於 MONAI 與 SwinUNETR 模型的 3D 醫療影像分割專案。
完美支援 MSD (Medical Segmentation Decathlon) 的三個經典任務，並且解決了記憶體爆滿的問題：
- **Task 05**: Prostate (攝護腺 MRI)
- **Task 06**: Lung (肺部 CT)
- **Task 07**: Pancreas (胰臟 CT)

---

## 🛠 1. 環境安裝 (Installation)
請確保你的電腦有安裝 Python (建議 3.9+) 以及對應你顯示卡的 PyTorch (需支援 CUDA)。
打開終端機 (Terminal / PowerShell)，依序執行：
```bash
# 把專案抓下來
git clone https://github.com/edge121212/SwinUNETR-V2.git

# 進入資料夾
cd SwinUNETR-V2

# 安裝所有必備套件
pip install -r requirements.txt
```

---

## 📥 2. 下載資料集 (Download Datasets)
我們提供了一個防呆腳本 `run.bat`，可以一鍵下載並自動解壓縮：
*(注意：醫療影像檔案極大，請確保硬碟有足夠空間)*

```cmd
# 下載指定的器官資料
.\run.bat download Lung
.\run.bat download Prostate
.\run.bat download Pancreas

# 或者一次把三個全部載滿
.\run.bat download All
```
下載完成後，資料集會自動放在 `dataset/` 目錄下。

---

## 🏋️ 3. 模型訓練 (Training)
下載完資料集後，直接呼叫腳本即可開始訓練。
*(已自動根據資料集大小調整最佳訓練長度：Prostate 會跑 2000 Epochs，Lung/Pancreas 會跑 700 Epochs，皆完美對齊官方 40k iterations 的收斂標準)*

```cmd
.\run.bat train Lung
.\run.bat train Prostate
.\run.bat train Pancreas
```
訓練好的大腦 (`model_final.pt`) 會自動存檔在 `runs/` 對應的資料夾裡面。

---

## 🧪 4. 模型測試 (Testing)
訓練結束後，想要看看模型有多準（計算 Dice Score），請執行：

```cmd
.\run.bat test Lung
.\run.bat test Prostate
.\run.bat test Pancreas
```
分數會直接顯示在螢幕上，並且 Prostate 任務會貼心地幫你拆分出 PZ (周邊區) 與 TZ (過渡區) 的獨立分數！

---

## 💡 (選用) 如果你想在 Google Colab 上跑...
如果你朋友想借用 Google 的免費 GPU (Colab)，因為 Colab 不支援 `.bat` 腳本，請貼以下 Python 指令給她：

```bash
# 下載
!python dataset/download_msd_aws.py --task Lung

# 訓練 (務必加上 --use_normal_dataset 避免記憶體爆滿)
!python main.py --task Lung --fold 0 --data_dir dataset/Task06_Lung/ --json_list dataset.json --use_checkpoint --workers 2 --roi_x 64 --roi_y 64 --roi_z 64 --max_epochs 50 --val_every 5 --save_checkpoint --logdir runs/lung --use_normal_dataset

# 測試 (務必加上 --workers 0 避免 DataLoader 崩潰)
!python test.py --task Lung --fold 0 --data_dir dataset/Task06_Lung/ --json_list dataset.json --pretrained_dir ./runs/lung/ --pretrained_model_name model_final.pt --roi_x 64 --roi_y 64 --roi_z 64 --workers 0
```
