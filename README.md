# SwinUNETR-V2 醫學影像分割 (Medical Image Segmentation)

這是一個基於 MONAI 與 SwinUNETR 模型的 3D 醫療影像分割專案。
完美支援 MSD (Medical Segmentation Decathlon) 的三個經典任務，並且解決了記憶體爆滿的問題：
- **Task 05**: Prostate (攝護腺 MRI)
- **Task 06**: Lung (肺部 CT)
- **Task 07**: Pancreas (胰臟 CT)

> **模型架構**：本專案在 `main.py` / `test.py` 中以 `use_v2=True` 啟用 MONAI `SwinUNETR` 的
> **V2 結構**（每個 stage 開頭加上 stage-wise 殘差卷積 ResConv block），對應 SwinUNETR-V2 論文。
> 若要改用原版 SwinUNETR backbone，把這兩處的 `use_v2=True` 移除即可（注意訓練與測試需一致）。

> **作業系統**：Windows 用 `run.bat`，Linux / WSL / macOS 用 `run.sh`（指令參數相同，以下兩種寫法擇一）。

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
:: Windows
.\run.bat download Prostate      :: 下載指定器官 (Lung / Prostate / Pancreas)
.\run.bat download All           :: 一次載滿三個
```
```bash
# Linux / WSL / macOS
./run.sh download Prostate
./run.sh download All
```
下載完成後，資料集會自動放在 `dataset/` 目錄下。

---

## 🏋️ 3. 模型訓練 (Training) — 單一 fold
下載完資料集後，直接呼叫腳本即可開始訓練（只跑 fold 0，適合快速驗證流程）。

```cmd
:: Windows
.\run.bat train Prostate
```
```bash
# Linux / WSL
./run.sh train Prostate
```
訓練好的權重 (`model_final.pt`) 會自動存檔在 `runs/` 對應的資料夾裡面。
可在指令最後加上 epoch 數覆寫預設值，例如 `./run.sh train Prostate 500`。

---

## 🧪 4. 模型測試 (Testing) — 單一 fold
訓練結束後，想看看模型有多準（計算 Dice Score），請執行：

```cmd
:: Windows
.\run.bat test Prostate
```
```bash
# Linux / WSL
./run.sh test Prostate
```
分數會直接顯示在螢幕上，並且 Prostate 任務會貼心地幫你拆分出 PZ (周邊區) 與 TZ (過渡區) 的獨立分數！

---

## 🔁 5. 完整 5-fold 交叉驗證 (kfold) — 論文重現用
這是**對齊論文 Table 4 的標準做法**：對 fold 0~4 依序「訓練 → 測試」，最後自動彙整出 5-fold 平均 Dice。

```cmd
:: Windows
.\run.bat kfold Prostate 500
```
```bash
# Linux / WSL
./run.sh kfold Prostate 500
```
- 最後一個參數是每個 fold 的訓練 epoch 數（範例為 500）。
- 每個 fold 的測試結果會存到 `runs/<task>_foldN_test.log`。
- 全部跑完會自動呼叫 `utils/aggregate_kfold.py` 印出各指標的 **平均 ± 標準差**。
- **斷點續跑**：`run.sh` 若偵測到某 fold 的 test log 已存在會跳過該 fold；若只有 checkpoint 則會從該 checkpoint 續訓，因此中途關機也能接著跑。
- ⚠️ **改了模型架構（例如切換 `use_v2`）後**，舊的 `runs/` 權重不可再用，請先把 `runs/` 移走或刪除再重跑，否則會載入到不相容的舊權重。

---

## 💡 6. (選用) 如果你想在 Google Colab 上跑...
Colab 不支援 `.bat` / `.sh` 腳本，可直接貼以下 Python 指令（以 Prostate 為例）：

```bash
# 下載
!python dataset/download_msd_aws.py --task Prostate

# 訓練 (務必加上 --use_normal_dataset 避免記憶體爆滿；max_epochs 視時間調整，論文設定約 500)
!python main.py --task Prostate --fold 0 --data_dir dataset/Task05_Prostate/ --json_list dataset.json --use_checkpoint --workers 2 --roi_x 64 --roi_y 64 --roi_z 64 --max_epochs 500 --val_every 25 --save_checkpoint --logdir prostate_fold0 --use_normal_dataset

# 測試 (務必加上 --workers 0 避免 DataLoader 崩潰)
!python test.py --task Prostate --fold 0 --data_dir dataset/Task05_Prostate/ --json_list dataset.json --pretrained_dir ./runs/prostate_fold0/ --pretrained_model_name model_final.pt --roi_x 64 --roi_y 64 --roi_z 64 --workers 0
```
