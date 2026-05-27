@echo off
setlocal enabledelayedexpansion

set ACTION=%1
set TASK=%2

if "%ACTION%"=="" (
    echo Usage: run.bat [download ^| train ^| test] [Prostate ^| Lung ^| Pancreas ^| All]
    echo Example 1: run.bat download Lung
    echo Example 2: run.bat train Prostate
    echo Example 3: run.bat test Pancreas
    exit /b
)

if "%ACTION%"=="download" (
    if "%TASK%"=="" set TASK=All
    echo [Downloading] Task: !TASK!
    python dataset\download_msd_aws.py --task !TASK!
    exit /b
)

if "%TASK%"=="" (
    echo Error: You must specify a task for training/testing.
    echo Example: run.bat train Lung
    exit /b
)

if "%TASK%"=="Prostate" (
    set DATA_DIR=dataset/Task05_Prostate/
    set LOG_DIR=runs/prostate
    set MAX_EPOCHS=2000
    set VAL_EVERY=25
) else if "%TASK%"=="Lung" (
    set DATA_DIR=dataset/Task06_Lung/
    set LOG_DIR=runs/lung
    set MAX_EPOCHS=700
    set VAL_EVERY=10
) else if "%TASK%"=="Pancreas" (
    set DATA_DIR=dataset/Task07_Pancreas/
    set LOG_DIR=runs/pancreas
    set MAX_EPOCHS=700
    set VAL_EVERY=10
) else (
    echo Error: Invalid task '!TASK!'. Choose Prostate, Lung, or Pancreas.
    exit /b
)

if "%ACTION%"=="train" (
    echo ========================================================
    echo [Training] Task: !TASK! 
    echo [Data Dir] !DATA_DIR!
    echo [Log Dir]  !LOG_DIR!
    echo [Epochs]   !MAX_EPOCHS!
    echo ========================================================
    python main.py --task !TASK! --fold 0 --data_dir !DATA_DIR! --json_list dataset.json --use_checkpoint --workers 2 --roi_x 64 --roi_y 64 --roi_z 64 --max_epochs !MAX_EPOCHS! --val_every !VAL_EVERY! --save_checkpoint --logdir !LOG_DIR! --use_normal_dataset
) else if "%ACTION%"=="test" (
    echo ========================================================
    echo [Testing] Task: !TASK! 
    echo [Data Dir] !DATA_DIR!
    echo [Model]    ./!LOG_DIR!/model_final.pt
    echo ========================================================
    python test.py --task !TASK! --fold 0 --data_dir !DATA_DIR! --json_list dataset.json --pretrained_dir ./!LOG_DIR!/ --pretrained_model_name model_final.pt --roi_x 64 --roi_y 64 --roi_z 64 --workers 0
) else (
    echo Error: Invalid action '!ACTION!'. Choose download, train, or test.
)

endlocal
