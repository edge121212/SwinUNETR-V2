@echo off
setlocal enabledelayedexpansion

set ACTION=%1
set TASK=%2
set EPOCHS_OVERRIDE=%3

if "%ACTION%"=="" (
    echo Usage: run.bat [download ^| train ^| test ^| kfold] [Prostate ^| Lung ^| Pancreas ^| All] [epochs?]
    echo Example 1: run.bat download Prostate
    echo Example 2: run.bat train Prostate
    echo Example 3: run.bat kfold Prostate 500
    exit /b
)

if "%ACTION%"=="download" (
    if "%TASK%"=="" set TASK=All
    echo [Downloading] Task: !TASK!
    python dataset\download_msd_aws.py --task !TASK!
    exit /b
)

if "%TASK%"=="" (
    echo Error: You must specify a task for %ACTION%.
    exit /b
)

:: TARGET_ITERS: paper recipe adapts epochs per task so total training iterations ~= 40000.
:: ROI: paper/SwinUNETR recipe uses 96^3; we keep 96^3 for Prostate to align with the paper
::      (SpatialPadd handles volumes thinner than 96 in z after 1mm resampling).
if "%TASK%"=="Prostate" (
    set DATA_DIR=dataset/Task05_Prostate/
    set LOG_BASE=prostate
    set DEFAULT_EPOCHS=2000
    set VAL_EVERY=25
    set IN_CH=2
    set OUT_CH=3
    set TARGET_ITERS=40000
    set ROI_X=96
    set ROI_Y=96
    set ROI_Z=96
) else if "%TASK%"=="Lung" (
    set DATA_DIR=dataset/Task06_Lung/
    set LOG_BASE=lung
    set DEFAULT_EPOCHS=700
    set VAL_EVERY=10
    set IN_CH=1
    set OUT_CH=2
    set TARGET_ITERS=40000
    set ROI_X=64
    set ROI_Y=64
    set ROI_Z=64
) else if "%TASK%"=="Pancreas" (
    set DATA_DIR=dataset/Task07_Pancreas/
    set LOG_BASE=pancreas
    set DEFAULT_EPOCHS=700
    set VAL_EVERY=10
    set IN_CH=1
    set OUT_CH=3
    set TARGET_ITERS=40000
    set ROI_X=64
    set ROI_Y=64
    set ROI_Z=64
) else (
    echo Error: Invalid task '!TASK!'. Choose Prostate, Lung, or Pancreas.
    exit /b
)

:: If the user passes an explicit epoch count, honor it (fixed max_epochs); otherwise auto-target ~40k iters.
if "!EPOCHS_OVERRIDE!"=="" (
    set EPOCH_ARGS=--target_iters !TARGET_ITERS!
    set EPOCH_DESC=~!TARGET_ITERS! iters auto-epochs
) else (
    set EPOCH_ARGS=--max_epochs !EPOCHS_OVERRIDE!
    set EPOCH_DESC=!EPOCHS_OVERRIDE! epochs
)

if not exist "!DATA_DIR!" (
    echo Error: 資料夾 '!DATA_DIR!' 不存在，請先執行 'run.bat download !TASK!'。
    exit /b
)

if "%ACTION%"=="train" (
    echo ========================================================
    echo [Training] Task: !TASK!  Fold: 0  Budget: !EPOCH_DESC!  ROI: !ROI_X!x!ROI_Y!x!ROI_Z!
    echo ========================================================
    python main.py --task !TASK! --fold 0 --data_dir !DATA_DIR! --json_list dataset.json --use_checkpoint --workers 2 --roi_x !ROI_X! --roi_y !ROI_Y! --roi_z !ROI_Z! !EPOCH_ARGS! --val_every !VAL_EVERY! --in_channels !IN_CH! --out_channels !OUT_CH! --save_checkpoint --logdir !LOG_BASE! --use_normal_dataset
    exit /b
)

if "%ACTION%"=="test" (
    echo ========================================================
    echo [Testing]  Task: !TASK!  Fold: 0
    echo [Model]    ./runs/!LOG_BASE!/model.pt  (best-on-validation)
    echo ========================================================
    python test.py --task !TASK! --fold 0 --data_dir !DATA_DIR! --json_list dataset.json --pretrained_dir ./runs/!LOG_BASE!/ --pretrained_model_name model.pt --roi_x !ROI_X! --roi_y !ROI_Y! --roi_z !ROI_Z! --workers 0 --in_channels !IN_CH! --out_channels !OUT_CH!
    exit /b
)

if "%ACTION%"=="kfold" (
    if not exist "runs" mkdir runs
    for %%F in (0 1 2 3 4) do (
        set LOGDIR=!LOG_BASE!_fold%%F
        set TESTLOG=runs\!LOGDIR!_test.log
        if exist "runs\!LOGDIR!\model_final.pt" (
            echo [Skip train] runs\!LOGDIR!\model_final.pt already exists.
        ) else (
            echo ========================================================
            echo [Training] Task: !TASK!  Fold: %%F  Budget: !EPOCH_DESC!  ROI: !ROI_X!x!ROI_Y!x!ROI_Z!
            echo ========================================================
            python main.py --task !TASK! --fold %%F --data_dir !DATA_DIR! --json_list dataset.json --use_checkpoint --workers 2 --roi_x !ROI_X! --roi_y !ROI_Y! --roi_z !ROI_Z! !EPOCH_ARGS! --val_every !VAL_EVERY! --in_channels !IN_CH! --out_channels !OUT_CH! --save_checkpoint --logdir !LOGDIR! --use_normal_dataset
        )
        echo ========================================================
        echo [Testing]  Task: !TASK!  Fold: %%F
        echo ========================================================
        python test.py --task !TASK! --fold %%F --data_dir !DATA_DIR! --json_list dataset.json --pretrained_dir ./runs/!LOGDIR!/ --pretrained_model_name model.pt --roi_x !ROI_X! --roi_y !ROI_Y! --roi_z !ROI_Z! --workers 0 --in_channels !IN_CH! --out_channels !OUT_CH! --exp_name !LOGDIR! > !TESTLOG! 2>&1
        type !TESTLOG!
    )
    echo ========================================================
    echo [Aggregating 5-fold results for !TASK!]
    echo ========================================================
    python utils\aggregate_kfold.py --task !TASK! --log_base !LOG_BASE!
    exit /b
)

echo Error: Invalid action '!ACTION!'. Choose download, train, test, or kfold.
endlocal
