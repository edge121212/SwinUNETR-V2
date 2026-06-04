#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
TASK="${2:-}"
EPOCHS_OVERRIDE="${3:-}"

usage() {
    echo "Usage: ./run.sh [download | train | test | kfold] [Prostate | Lung | Pancreas | All] [epochs?]"
    echo "  train/test : single fold (fold 0)"
    echo "  kfold      : 5-fold cross validation (paper setting), aggregates results"
    echo "Examples:"
    echo "  ./run.sh download Prostate"
    echo "  ./run.sh kfold Prostate 500"
}

if [[ -z "$ACTION" ]]; then usage; exit 0; fi

if [[ "$ACTION" == "download" ]]; then
    : "${TASK:=All}"
    echo "[Downloading] Task: $TASK"
    python3 dataset/download_msd_aws.py --task "$TASK"
    # Move to dataset/ if extracted in repo root
    for t in Task05_Prostate Task06_Lung Task07_Pancreas; do
        if [[ -d "$t" && ! -d "dataset/$t" ]]; then mv "$t" dataset/; fi
        [[ -f "${t}.tar" ]] && rm -f "${t}.tar"
    done
    exit 0
fi

if [[ -z "$TASK" ]]; then echo "Error: TASK required for $ACTION."; usage; exit 1; fi

# TARGET_ITERS: paper recipe adapts epochs per task so total training iterations ~= 40000.
# ROI: paper/SwinUNETR recipe uses 96^3; we keep 96^3 for Prostate to align with the paper
# (SpatialPadd handles volumes thinner than 96 in z after 1mm resampling).
case "$TASK" in
    Prostate)  DATA_DIR="dataset/Task05_Prostate/"; LOG_BASE="prostate"; DEFAULT_EPOCHS=2000; VAL_EVERY=25; IN_CH=2; OUT_CH=3; TARGET_ITERS=40000; ROI_X=96; ROI_Y=96; ROI_Z=96 ;;
    Lung)      DATA_DIR="dataset/Task06_Lung/";      LOG_BASE="lung";      DEFAULT_EPOCHS=700;  VAL_EVERY=10; IN_CH=1; OUT_CH=2; TARGET_ITERS=40000; ROI_X=64; ROI_Y=64; ROI_Z=64 ;;
    Pancreas)  DATA_DIR="dataset/Task07_Pancreas/";  LOG_BASE="pancreas";  DEFAULT_EPOCHS=700;  VAL_EVERY=10; IN_CH=1; OUT_CH=3; TARGET_ITERS=40000; ROI_X=64; ROI_Y=64; ROI_Z=64 ;;
    *) echo "Error: Invalid task '$TASK'."; exit 1 ;;
esac

# If the user passes an explicit epoch count, honor it (fixed max_epochs); otherwise auto-target ~40k iters.
if [[ -n "$EPOCHS_OVERRIDE" ]]; then
    EPOCH_ARGS=(--max_epochs "$EPOCHS_OVERRIDE")
    EPOCH_DESC="$EPOCHS_OVERRIDE epochs"
else
    EPOCH_ARGS=(--target_iters "$TARGET_ITERS")
    EPOCH_DESC="~${TARGET_ITERS} iters (auto epochs)"
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: '$DATA_DIR' not found. Run './run.sh download $TASK' first."
    exit 1
fi

run_train_one_fold() {
    local fold="$1"
    local logdir="$2"
    local ckpt="runs/${logdir}/model_final.pt"
    local resume_args=()
    if [[ -f "$ckpt" ]]; then
        resume_args=(--checkpoint "$ckpt")
        echo "[Resume]   Found existing checkpoint, resuming from $ckpt"
    fi
    echo "========================================================"
    echo "[Training] Task: $TASK  Fold: $fold  Budget: $EPOCH_DESC  ROI: ${ROI_X}x${ROI_Y}x${ROI_Z}"
    echo "[Logdir]   runs/$logdir"
    echo "========================================================"
    python3 main.py --task "$TASK" --fold "$fold" \
        --data_dir "$DATA_DIR" --json_list dataset.json \
        --use_checkpoint --workers 2 \
        --roi_x "$ROI_X" --roi_y "$ROI_Y" --roi_z "$ROI_Z" \
        "${EPOCH_ARGS[@]}" --val_every "$VAL_EVERY" \
        --in_channels "$IN_CH" --out_channels "$OUT_CH" \
        --save_checkpoint --logdir "$logdir" --use_normal_dataset \
        "${resume_args[@]}"
}

run_test_one_fold() {
    local fold="$1"
    local logdir="$2"
    local outlog="$3"
    echo "========================================================"
    echo "[Testing]  Task: $TASK  Fold: $fold"
    echo "[Model]    ./runs/$logdir/model.pt  (best-on-validation; matches paper's val-based model selection)"
    echo "========================================================"
    python3 test.py --task "$TASK" --fold "$fold" \
        --data_dir "$DATA_DIR" --json_list dataset.json \
        --pretrained_dir "./runs/$logdir/" --pretrained_model_name model.pt \
        --roi_x "$ROI_X" --roi_y "$ROI_Y" --roi_z "$ROI_Z" --workers 0 \
        --in_channels "$IN_CH" --out_channels "$OUT_CH" \
        --exp_name "${logdir}" 2>&1 | tee "$outlog"
}

if [[ "$ACTION" == "train" ]]; then
    run_train_one_fold 0 "${LOG_BASE}"
elif [[ "$ACTION" == "test" ]]; then
    run_test_one_fold 0 "${LOG_BASE}" "runs/${LOG_BASE}_test.log"
elif [[ "$ACTION" == "kfold" ]]; then
    mkdir -p runs
    for fold in 0 1 2 3 4; do
        LOGDIR="${LOG_BASE}_fold${fold}"
        TESTLOG="runs/${LOGDIR}_test.log"
        if [[ -f "$TESTLOG" ]]; then
            echo "[Skip fold $fold] test log $TESTLOG already exists, fold complete."
            continue
        fi
        run_train_one_fold "$fold" "$LOGDIR"
        run_test_one_fold "$fold" "$LOGDIR" "$TESTLOG"
    done
    echo
    echo "========================================================"
    echo "[Aggregating 5-fold results for $TASK]"
    echo "========================================================"
    python3 utils/aggregate_kfold.py --task "$TASK" --log_base "$LOG_BASE"
else
    echo "Error: Invalid action '$ACTION'."; usage; exit 1
fi
