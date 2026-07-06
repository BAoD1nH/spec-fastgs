#!/bin/bash

# ============================================================
# SPEC-FASTGS TEAPOT (SYNTHETIC) RUN SCRIPT
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/Ref-NeRF/refnerf
OUTPUT_ROOT=./output
SCENE=toaster
ASG_DEGREE=24
USE_REF_SCORE=False
EXTRACT_REF_PRIOR=False
BACKUP_REF_PRIOR=True
NUM_SCORE_CAMERAS=10
FULL_ASG_INTERVAL=0
F_REST_WARMUP_UNTIL=0
F_REST_INTERVAL_EARLY=16
F_REST_INTERVAL_MID=32
F_REST_INTERVAL_LATE=64
REF_PRIOR_METHOD=tan
TI_THRESH=0.35
TI_BRIGHT=0.6
SK_INTENSITY=0.7
SK_SATURATION=0.2

REF_SCORE_FLAG=""
if [ "$USE_REF_SCORE" = "True" ]; then
    REF_SCORE_FLAG="--use_ref_score"
fi

echo "========================================================================"
echo " Starting spec-fastgs Training Pipeline for Synthetic Dataset"
echo "========================================================================"
echo "Dataset Path : ${DATA_ROOT}/${SCENE}"
echo "Scene Name   : $SCENE"
echo "Output Path  : ${OUTPUT_ROOT}/${SCENE}"
echo "========================================================================"

# 0. EXTRACT REFLECTION PRIOR
if [ "$USE_REF_SCORE" = "True" ] && [ "$EXTRACT_REF_PRIOR" = "True" ]; then
    echo "[0/4] Running extract_reflection_prior.py..."
    PRIOR_DIR=${DATA_ROOT}/${SCENE}/reflection_prior
    if [ "$BACKUP_REF_PRIOR" = "True" ] && [ -d "$PRIOR_DIR" ]; then
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        BACKUP_ROOT=${DATA_ROOT}/${SCENE}/backups
        BACKUP_DIR=${BACKUP_ROOT}/reflection_prior_${TIMESTAMP}
        echo "Backing up existing reflection_prior to:"
        echo "  ${BACKUP_DIR}"
        mkdir -p "$BACKUP_ROOT"
        mv "$PRIOR_DIR" "$BACKUP_DIR"
    fi
    python extract_reflection_prior.py \
        -s ${DATA_ROOT}/${SCENE} \
        --ref_prior_method ${REF_PRIOR_METHOD} \
        --ti_thresh ${TI_THRESH} \
        --ti_bright ${TI_BRIGHT} \
        --sk_intensity ${SK_INTENSITY} \
        --sk_saturation ${SK_SATURATION}
else
    echo "[0/4] Skipping extract_reflection_prior.py"
fi

# 1. TRAIN (Giai đoạn 1)
echo "[1/4] Running train.py..."
python train.py \
    -s ${DATA_ROOT}/${SCENE} \
    -m ${OUTPUT_ROOT}/${SCENE} \
    --eval \
    --white_background \
    --iterations 30000 \
    --densification_interval 500 \
    --optimizer_type default \
    --asg_degree ${ASG_DEGREE} \
    --sh_degree 3 \
    --specular_start_iter 3000 \
    --num_score_cameras ${NUM_SCORE_CAMERAS} \
    --full_asg_interval ${FULL_ASG_INTERVAL} \
    --f_rest_warmup_until ${F_REST_WARMUP_UNTIL} \
    --f_rest_interval_early ${F_REST_INTERVAL_EARLY} \
    --f_rest_interval_mid ${F_REST_INTERVAL_MID} \
    --f_rest_interval_late ${F_REST_INTERVAL_LATE} \
    --ref_prior_method ${REF_PRIOR_METHOD} \
    --ti_thresh ${TI_THRESH} \
    --ti_bright ${TI_BRIGHT} \
    --sk_intensity ${SK_INTENSITY} \
    --sk_saturation ${SK_SATURATION} \
    ${REF_SCORE_FLAG}

# 2. RENDER (Giai đoạn 2)
echo "[2/4] Running render.py..."
python render.py \
    -m ${OUTPUT_ROOT}/${SCENE} \
    --skip_train

# 3. METRICS (Giai đoạn 3)
echo "[3/4] Running metrics.py..."
python metrics.py \
    -m ${OUTPUT_ROOT}/${SCENE}

echo "========================================================================"
echo " Pipeline Completed Successfully!"
echo " Results and train_info.json are saved in ${OUTPUT_ROOT}/${SCENE}"
echo "========================================================================"
