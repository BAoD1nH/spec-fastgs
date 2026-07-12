#!/bin/bash

# ============================================================
# SPEC-FASTGS BIG RUN SCRIPT
# Minimal final-style config: train -> render -> metrics
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output
SCENE=counter
IMAGES=images_8

# Important final knobs
ASG_DEGREE=64
USE_REF_SCORE=True
USE_ADAPTIVE_PRIOR=True
USE_SH_SPEC_MASK=True

# Representation Capacity: light SH suppression on high-confidence specular areas.
SH_SPEC_GRAD_SCALE=0.75
SH_SPEC_MASK_START=8000
SH_SPEC_MASK_THRESHOLD=0.75
SH_SPEC_MIN_METRIC_COUNT=2

REF_SCORE_FLAG=""
if [ "$USE_REF_SCORE" = "True" ]; then
    REF_SCORE_FLAG="--use_ref_score"
fi

ADAPTIVE_PRIOR_FLAG=""
if [ "$USE_ADAPTIVE_PRIOR" = "True" ]; then
    ADAPTIVE_PRIOR_FLAG="--use_adaptive_prior"
fi

SH_SPEC_MASK_FLAG=""
if [ "$USE_SH_SPEC_MASK" = "True" ]; then
    SH_SPEC_MASK_FLAG="--use_sh_spec_mask"
fi

echo "========================================================================"
echo " Starting spec-fastgs BIG Training Pipeline"
echo "========================================================================"
echo "Dataset Path : ${DATA_ROOT}/${SCENE}"
echo "Images       : ${IMAGES}"
echo "Output Path  : ${OUTPUT_ROOT}/${SCENE}"
echo "ASG Degree   : ${ASG_DEGREE}"
echo "Use RefScore : ${USE_REF_SCORE}"
echo "AdaptivePrior: ${USE_ADAPTIVE_PRIOR}"
echo "SH Spec Mask : ${USE_SH_SPEC_MASK}"
echo "========================================================================"

# 1. TRAIN
python train.py \
    -s ${DATA_ROOT}/${SCENE} \
    -m ${OUTPUT_ROOT}/${SCENE} \
    -i ${IMAGES} \
    --eval \
    --iterations 30000 \
    --densification_interval 100 \
    --optimizer_type default \
    --asg_degree ${ASG_DEGREE} \
    --is_real \
    --is_indoor \
    --sh_degree 3 \
    --highfeature_lr 0.02 \
    --grad_abs_thresh 0.0004 \
    --specular_start_iter 3000 \
    --sh_spec_grad_scale ${SH_SPEC_GRAD_SCALE} \
    --sh_spec_mask_start ${SH_SPEC_MASK_START} \
    --sh_spec_mask_threshold ${SH_SPEC_MASK_THRESHOLD} \
    --sh_spec_min_metric_count ${SH_SPEC_MIN_METRIC_COUNT} \
    ${REF_SCORE_FLAG} \
    ${ADAPTIVE_PRIOR_FLAG} \
    ${SH_SPEC_MASK_FLAG}

# 2. RENDER
python render.py \
    -m ${OUTPUT_ROOT}/${SCENE} \
    --skip_train

# 3. METRICS
python metrics.py \
    -m ${OUTPUT_ROOT}/${SCENE}
