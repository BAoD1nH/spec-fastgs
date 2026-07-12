#!/bin/bash

# ============================================================
# SPEC-FASTGS SHINY / REF-NERF RUN SCRIPT
# Minimal synthetic config: train -> render -> metrics
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/Ref-NeRF/refnerf
OUTPUT_ROOT=./output
SCENE=toaster

# Important final knobs
ASG_DEGREE=64
USE_REF_SCORE=True
USE_ADAPTIVE_PRIOR=True
USE_SH_SPEC_MASK=True

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
echo " Starting spec-fastgs Training Pipeline for Synthetic Dataset"
echo "========================================================================"
echo "Dataset Path : ${DATA_ROOT}/${SCENE}"
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
    --eval \
    --white_background \
    --iterations 30000 \
    --densification_interval 500 \
    --optimizer_type default \
    --asg_degree ${ASG_DEGREE} \
    --sh_degree 3 \
    --specular_start_iter 3000 \
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

echo "========================================================================"
echo " Pipeline Completed Successfully!"
echo " Results and train_info.json are saved in ${OUTPUT_ROOT}/${SCENE}"
echo "========================================================================"
