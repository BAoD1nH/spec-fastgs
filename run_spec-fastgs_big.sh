#!/bin/bash

# ============================================================
# SPEC-FASTGS BIG RUN SCRIPT
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output
SCENE=counter
IMAGES=images_8

# 0. EXTRACT REFLECTION PRIOR
echo "[0/3] Running extract_reflection_prior.py..."
python extract_reflection_prior.py \
    -s ${DATA_ROOT}/${SCENE} \
    -i ${IMAGES} \
    --sk_intensity 0.7 \
    --sk_saturation 0.2

# 1. TRAIN
echo "[1/3] Running train.py..."
python train.py \
    -s ${DATA_ROOT}/${SCENE} \
    -m ${OUTPUT_ROOT}/${SCENE} \
    -i ${IMAGES} \
    --eval \
    --iterations 30000 \
    --densification_interval 100 \
    --optimizer_type default \
    --asg_degree 24 \
    --is_real \
    --is_indoor \
    --sh_degree 3 \
    --highfeature_lr 0.02 \
    --grad_abs_thresh 0.0004 \
    --specular_start_iter 3000 \
    --densification_refscore_interval 3000 \
    --max_refscore_gaussians 200000 \
    --lambda_spec_reg 0.01 \
    --disable_ref_score
# 2. RENDER
python render.py \
    -m ${OUTPUT_ROOT}/${SCENE} \
    --skip_train

# 3. METRICS
python metrics.py \
    -m ${OUTPUT_ROOT}/${SCENE}
