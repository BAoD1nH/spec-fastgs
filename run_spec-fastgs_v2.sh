#!/bin/bash

# ============================================================
# SPEC-FASTGS CLEAN RUN SCRIPT
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output_spec_fastgs


# ============================================================
# TRAIN
# ============================================================

SCENE=counter

python train.py \
    -s ${DATA_ROOT}/${SCENE} \
    -m ${OUTPUT_ROOT}/${SCENE} \
    -i images_4 \
    --eval \
    --iterations 30000 \
    --densification_interval 500 \
    --optimizer_type default \
    --asg_degree 24 \
    --is_real \
    --is_indoor \
    --sh_degree 0


# ============================================================
# RENDER
# ============================================================

python render.py \
    -m ${OUTPUT_ROOT}/${SCENE} \
    --iteration 30000 \
    --skip_train


# ============================================================
# METRICS
# ============================================================

python metrics.py \
    -m ${OUTPUT_ROOT}/${SCENE}

