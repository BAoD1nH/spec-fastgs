#!/bin/bash

# ============================================================
# SPEC-FASTGS TEAPOT (SYNTHETIC) RUN SCRIPT
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/Ref-NeRF/refnerf
OUTPUT_ROOT=./output
SCENE=toaster

echo "========================================================================"
echo " Starting spec-fastgs Training Pipeline for Synthetic Dataset"
echo "========================================================================"
echo "Dataset Path : ${DATA_ROOT}/${SCENE}"
echo "Scene Name   : $SCENE"
echo "Output Path  : ${OUTPUT_ROOT}/${SCENE}"
echo "========================================================================"

# 0. EXTRACT REFLECTION PRIOR (Giai đoạn 1)
echo "[0/3] Running extract_reflection_prior.py..."
python extract_reflection_prior.py \
    -s ${DATA_ROOT}/${SCENE} \
    -r 1 \
    --sk_intensity 0.7 \
    --sk_saturation 0.2
    
# 1. TRAIN (Giai đoạn 2)
echo "[1/3] Running train.py..."
python train.py \
    -s ${DATA_ROOT}/${SCENE} \
    -m ${OUTPUT_ROOT}/${SCENE} \
    --eval \
    --white_background \
    --iterations 30000 \
    --densification_interval 500 \
    --optimizer_type default \
    --asg_degree 24 \
    --sh_degree 3 \
    --specular_start_iter 3000 \
    --densification_refscore_interval 3000 \
    --max_refscore_gaussians 150000 \
    --lambda_spec_reg 0.0 \
    --disable_ref_score

# 2. RENDER
echo "[2/3] Running render.py..."
python render.py \
    -m ${OUTPUT_ROOT}/${SCENE} \
    --skip_train

# 3. METRICS
echo "[3/3] Running metrics.py..."
python metrics.py \
    -m ${OUTPUT_ROOT}/${SCENE}

echo "========================================================================"
echo " Pipeline Completed Successfully!"
echo " Results and train_info.json are saved in ${OUTPUT_ROOT}/${SCENE}"
echo "========================================================================"
