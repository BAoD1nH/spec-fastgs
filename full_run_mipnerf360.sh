#!/bin/bash

# I. Indoor 
# 1. Counter
python extract_reflection_prior.py \
    -s datasets/mipnerf360/counter \
    -i images \
    -r 8 \
    --eval \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

python train.py \
    -s datasets/mipnerf360/counter \
    -m output/mipnerf360/counter \
    -i images \
    -r 8 \
    --eval \
    --is_real \
    --is_indoor \
    --asg_degree 12 \
    --densification_interval 100 \
    --densification_refscore_interval 500 \
    --num_score_cameras 10 \
    --highfeature_lr 0.02 \
    --grad_abs_thresh 0.0008 \
    --optimizer_type default \
    --use_ref_score \
    --use_adaptive_prior \
    --use_reflection_view_sampling

python render.py \
    -s datasets/mipnerf360/counter \
    -m output/mipnerf360/counter \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

python metrics.py \
    -m output/mipnerf360/counter
