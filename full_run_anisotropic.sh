#!/usr/bin/env bash
set -e

# =============================================================================
# 1. EXTRACT REFLECTION PRIOR FOR ALL SCENES
# =============================================================================

# 1. Ashtray
python extract_reflection_prior.py \
    -s datasets/anisotropic-synthetic/ashtray \
    --eval \
    --white_background \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# 2. Dishes
python extract_reflection_prior.py \
    -s datasets/anisotropic-synthetic/dishes \
    --eval \
    --white_background \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# 3. Headphone
python extract_reflection_prior.py \
    -s datasets/anisotropic-synthetic/headphone \
    --eval \
    --white_background \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# 4. Jupyter
python extract_reflection_prior.py \
    -s datasets/anisotropic-synthetic/jupyter \
    --eval \
    --white_background \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# 5. Lock
python extract_reflection_prior.py \
    -s datasets/anisotropic-synthetic/lock \
    --eval \
    --white_background \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# 6. Plane
python extract_reflection_prior.py \
    -s datasets/anisotropic-synthetic/plane \
    --eval \
    --white_background \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# 7. Record
python extract_reflection_prior.py \
    -s datasets/anisotropic-synthetic/record \
    --eval \
    --white_background \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# 8. Teapot
python extract_reflection_prior.py \
    -s datasets/anisotropic-synthetic/teapot \
    --eval \
    --white_background \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# =============================================================================
# 2. TRAIN ALL SCENES
# =============================================================================

# 1. Ashtray
python train.py \
    -s datasets/anisotropic-synthetic/ashtray \
    -m output/anisotropic_synthetic/ashtray \
    --eval \
    --white_background \
    --asg_degree 24 \
    --densification_interval 100 \
    --densification_refscore_interval 500 \
    --num_score_cameras 10 \
    --highfeature_lr 0.02 \
    --grad_abs_thresh 0.0008 \
    --optimizer_type default \
    --use_ref_score \
    --use_adaptive_prior
    # --use_reflection_view_sampling
    # --disable_asg
    # --disable_multiview_contribution

# 2. Dishes
python train.py \
    -s datasets/anisotropic-synthetic/dishes \
    -m output/anisotropic_synthetic/dishes \
    --eval \
    --white_background \
    --asg_degree 24 \
    --densification_interval 100 \
    --densification_refscore_interval 500 \
    --num_score_cameras 10 \
    --highfeature_lr 0.02 \
    --grad_abs_thresh 0.0008 \
    --optimizer_type default \
    --use_ref_score \
    --use_adaptive_prior
    # --use_reflection_view_sampling
    # --disable_asg
    # --disable_multiview_contribution

# 3. Headphone
python train.py \
    -s datasets/anisotropic-synthetic/headphone \
    -m output/anisotropic_synthetic/headphone \
    --eval \
    --white_background \
    --asg_degree 24 \
    --densification_interval 100 \
    --densification_refscore_interval 500 \
    --num_score_cameras 10 \
    --highfeature_lr 0.02 \
    --grad_abs_thresh 0.0008 \
    --optimizer_type default \
    --use_ref_score \
    --use_adaptive_prior
    # --use_reflection_view_sampling
    # --disable_asg
    # --disable_multiview_contribution

# 4. Jupyter
python train.py \
    -s datasets/anisotropic-synthetic/jupyter \
    -m output/anisotropic_synthetic/jupyter \
    --eval \
    --white_background \
    --asg_degree 24 \
    --densification_interval 100 \
    --densification_refscore_interval 500 \
    --num_score_cameras 10 \
    --highfeature_lr 0.02 \
    --grad_abs_thresh 0.0008 \
    --optimizer_type default \
    --use_ref_score \
    --use_adaptive_prior
    # --use_reflection_view_sampling
    # --disable_asg
    # --disable_multiview_contribution

# 5. Lock
python train.py \
    -s datasets/anisotropic-synthetic/lock \
    -m output/anisotropic_synthetic/lock \
    --eval \
    --white_background \
    --asg_degree 24 \
    --densification_interval 100 \
    --densification_refscore_interval 500 \
    --num_score_cameras 10 \
    --highfeature_lr 0.02 \
    --grad_abs_thresh 0.0008 \
    --optimizer_type default \
    --use_ref_score \
    --use_adaptive_prior
    # --use_reflection_view_sampling
    # --disable_asg
    # --disable_multiview_contribution

# 6. Plane
python train.py \
    -s datasets/anisotropic-synthetic/plane \
    -m output/anisotropic_synthetic/plane \
    --eval \
    --white_background \
    --asg_degree 24 \
    --densification_interval 100 \
    --densification_refscore_interval 500 \
    --num_score_cameras 10 \
    --highfeature_lr 0.02 \
    --grad_abs_thresh 0.0008 \
    --optimizer_type default \
    --use_ref_score \
    --use_adaptive_prior
    # --use_reflection_view_sampling
    # --disable_asg
    # --disable_multiview_contribution

# 7. Record
python train.py \
    -s datasets/anisotropic-synthetic/record \
    -m output/anisotropic_synthetic/record \
    --eval \
    --white_background \
    --asg_degree 24 \
    --densification_interval 100 \
    --densification_refscore_interval 500 \
    --num_score_cameras 10 \
    --highfeature_lr 0.02 \
    --grad_abs_thresh 0.0008 \
    --optimizer_type default \
    --use_ref_score \
    --use_adaptive_prior
    # --use_reflection_view_sampling
    # --disable_asg
    # --disable_multiview_contribution

# 8. Teapot
python train.py \
    -s datasets/anisotropic-synthetic/teapot \
    -m output/anisotropic_synthetic/teapot \
    --eval \
    --white_background \
    --asg_degree 24 \
    --densification_interval 100 \
    --densification_refscore_interval 500 \
    --num_score_cameras 10 \
    --highfeature_lr 0.02 \
    --grad_abs_thresh 0.0008 \
    --optimizer_type default \
    --use_ref_score \
    --use_adaptive_prior
    # --use_reflection_view_sampling
    # --disable_asg
    # --disable_multiview_contribution

# =============================================================================
# 3. RENDER ALL SCENES
# =============================================================================

# 1. Ashtray
python render.py \
    -s datasets/anisotropic-synthetic/ashtray \
    -m output/anisotropic_synthetic/ashtray \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# 2. Dishes
python render.py \
    -s datasets/anisotropic-synthetic/dishes \
    -m output/anisotropic_synthetic/dishes \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# 3. Headphone
python render.py \
    -s datasets/anisotropic-synthetic/headphone \
    -m output/anisotropic_synthetic/headphone \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# 4. Jupyter
python render.py \
    -s datasets/anisotropic-synthetic/jupyter \
    -m output/anisotropic_synthetic/jupyter \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# 5. Lock
python render.py \
    -s datasets/anisotropic-synthetic/lock \
    -m output/anisotropic_synthetic/lock \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# 6. Plane
python render.py \
    -s datasets/anisotropic-synthetic/plane \
    -m output/anisotropic_synthetic/plane \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# 7. Record
python render.py \
    -s datasets/anisotropic-synthetic/record \
    -m output/anisotropic_synthetic/record \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# 8. Teapot
python render.py \
    -s datasets/anisotropic-synthetic/teapot \
    -m output/anisotropic_synthetic/teapot \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# =============================================================================
# 4. COMPUTE METRICS FOR ALL SCENES
# =============================================================================

# 1. Ashtray
python metrics.py \
    -m output/anisotropic_synthetic/ashtray

# 2. Dishes
python metrics.py \
    -m output/anisotropic_synthetic/dishes

# 3. Headphone
python metrics.py \
    -m output/anisotropic_synthetic/headphone

# 4. Jupyter
python metrics.py \
    -m output/anisotropic_synthetic/jupyter

# 5. Lock
python metrics.py \
    -m output/anisotropic_synthetic/lock

# 6. Plane
python metrics.py \
    -m output/anisotropic_synthetic/plane

# 7. Record
python metrics.py \
    -m output/anisotropic_synthetic/record

# 8. Teapot
python metrics.py \
    -m output/anisotropic_synthetic/teapot
