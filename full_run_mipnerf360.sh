#!/usr/bin/env bash
set -e

# Indoor scenes use images_2; outdoor scenes use images_4.

# =============================================================================
# 1. EXTRACT REFLECTION PRIOR FOR ALL SCENES
# =============================================================================

# 1. Bonsai
python extract_reflection_prior.py \
    -s datasets/mipnerf360/bonsai \
    -i images_2 \
    --eval \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# 2. Counter
python extract_reflection_prior.py \
    -s datasets/mipnerf360/counter \
    -i images_2 \
    --eval \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# 3. Kitchen
python extract_reflection_prior.py \
    -s datasets/mipnerf360/kitchen \
    -i images_2 \
    --eval \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# 4. Room
python extract_reflection_prior.py \
    -s datasets/mipnerf360/room \
    -i images_2 \
    --eval \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# 5. Bicycle
python extract_reflection_prior.py \
    -s datasets/mipnerf360/bicycle \
    -i images_4 \
    --eval \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# 6. Flowers
python extract_reflection_prior.py \
    -s datasets/mipnerf360/flowers \
    -i images_4 \
    --eval \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# 7. Garden
python extract_reflection_prior.py \
    -s datasets/mipnerf360/garden \
    -i images_4 \
    --eval \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# 8. Stump
python extract_reflection_prior.py \
    -s datasets/mipnerf360/stump \
    -i images_4 \
    --eval \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# 9. Treehill
python extract_reflection_prior.py \
    -s datasets/mipnerf360/treehill \
    -i images_4 \
    --eval \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

# =============================================================================
# 2. TRAIN ALL SCENES
# =============================================================================

# 1. Bonsai
python train.py \
    -s datasets/mipnerf360/bonsai \
    -m output/mipnerf360/bonsai \
    -i images_2 \
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
    # --disable_asg
    # --disable_multiview_contribution

# 2. Counter
python train.py \
    -s datasets/mipnerf360/counter \
    -m output/mipnerf360/counter \
    -i images_2 \
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
    # --disable_asg
    # --disable_multiview_contribution

# 3. Kitchen
python train.py \
    -s datasets/mipnerf360/kitchen \
    -m output/mipnerf360/kitchen \
    -i images_2 \
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
    # --disable_asg
    # --disable_multiview_contribution

# 4. Room
python train.py \
    -s datasets/mipnerf360/room \
    -m output/mipnerf360/room \
    -i images_2 \
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
    # --disable_asg
    # --disable_multiview_contribution

# 5. Bicycle
python train.py \
    -s datasets/mipnerf360/bicycle \
    -m output/mipnerf360/bicycle \
    -i images_4 \
    --eval \
    --is_real \
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
    # --disable_asg
    # --disable_multiview_contribution

# 6. Flowers
python train.py \
    -s datasets/mipnerf360/flowers \
    -m output/mipnerf360/flowers \
    -i images_4 \
    --eval \
    --is_real \
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
    # --disable_asg
    # --disable_multiview_contribution

# 7. Garden
python train.py \
    -s datasets/mipnerf360/garden \
    -m output/mipnerf360/garden \
    -i images_4 \
    --eval \
    --is_real \
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
    # --disable_asg
    # --disable_multiview_contribution

# 8. Stump
python train.py \
    -s datasets/mipnerf360/stump \
    -m output/mipnerf360/stump \
    -i images_4 \
    --eval \
    --is_real \
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
    # --disable_asg
    # --disable_multiview_contribution

# 9. Treehill
python train.py \
    -s datasets/mipnerf360/treehill \
    -m output/mipnerf360/treehill \
    -i images_4 \
    --eval \
    --is_real \
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
    # --disable_asg
    # --disable_multiview_contribution

# =============================================================================
# 3. RENDER ALL SCENES
# =============================================================================

# 1. Bonsai
python render.py \
    -s datasets/mipnerf360/bonsai \
    -m output/mipnerf360/bonsai \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# 2. Counter
python render.py \
    -s datasets/mipnerf360/counter \
    -m output/mipnerf360/counter \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# 3. Kitchen
python render.py \
    -s datasets/mipnerf360/kitchen \
    -m output/mipnerf360/kitchen \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# 4. Room
python render.py \
    -s datasets/mipnerf360/room \
    -m output/mipnerf360/room \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# 5. Bicycle
python render.py \
    -s datasets/mipnerf360/bicycle \
    -m output/mipnerf360/bicycle \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# 6. Flowers
python render.py \
    -s datasets/mipnerf360/flowers \
    -m output/mipnerf360/flowers \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# 7. Garden
python render.py \
    -s datasets/mipnerf360/garden \
    -m output/mipnerf360/garden \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# 8. Stump
python render.py \
    -s datasets/mipnerf360/stump \
    -m output/mipnerf360/stump \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# 9. Treehill
python render.py \
    -s datasets/mipnerf360/treehill \
    -m output/mipnerf360/treehill \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

# =============================================================================
# 4. COMPUTE METRICS FOR ALL SCENES
# =============================================================================

# 1. Bonsai
python metrics.py \
    -m output/mipnerf360/bonsai

# 2. Counter
python metrics.py \
    -m output/mipnerf360/counter

# 3. Kitchen
python metrics.py \
    -m output/mipnerf360/kitchen

# 4. Room
python metrics.py \
    -m output/mipnerf360/room

# 5. Bicycle
python metrics.py \
    -m output/mipnerf360/bicycle

# 6. Flowers
python metrics.py \
    -m output/mipnerf360/flowers

# 7. Garden
python metrics.py \
    -m output/mipnerf360/garden

# 8. Stump
python metrics.py \
    -m output/mipnerf360/stump

# 9. Treehill
python metrics.py \
    -m output/mipnerf360/treehill
