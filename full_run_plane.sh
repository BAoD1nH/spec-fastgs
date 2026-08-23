python extract_reflection_prior.py \
    -s datasets/anisotropic-synthetic/plane \
    --eval \
    --white_background \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

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

python render.py \
    -s datasets/anisotropic-synthetic/plane \
    -m output/anisotropic_synthetic/plane \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

python metrics.py \
    -m output/anisotropic_synthetic/plane