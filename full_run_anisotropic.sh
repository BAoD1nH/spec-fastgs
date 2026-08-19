

#1. Teapot
python extract_reflection_prior.py \
    -s datasets/Anisotropic-Synthesis/teapot \
    --eval \
    --white_background \
    --data_device cpu \
    --ref_prior_method tan \
    --ti_thresh 0.35 \
    --ti_bright 0.60

python train.py \
    -s datasets/Anisotropic-Synthesis/teapot \
    -m output/anisotropic_synthetic/teapot \
    --eval \
    --white_background \
    --asg_degree 12 \
    --densification_interval 500 \
    --densification_refscore_interval 500 \
    --num_score_cameras 10 \
    --optimizer_type default \
    --use_ref_score \
    --use_adaptive_prior \
    --use_reflection_view_sampling \
    --disable_multiview_contribution


python render.py -s datasets/Anisotropic-Synthesis/teapot -m output/anisotropic_synthetic/teapot --iteration 30000 --skip_train --data_device cpu

python metrics.py -m output/anisotropic_synthetic/teapot

# #2. record

# python render.py -s datasets/Anisotropic-Synthesis/record -m output/anisotropic_synthetic/record --iteration 30000 --skip_train --data_device cpu

# python metrics.py -m output/anisotropic_synthetic/record

# #3. plane
# python render.py -s ../spec-fastgs/datasets/Anisotropic-Synthesis/plane -m output/anisotropic_synthetic/plane --iteration 30000 --skip_train --data_device cpu

# python metrics.py -m output/anisotropic_synthetic/plane
