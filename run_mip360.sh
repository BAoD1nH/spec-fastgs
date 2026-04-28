CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=bicycle python train.py -s ./datasets/mipnerf360/bicycle -i images_8 --eval --densification_interval 500  --optimizer_type default --iterations 5000 --test_iterations 5000  --grad_abs_thresh 0.0012
CUDA_VISIBLE_DEVICES=0 python render.py -m output/bicycle --skip_train
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/bicycle