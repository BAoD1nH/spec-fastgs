#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=bicycle python train.py -s ./datasets/mipnerf360/bicycle -m ./output_spec_fastgs/bicycle -i images_2 --eval --densification_interval 500 --optimizer_type default --iterations 30000 --test_iterations 30000
# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=flowers python train.py -s ./datasets/mipnerf360/flowers -m ./output_spec_fastgs/flowers -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000
# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=garden python train.py -s ./datasets/mipnerf360/garden -m ./output_spec_fastgs/garden -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000
# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=stump python train.py -s ./datasets/mipnerf360/stump -m ./output_spec_fastgs/stump -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000
# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=treehill python train.py -s ./datasets/mipnerf360/treehill -m ./output_spec_fastgs/treehill -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000
# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=room python train.py -s ./datasets/mipnerf360/room -m ./output_spec_fastgs/room -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=counter python train.py -s ./datasets/mipnerf360/counter -m ./output_spec_fastgs/counter -i images_4  --eval --densification_interval 500 --optimizer_type default --iterations 30000 --test_iterations 30000
# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=kitchen python train.py -s ./datasets/mipnerf360/kitchen -m ./output_spec_fastgs/kitchen -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000
# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=bonsai python train.py -s ./datasets/mipnerf360/bonsai -m ./output_spec_fastgs/bonsai -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000

# CUDA_VISIBLE_DEVICES=0 python render.py -m ./output_spec_fastgs/bicycle --skip_train
# CUDA_VISIBLE_DEVICES=0 python render.py -m ./output_spec_fastgs/flowers --skip_train
# CUDA_VISIBLE_DEVICES=0 python render.py -m ./output_spec_fastgs/garden --skip_train
# CUDA_VISIBLE_DEVICES=0 python render.py -m ./output_spec_fastgs/stump --skip_train
# CUDA_VISIBLE_DEVICES=0 python render.py -m ./output_spec_fastgs/treehill --skip_train
# CUDA_VISIBLE_DEVICES=0 python render.py -m ./output_spec_fastgs/room --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m ./output_spec_fastgs/counter --skip_train
# CUDA_VISIBLE_DEVICES=0 python render.py -m ./output_spec_fastgs/kitchen --skip_train
# CUDA_VISIBLE_DEVICES=0 python render.py -m ./output_spec_fastgs/bonsai --skip_train

# CUDA_VISIBLE_DEVICES=0 python metrics.py -m ./output_spec_fastgs/bicycle
# CUDA_VISIBLE_DEVICES=0 python metrics.py -m ./output_spec_fastgs/flowers
# CUDA_VISIBLE_DEVICES=0 python metrics.py -m ./output_spec_fastgs/garden
# CUDA_VISIBLE_DEVICES=0 python metrics.py -m ./output_spec_fastgs/stump
# CUDA_VISIBLE_DEVICES=0 python metrics.py -m ./output_spec_fastgs/treehill
# CUDA_VISIBLE_DEVICES=0 python metrics.py -m ./output_spec_fastgs/room
CUDA_VISIBLE_DEVICES=0 python metrics.py -m ./output_spec_fastgs/counter
# CUDA_VISIBLE_DEVICES=0 python metrics.py -m ./output_spec_fastgs/kitchen
# CUDA_VISIBLE_DEVICES=0 python metrics.py -m ./output_spec_fastgs/bonsai