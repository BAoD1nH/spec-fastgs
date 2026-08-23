#!/usr/bin/env bash
set -Eeuo pipefail

DATASET_PATH="datasets/Anisotropic-Synthesis/teapot"
OUTPUT_PATH="output/anisotropic_synthetic/teapot"
PRIOR_TRAIN_DIR="${DATASET_PATH}/reflection_prior/train"
EXPECTED_TRAIN_VIEWS=600

count_prior_files() {
    local pattern="$1"
    find "${PRIOR_TRAIN_DIR}" -maxdepth 1 -type f -name "${pattern}" 2>/dev/null | wc -l
}

score_count="$(count_prior_files '*_ref_score.png')"
conf_count="$(count_prior_files '*_ref_conf.png')"

if [[ "${score_count}" -eq "${EXPECTED_TRAIN_VIEWS}" && \
      "${conf_count}" -eq "${EXPECTED_TRAIN_VIEWS}" ]]; then
    echo "Reflection priors already complete (${score_count} train views); skipping extraction."
else
    echo "Reflection priors incomplete (score=${score_count}, confidence=${conf_count}); extracting..."
    python extract_reflection_prior.py \
        -s "${DATASET_PATH}" \
        --eval \
        --white_background \
        --data_device cpu \
        --ref_prior_method tan \
        --ti_thresh 0.35 \
        --ti_bright 0.60

    score_count="$(count_prior_files '*_ref_score.png')"
    conf_count="$(count_prior_files '*_ref_conf.png')"
    if [[ "${score_count}" -ne "${EXPECTED_TRAIN_VIEWS}" || \
          "${conf_count}" -ne "${EXPECTED_TRAIN_VIEWS}" ]]; then
        echo "ERROR: Reflection-prior extraction is incomplete (score=${score_count}, confidence=${conf_count})." >&2
        exit 1
    fi
fi

python train.py \
    -s "${DATASET_PATH}" \
    -m "${OUTPUT_PATH}" \
    --eval \
    --white_background \
    --asg_degree 24 \
    --densification_interval 500 \
    --densification_refscore_interval 500 \
    --num_score_cameras 10 \
    --optimizer_type default \
    --use_ref_score

CHECKPOINT="${OUTPUT_PATH}/point_cloud/iteration_30000/point_cloud.ply"
if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "ERROR: Training ended without the expected checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

python render.py \
    -s "${DATASET_PATH}" \
    -m "${OUTPUT_PATH}" \
    --iteration 30000 \
    --skip_train \
    --data_device cpu

python metrics.py -m "${OUTPUT_PATH}"
