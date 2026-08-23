#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="spec_fastgs"
MODE="packed"
CACHE_DIR="${REPO_DIR}/.cache/environment"
ARCHIVE_URL="https://drive.google.com/file/d/1SseChaO4fvjW5eo0VcmuJv5bupVqJhjH/view?usp=sharing"
ARCHIVE_PATH=""
SKIP_VERIFY=0

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_environment.sh [options]

Options:
  --mode packed|build   Install the prepared archive or build environment.yml
                        (default: packed).
  --name NAME           Conda environment name (default: spec_fastgs).
  --archive PATH        Use a local packed environment instead of downloading.
  --cache-dir PATH      Download cache directory.
  --skip-verify         Do not run the final environment verification.
  -h, --help            Show this help.

The script never deletes or overwrites an existing Conda environment.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:?Missing value for --mode}"
            shift 2
            ;;
        --name)
            ENV_NAME="${2:?Missing value for --name}"
            shift 2
            ;;
        --archive)
            ARCHIVE_PATH="${2:?Missing value for --archive}"
            shift 2
            ;;
        --cache-dir)
            CACHE_DIR="${2:?Missing value for --cache-dir}"
            shift 2
            ;;
        --skip-verify)
            SKIP_VERIFY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${MODE}" != "packed" && "${MODE}" != "build" ]]; then
    echo "--mode must be 'packed' or 'build'." >&2
    exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found. Install Miniconda, then run 'conda init bash'." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
ENV_DIR="${CONDA_BASE}/envs/${ENV_NAME}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}" || [[ -d "${ENV_DIR}" ]]; then
    cat >&2 <<EOF
Environment '${ENV_NAME}' already exists at ${ENV_DIR}.
It was left unchanged. To replace it, remove it explicitly first:
  conda env remove -n ${ENV_NAME}
EOF
    exit 1
fi

download_archive() {
    mkdir -p "${CACHE_DIR}"
    local destination="${CACHE_DIR}/spec-fastgs-env.tar.gz"
    local partial="${destination}.part"

    if [[ -s "${destination}" ]]; then
        echo "[cache] Using ${destination}"
        ARCHIVE_PATH="${destination}"
        return
    fi

    local downloader_python="${CONDA_BASE}/bin/python"
    if ! "${downloader_python}" -c 'import gdown' >/dev/null 2>&1; then
        echo "[setup] Installing gdown into the Conda base environment..."
        "${downloader_python}" -m pip install "gdown==4.7.3"
    fi

    echo "[download] Prepared Spec-FastGS environment"
    rm -f "${partial}"
    "${downloader_python}" -m gdown --fuzzy "${ARCHIVE_URL}" -O "${partial}"
    [[ -s "${partial}" ]] || { echo "Environment download failed." >&2; exit 1; }
    mv "${partial}" "${destination}"
    ARCHIVE_PATH="${destination}"
}

install_packed_environment() {
    if [[ -z "${ARCHIVE_PATH}" ]]; then
        download_archive
    else
        ARCHIVE_PATH="$(cd "$(dirname "${ARCHIVE_PATH}")" && pwd)/$(basename "${ARCHIVE_PATH}")"
    fi

    [[ -f "${ARCHIVE_PATH}" ]] || { echo "Archive not found: ${ARCHIVE_PATH}" >&2; exit 1; }
    tar -tf "${ARCHIVE_PATH}" >/dev/null 2>&1 || {
        echo "The downloaded file is not a valid TAR archive: ${ARCHIVE_PATH}" >&2
        exit 1
    }

    local unpack_dir
    local packed_root
    unpack_dir="$(mktemp -d "${CACHE_DIR}/unpack.XXXXXX")"
    trap 'rm -rf "${unpack_dir}"' RETURN

    echo "[extract] ${ARCHIVE_PATH}"
    tar -xf "${ARCHIVE_PATH}" -C "${unpack_dir}"

    if [[ -x "${unpack_dir}/bin/python" ]]; then
        packed_root="${unpack_dir}"
    else
        mapfile -t python_candidates < <(find "${unpack_dir}" -mindepth 3 -maxdepth 3 -type f -path '*/bin/python')
        if [[ "${#python_candidates[@]}" -ne 1 ]]; then
            echo "Could not locate a unique Conda environment inside the archive." >&2
            exit 1
        fi
        packed_root="$(cd "$(dirname "${python_candidates[0]}")/.." && pwd)"
    fi

    echo "[install] ${packed_root} -> ${ENV_DIR}"
    mkdir -p "${ENV_DIR}"
    cp -a "${packed_root}/." "${ENV_DIR}/"

    if [[ -x "${ENV_DIR}/bin/conda-unpack" ]]; then
        "${ENV_DIR}/bin/conda-unpack"
    else
        echo "WARNING: conda-unpack was not found; archive may not be from conda-pack." >&2
    fi

    rm -rf "${unpack_dir}"
    trap - RETURN
}

install_built_environment() {
    command -v nvcc >/dev/null 2>&1 || {
        echo "nvcc was not found. Install a CUDA toolkit before building extensions." >&2
        exit 1
    }

    echo "[conda] Creating ${ENV_NAME} from environment.yml"
    conda env create -n "${ENV_NAME}" -f "${REPO_DIR}/environment.yml"

    local env_python="${ENV_DIR}/bin/python"
    local nvcc_path
    nvcc_path="$(command -v nvcc)"
    export CUDA_HOME="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"

    echo "[build] Compiling CUDA extensions with CUDA_HOME=${CUDA_HOME}"
    "${env_python}" -m pip install --no-build-isolation "${REPO_DIR}/submodules/simple-knn"
    "${env_python}" -m pip install --no-build-isolation "${REPO_DIR}/submodules/diff-gaussian-rasterization_fastgs"
    "${env_python}" -m pip install --no-build-isolation "${REPO_DIR}/submodules/fused-ssim"
}

cd "${REPO_DIR}"
if [[ "${MODE}" == "packed" ]]; then
    install_packed_environment
else
    install_built_environment
fi

if [[ "${SKIP_VERIFY}" -eq 0 ]]; then
    "${ENV_DIR}/bin/python" "${SCRIPT_DIR}/verify_environment.py"
fi

cat <<EOF

Environment setup completed.

Activate it with:
  conda activate ${ENV_NAME}

Then prepare data and train:
  python dataset_preparation.py
  bash full_run_mipnerf360.sh
EOF
