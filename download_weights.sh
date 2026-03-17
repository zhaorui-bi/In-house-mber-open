#!/bin/bash
# Download all model weights required for mBER
# This includes AlphaFold2, NanoBodyBuilder2, and ESM models
#
# Usage: bash download_weights.sh [target_dir]
# Default target: $MBER_WEIGHTS_DIR or ~/.mber
#
# Options:
#   --skip-esm    Skip ESM model downloads (saves ~5GB, but ESM2 is required)
#   --with-esmfold  Also download ESMFold weights (~16GB extra)

set -e

# Parse arguments
MBER_DIR="${MBER_WEIGHTS_DIR:-${HOME}/.mber}"
SKIP_ESM=false
WITH_ESMFOLD=false

for arg in "$@"; do
    case $arg in
        --skip-esm)
            SKIP_ESM=true
            shift
            ;;
        --with-esmfold)
            WITH_ESMFOLD=true
            shift
            ;;
        *)
            if [[ ! "$arg" == -* ]]; then
                MBER_DIR="$arg"
            fi
            ;;
    esac
done

echo "=========================================="
echo "mBER Model Weights Downloader"
echo "=========================================="
echo "Weights root: ${MBER_DIR}"
echo ""

# ==========================================
# AlphaFold2 Weights
# ==========================================
AF_DIR="${MBER_AF_PARAMS_DIR:-${MBER_DIR}/af_params}"
AF_PARAMS_FILE="${AF_DIR}/alphafold_params_2022-12-06.tar"
AF_CHECK_FILE="${AF_DIR}/params_model_5_ptm.npz"
mkdir -p "${AF_DIR}"

echo "[1/4] AlphaFold2 Weights"
echo "----------------------------------------"

if [ -f "${AF_CHECK_FILE}" ]; then
    echo "✓ AlphaFold2 weights already exist"
else
    echo "Downloading AlphaFold2 weights (~3.5GB)..."
    wget -q --show-progress -O "${AF_PARAMS_FILE}" \
        "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
    
    echo "Verifying archive integrity..."
    if ! tar tf "${AF_PARAMS_FILE}" >/dev/null 2>&1; then
        echo "ERROR: Downloaded archive is corrupt. Please try again."
        rm -f "${AF_PARAMS_FILE}"
        exit 1
    fi
    
    echo "Extracting weights..."
    tar -xf "${AF_PARAMS_FILE}" -C "${AF_DIR}"
    
    if [ ! -f "${AF_CHECK_FILE}" ]; then
        echo "ERROR: Extraction failed. Expected file not found."
        exit 1
    fi
    
    echo "Cleaning up archive..."
    rm -f "${AF_PARAMS_FILE}"
    
    echo "✓ AlphaFold2 weights installed"
fi
echo ""

# ==========================================
# NanoBodyBuilder2 Weights
# ==========================================
NBB2_DIR="${MBER_NBB2_WEIGHTS_DIR:-${MBER_DIR}/nbb2_weights}"

# ==========================================
# HuggingFace / ESM Weights
# ==========================================
HF_DIR="${MBER_HF_HOME:-${HF_HOME:-${MBER_DIR}/huggingface}}"
export HF_HOME="${HF_DIR}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_DIR}/hub}"

echo "Resolved directories:"
echo "  AlphaFold2: ${AF_DIR}"
echo "  NanoBodyBuilder2: ${NBB2_DIR}"
echo "  HuggingFace / ESM: ${HF_DIR}"
echo ""

# Create directory structure
mkdir -p "${NBB2_DIR}"
mkdir -p "${HF_DIR}"
mkdir -p "${HF_HUB_CACHE}"

echo "[2/4] NanoBodyBuilder2 Weights"
echo "----------------------------------------"

NBB2_MODELS=(
    "nanobody_model_1:https://zenodo.org/record/7258553/files/nanobody_model_1?download=1"
    "nanobody_model_2:https://zenodo.org/record/7258553/files/nanobody_model_2?download=1"
    "nanobody_model_3:https://zenodo.org/record/7258553/files/nanobody_model_3?download=1"
    "nanobody_model_4:https://zenodo.org/record/7258553/files/nanobody_model_4?download=1"
)

all_exist=true
for model_entry in "${NBB2_MODELS[@]}"; do
    model_name="${model_entry%%:*}"
    model_path="${NBB2_DIR}/${model_name}"
    if [ ! -f "${model_path}" ] || [ ! -s "${model_path}" ]; then
        all_exist=false
        break
    fi
done

if [ "$all_exist" = true ]; then
    echo "✓ NanoBodyBuilder2 weights already exist"
else
    echo "Downloading NanoBodyBuilder2 weights (4 models)..."
    
    for model_entry in "${NBB2_MODELS[@]}"; do
        model_name="${model_entry%%:*}"
        model_url="${model_entry#*:}"
        model_path="${NBB2_DIR}/${model_name}"
        
        if [ -f "${model_path}" ] && [ -s "${model_path}" ]; then
            echo "  ✓ ${model_name} (exists)"
        else
            echo "  Downloading ${model_name}..."
            wget -q --show-progress -O "${model_path}" "${model_url}"
            
            if [ ! -s "${model_path}" ]; then
                echo "  ERROR: Failed to download ${model_name}"
                rm -f "${model_path}"
                exit 1
            fi
            echo "  ✓ ${model_name}"
        fi
    done
    
    echo "✓ NanoBodyBuilder2 weights installed"
fi
echo ""

echo "[3/4] ESM2 Model (HuggingFace)"
echo "----------------------------------------"

if [ "$SKIP_ESM" = true ]; then
    echo "⊘ Skipped (--skip-esm flag)"
else
    ESM2_CHECK="${HF_HUB_CACHE}/models--facebook--esm2_t33_650M_UR50D"
    
    if [ -d "${ESM2_CHECK}" ]; then
        echo "✓ ESM2 model already exists"
    else
        echo "Downloading ESM2 model (~5GB)..."
        echo "This may take several minutes..."
        
        # Use Python to download via HuggingFace
        python3 -c "
import os
os.environ['HF_HOME'] = '${HF_DIR}'
os.environ['HF_HUB_CACHE'] = '${HF_HUB_CACHE}'
from transformers import AutoTokenizer, EsmForMaskedLM
print('  Downloading tokenizer...')
AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D', use_safetensors=False, cache_dir='${HF_HUB_CACHE}')
print('  Downloading model weights...')
EsmForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D', use_safetensors=False, cache_dir='${HF_HUB_CACHE}')
print('  ✓ ESM2 model installed')
" 2>&1 | grep -v "^$"
    fi
fi
echo ""

# ==========================================
# ESMFold Model (Optional, HuggingFace)
# ==========================================
echo "[4/4] ESMFold Model (Optional)"
echo "----------------------------------------"

if [ "$WITH_ESMFOLD" = true ]; then
    ESMFOLD_CHECK="${HF_HUB_CACHE}/models--facebook--esmfold_v1"
    
    if [ -d "${ESMFOLD_CHECK}" ]; then
        echo "✓ ESMFold model already exists"
    else
        echo "Downloading ESMFold model (~16GB)..."
        echo "This may take a long time..."
        
        python3 -c "
import os
os.environ['HF_HOME'] = '${HF_DIR}'
os.environ['HF_HUB_CACHE'] = '${HF_HUB_CACHE}'
from transformers import EsmForProteinFolding
print('  Downloading ESMFold model...')
EsmForProteinFolding.from_pretrained('facebook/esmfold_v1', use_safetensors=False, cache_dir='${HF_HUB_CACHE}')
print('  ✓ ESMFold model installed')
" 2>&1 | grep -v "^$"
    fi
else
    echo "⊘ Skipped (use --with-esmfold to download)"
    echo "  Note: ESMFold is only needed if using esmfold for structure prediction."
    echo "  The default VHH protocol uses NanoBodyBuilder2 instead."
fi
echo ""

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Resolved directories:"
echo "  AlphaFold2: ${AF_DIR}"
echo "  NanoBodyBuilder2: ${NBB2_DIR}"
echo "  HuggingFace / ESM: ${HF_DIR}"
echo ""
echo "Directory sizes:"
du -sh "${AF_DIR}" "${NBB2_DIR}" "${HF_DIR}" 2>/dev/null | sed 's/^/  /'
echo ""
echo "Shell config example (~/.bashrc or ~/.zshrc):"
echo "  export MBER_WEIGHTS_DIR=${MBER_DIR}"
echo ""
echo "Per-model overrides are also supported:"
echo "  export MBER_AF_PARAMS_DIR=${AF_DIR}"
echo "  export MBER_NBB2_WEIGHTS_DIR=${NBB2_DIR}"
echo "  export MBER_HF_HOME=${HF_DIR}"
echo ""
echo "To use with Docker, mount the shared root directory:"
echo "  docker run --gpus all \\"
echo "    -v ${MBER_DIR}:/mber_weights:ro \\"
echo "    ..."
