#!/bin/bash
set -euo pipefail

# Theta-mBER Docker Entrypoint
# Uses MBER_WEIGHTS_DIR as the single runtime weight root (default: /mber_weights)

WEIGHTS_DIR="${MBER_WEIGHTS_DIR:-/mber_weights}"
export MBER_WEIGHTS_DIR="${WEIGHTS_DIR}"
export MBER_HF_HOME="${MBER_HF_HOME:-${WEIGHTS_DIR}/huggingface}"
export HF_HOME="${MBER_HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
unset TRANSFORMERS_CACHE

# Show help if no arguments provided
if [ $# -eq 0 ]; then
    echo "Theta-mBER Docker Container"
    echo ""
    echo "Usage: docker run --gpus all [mounts] theta-mber:latest [options]"
    echo ""
    echo "Required options (if not using --settings):"
    echo "  --input-pdb PATH      Target PDB file"
    echo "  --output-dir PATH     Output directory"
    echo "  --chains CHAINS       Target chains (e.g., 'A' or 'A,B')"
    echo ""
    echo "Common options:"
    echo "  --settings PATH       Use YAML settings file"
    echo "  --hotspots RESIDUES   Target residues (e.g., 'A56')"
    echo "  --num-accepted N      Designs to generate (default: 100)"
    echo "  --help                Show full help"
    echo ""
    echo "Weights:"
    echo "  Mount a complete weight root at /mber_weights, for example:"
    echo "    -v /scratch2/mz32/share/mber_weights:/mber_weights:ro"
    echo ""
    echo "Example:"
    echo "  docker run --gpus all \\"
    echo "    -v \$(pwd)/output:/outputs \\"
    echo "    -v \$(pwd)/inputs:/inputs:ro \\"
    echo "    -v /path/to/mber_weights:/mber_weights:ro \\"
    echo "    theta-mber:latest \\"
    echo "    --input-pdb /inputs/target.pdb \\"
    echo "    --output-dir /outputs/my_run \\"
    echo "    --chains A"
    echo ""
    exit 0
fi

echo "=== Theta-mBER Docker Container ==="
echo "Setting up model weights..."
echo "  MBER_WEIGHTS_DIR=${WEIGHTS_DIR}"
echo "  HF_HOME=${HF_HOME}"

mkdir -p "${WEIGHTS_DIR}"

check_weights_exist() {
    local base_path=$1
    [ -f "${base_path}/af_params/params_model_5_ptm.npz" ] && \
    [ -f "${base_path}/nbb2_weights/nanobody_model_1" ] && \
    [ -d "${base_path}/huggingface/hub/models--facebook--esm2_t33_650M_UR50D" ]
}

if check_weights_exist "${WEIGHTS_DIR}"; then
    echo "Using weights from ${WEIGHTS_DIR}"
    echo "  ✓ AlphaFold2"
    echo "  ✓ NanoBodyBuilder2"
    echo "  ✓ ESM2"
else
    echo "Required weights missing under ${WEIGHTS_DIR}."
    echo "Downloading missing weights (this may take several minutes)..."
    bash /app/download_weights.sh "${WEIGHTS_DIR}"
fi

echo ""
echo "Starting mBER VHH binder design..."
echo ""

# Execute mber-vhh with all arguments passed to the container
exec conda run --no-capture-output -n mber mber-vhh "$@"
