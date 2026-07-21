# Theta-mBER Docker

Run Theta Team internal mBER VHH binder design in a containerized environment with GPU support.

## Prerequisites

- **NVIDIA GPU** with CUDA support (tested on A10G, A100, L4, L40S, H100)
- **Docker** (version 19.03+)
- **NVIDIA Container Toolkit** - [Installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **Disk space**: ~30GB (22GB image + 9GB for weights)

Verify your setup:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. Build the Image

From the repository root (takes 5-10 minutes):

```bash
docker build -t theta-mber:latest -f docker/Dockerfile .
```

**Or build with weights included** (~30GB image, no mounting needed):

```bash
docker build -t theta-mber:with-weights -f docker/Dockerfile --build-arg INCLUDE_WEIGHTS=true .
```

### 2. Run the PDL1 Example

```bash
# Create output directory
mkdir -p output

# Recommended: mount the shared Theta weight root
docker run --gpus all \
  -v $(pwd)/output:/outputs \
  -v $(pwd)/protocols/src/mber_protocols/stable/VHH_binder_design/examples:/inputs:ro \
  -v /scratch2/mz32/share/mber_weights:/mber_weights:ro \
  -e MBER_WEIGHTS_DIR=/mber_weights \
  theta-mber:latest \
  --input-pdb /inputs/PDL1.pdb \
  --output-dir /outputs/pdl1_test \
  --chains A \
  --hotspots A56 \
  --num-accepted 2
```

Results will be in `output/pdl1_test/Accepted/`.

### 3. Run Your Own Target

```bash
docker run --gpus all \
  -v /path/to/your/outputs:/outputs \
  -v /path/to/your/inputs:/inputs:ro \
  -v /scratch2/mz32/share/mber_weights:/mber_weights:ro \
  -e MBER_WEIGHTS_DIR=/mber_weights \
  theta-mber:latest \
  --input-pdb /inputs/your_target.pdb \
  --output-dir /outputs/my_design \
  --chains A \
  --num-accepted 100
```

## Using a Settings File

For more control, use a YAML settings file (see `example_settings.yml`):

```bash
docker run --gpus all \
  -v $(pwd)/output:/outputs \
  -v $(pwd)/my_inputs:/inputs:ro \
  -v $(pwd)/my_settings.yml:/settings.yml:ro \
  -v /scratch2/mz32/share/mber_weights:/mber_weights:ro \
  -e MBER_WEIGHTS_DIR=/mber_weights \
  theta-mber:latest \
  --settings /settings.yml
```

## Model Weights

mBER requires several model weights (~9GB total for the default VHH path):
- **AlphaFold2** (~3.5GB) - Structure prediction
- **NanoBodyBuilder2** (~0.7GB) - VHH structure folding
- **ESM2** (~5GB) - Protein language model

The container treats **`/mber_weights`** as the single weight root (`MBER_WEIGHTS_DIR`). Mount a host directory that already contains:

```text
mber_weights/
  af_params/
  nbb2_weights/
  huggingface/
```

**Theta cluster shared root (recommended):**

```bash
export MBER_WEIGHTS_DIR=/scratch2/mz32/share/mber_weights
bash download_weights.sh --check

docker run --gpus all \
  -v ${MBER_WEIGHTS_DIR}:/mber_weights:ro \
  -e MBER_WEIGHTS_DIR=/mber_weights \
  ...
```

If required files are missing under `/mber_weights`, the entrypoint will call `download_weights.sh` automatically.

**Note:** If ESMFold is needed (not used by the default VHH protocol), add `--with-esmfold` when downloading an additional ~16GB.

## Build Options

| Option | Image Size | Runtime Requirement |
|--------|-----------|---------------------|
| Default build | ~22GB | Mount weights at `/mber_weights` or allow first-run download |
| `--build-arg INCLUDE_WEIGHTS=true` | ~30GB | No mounting needed, weights built into `/mber_weights` |

**Build with weights included:**

```bash
docker build -t theta-mber:with-weights -f docker/Dockerfile --build-arg INCLUDE_WEIGHTS=true .

docker run --gpus all \
  -v $(pwd)/output:/outputs \
  -v $(pwd)/inputs:/inputs:ro \
  theta-mber:with-weights \
  --input-pdb /inputs/target.pdb \
  --output-dir /outputs/my_run \
  --chains A
```

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| Your output directory | `/outputs` | Design results (persisted) |
| Your input PDB files | `/inputs` | Target structures (read-only) |
| Shared mBER weight root | `/mber_weights` | Model weights (read-only recommended) |
| Settings YAML file | `/settings.yml` | Configuration (optional, read-only) |

## CLI Options

```
--input-pdb PATH      Target PDB file (required)
--output-dir PATH     Output directory (required)
--chains CHAINS       Target chains, e.g., "A" or "A,B" (required)
--hotspots RESIDUES   Target residues, e.g., "A56" or "A56,B20" (optional)
--num-accepted N      Number of designs to generate (default: 100)
--max-trajectories N  Maximum attempts (default: 10000)
--min-iptm FLOAT      Minimum iPTM score (default: 0.75)
--min-plddt FLOAT     Minimum pLDDT score (default: 0.70)
--settings PATH       Use YAML settings file instead of CLI flags
```

## Troubleshooting

**"could not select device driver" or GPU not found:**
```bash
# Install NVIDIA Container Toolkit
# Ubuntu/Debian:
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Permission denied on output files:**
Output files are created as root. To fix:
```bash
sudo chown -R $(whoami) output/
```

**Build fails with conda ToS error:**
The Dockerfile uses Miniforge to avoid this. If you see ToS errors, ensure you're using the latest Dockerfile.

**Out of GPU memory:**
Try a smaller target protein or use a GPU with more VRAM (32GB+ recommended).
