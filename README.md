# Theta-mBER

> Theta Team **internal fork** of [mBER](https://github.com/manifoldbio/mber-open) (Manifold Binder Engineering and Refinement) for protocol iteration and reproducible experiments on shared GPU clusters.

**Upstream:** [manifoldbio/mber-open](https://github.com/manifoldbio/mber-open) by [Manifold Bio](https://www.manifold.bio/).  
**Original authors:** Erik Swanson, Michael Nichols, Supriya Ravichandran, Pierce Ogden.  
This repository is **not** an official Manifold Bio release. Theta-mBER retains the upstream MIT license and copyright; Theta Team changes are limited to internal deployment, weight-path handling, environment tooling, and protocol ergonomics.

Preprint with experimental validation: [mBER: Controllable de novo antibody design with million-scale experimental screening](https://www.biorxiv.org/content/10.1101/2025.09.26.678877v1)

## Graphical Abstract

![Graphical Abstract](./assets/mBER_graphical_abstract.png)

## Abstract

`mBER` (developed by Manifold Bio) is a modular binder design framework that separates the workflow into template preparation, trajectory optimization, and evaluation, then exposes each stage through configurable protocols. **Theta-mBER** is a Theta Team internal derivative of that open-source codebase, used for cluster deployment, shared weight-path configuration, and protocol iteration. Method credit and scientific priority remain with the original mBER authors and preprint.

At a high level, mBER combines structure-aware template construction, sequence optimization with AlphaFold/ColabDesign-style objectives, and downstream binder evaluation into a unified research pipeline.

## Highlights

- **Research-first design**: rapid experimentation, ablation, and protocol iteration
- **Modular pipeline**: clean separation between `Template`, `Trajectory`, and `Evaluation`
- **Multiple model backends**: AlphaFold-derived design, ESM language modeling, nanobody folding
- **Cluster-friendly weights**: single shared root via `MBER_WEIGHTS_DIR`, with optional per-model overrides
- **Protocol-oriented structure**: reusable core engine with task-specific logic under `protocols/`

## Method Overview

1. **Template** — Prepare target structures, identify or ingest hotspots, build truncations, initialize binder templates.
2. **Trajectory** — Optimize sequence logits and sample candidates through iterative design trajectories.
3. **Evaluation** — Score and filter binders with structure-based and sequence-based metrics.

## Repository Map

- [Protocols Guide](./protocols/README.md) — how protocols are structured and extended
- [Core Components](./src/mber/core/README.md) — internal architecture and module-level concepts
- [Docker Guide](./docker/README.md) — containerized usage with GPU support
- [Notebooks](./notebooks) — exploratory examples and workflow demos
- [License](./LICENSE) / [Third-Party Notices](./THIRD_PARTY_NOTICES.md) — MIT + vendored Apache-2.0 / other obligations

## Installation

Tested on modern NVIDIA datacenter GPUs (A10G, A100, L4, L40S, H100). We recommend at least 32 GB of VRAM for comfortable operation.

All install paths pin **NumPy 1.26.4** and pull OpenMM / pdbfixer / ANARCI from conda-forge/bioconda. Do not upgrade NumPy to 2.x. PyTorch CUDA 12.8 wheels are installed via an extra index so other pip packages still resolve from PyPI.

```bash
git clone https://github.com/zhaorui-bi/Theta-mBER.git
cd Theta-mBER
cp .env.example .env
```

### Option A: Pixi (recommended)

```bash
pixi install
pixi shell

# Weights (pixi also exports MBER_WEIGHTS_DIR for the Theta cluster)
pixi run check-weights
# or: pixi run download-weights

pixi run mber-vhh --help
```

### Option B: Micromamba / Mamba

```bash
# micromamba (or: mamba env create -f environment.yml)
micromamba create -y -f environment.yml
micromamba activate mber

set -a && source .env && set +a
bash download_weights.sh --check
```

`environment.yml` already installs both `mber` and `mber-protocols` in editable mode (`mber-vhh` CLI included).

### Option C: Conda

```bash
conda env create -f environment.yml
conda activate mber

set -a && source .env && set +a
bash download_weights.sh --check
```

## Weight Management

Theta-mBER uses one shared weight root for download-time and runtime resolution.

### Recommended: shared root on the Theta cluster

```bash
export MBER_WEIGHTS_DIR=/scratch2/mz32/share/mber_weights
bash download_weights.sh --check
```

Expected layout:

```text
$MBER_WEIGHTS_DIR/
  af_params/
  nbb2_weights/
  huggingface/
```

### Project-local `.env`

```bash
cp .env.example .env
set -a && source .env && set +a
```

`download_weights.sh` automatically sources a repo-root `.env` when present.

Supported variables:

| Variable | Purpose |
|----------|---------|
| `MBER_WEIGHTS_DIR` | Shared root for all weights |
| `MBER_AF_PARAMS_DIR` | AlphaFold2 parameter directory |
| `MBER_NBB2_WEIGHTS_DIR` | NanoBodyBuilder2 weight directory |
| `MBER_HF_HOME` | HuggingFace / ESM cache directory |

### Optional: per-model overrides

```bash
export MBER_AF_PARAMS_DIR=/data/mber_weights/af_params
export MBER_NBB2_WEIGHTS_DIR=/data/mber_weights/nbb2_weights
export MBER_HF_HOME=/data/mber_weights/huggingface
bash download_weights.sh
```

## Quick Start

### CLI: VHH Binder Design

```bash
# Example settings file
mber-vhh --settings ./protocols/src/mber_protocols/stable/VHH_binder_design/examples/vhh_settings_example.yml

# Direct CLI invocation
mber-vhh \
  --input-pdb ./protocols/src/mber_protocols/stable/VHH_binder_design/examples/PDL1.pdb \
  --output-dir ./output/vhh_pdl1_A56 \
  --chains A \
  --hotspots A56

# Interactive mode
mber-vhh --interactive
```

### Internal Target Examples

```bash
# CXADR epitope-focused VHH design
CUDA_VISIBLE_DEVICES=1 \
  mber-vhh --settings ./protocols/src/mber_protocols/stable/VHH_binder_design/examples/7VYL_CXADR.yml

# FAP dual-side pocket/rim VHH design
CUDA_VISIBLE_DEVICES=2 \
  mber-vhh --settings ./protocols/src/mber_protocols/stable/VHH_binder_design/examples/1Z68_FAP.yml
```

These settings files include tuned VHH masked templates, stricter `min_iptm` / `min_plddt` filters, and structural hotspot constraints.

Full CLI documentation: [VHH_CLI.md](./protocols/src/mber_protocols/stable/VHH_binder_design/VHH_CLI.md).

### Notebooks

See [notebooks](./notebooks) for exploratory workflows.

### Docker

For containerized GPU runs, see the [Docker guide](./docker/README.md). Mount the same weight root at `/mber_weights`:

```bash
docker run --gpus all \
  -v /scratch2/mz32/share/mber_weights:/mber_weights:ro \
  -e MBER_WEIGHTS_DIR=/mber_weights \
  ...
```

## Experiment Notes

- If you pass `--hotspots`, those residues are used directly.
- If hotspots are omitted, the template stage can select them with strategies such as `random`, `top_k`, or `none`.
- On shared machines, pin a GPU with:

```bash
CUDA_VISIBLE_DEVICES=1 mber-vhh ...
```

Within the process, physical GPU `1` appears as logical `cuda:0`.

## Troubleshooting

### `pkg_resources` import error

```bash
conda install -n mber setuptools
# or
pip install setuptools
```

### NumPy 2.x / OpenMM / ImmuneBuilder crash

If you see `_ARRAY_API` or `compiled using NumPy 1.x cannot be run in NumPy 2.x`:

```bash
conda install -n mber "numpy==1.26.4" --force-reinstall
conda install -n mber openmm==8.0.0 pdbfixer==1.9 --force-reinstall
python -m pip install --force-reinstall --no-cache-dir ImmuneBuilder
```

### Mixed NumPy / JAX ABI state

If `numpy._core.umath`, `_multiarray_umath`, `ml_dtypes`, or `jaxlib` fail on import:

```bash
python -m pip install --force-reinstall --no-cache-dir \
  "numpy==1.26.4" \
  "ml_dtypes>=0.4,<1" \
  "jax[cuda12]==0.5.2"
```

If the environment remains inconsistent, recreate the `mber` conda environment from scratch.

### Weights not found

```bash
set -a && source .env && set +a
bash download_weights.sh --check
```

Confirm `MBER_WEIGHTS_DIR` points at a directory that contains `af_params/`, `nbb2_weights/`, and `huggingface/`.

## Citation

If you use mBER / Theta-mBER in research, please cite the **original mBER paper** (Manifold Bio):

```bibtex
@article {swanson2025mber,
  author = {Swanson, Erik and Nichols, Michael and Ravichandran, Supriya and Ogden, Pierce},
  title = {mBER: Controllable de novo antibody design with million-scale experimental screening},
  elocation-id = {2025.09.26.678877},
  year = {2025},
  doi = {10.1101/2025.09.26.678877},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2025/09/28/2025.09.26.678877},
  eprint = {https://www.biorxiv.org/content/early/2025/09/28/2025.09.26.678877.full.pdf},
  journal = {bioRxiv}
}
```

Do **not** cite Theta-mBER as a substitute for the upstream method paper. If you specifically used this fork's infrastructure changes, you may additionally note the repository URL in methods, while still citing Swanson et al.

## Acknowledgements

- **Upstream mBER:** [Manifold Bio](https://www.manifold.bio/) — [manifoldbio/mber-open](https://github.com/manifoldbio/mber-open). Copyright (c) 2025 Manifold Bio.
- Community VHH CLI contributions (e.g. Aaron Ring / @cytokineking), as present in the upstream lineage.
- Supporting open-source tools used by mBER:
  - [AlphaFold](https://github.com/deepmind/alphafold)
  - [ColabDesign](https://github.com/sokrypton/ColabDesign)
  - [ESM](https://github.com/facebookresearch/esm)
  - [AbLang](https://github.com/oxpig/AbLang)
  - [ImmuneBuilder](https://github.com/oxpig/ImmuneBuilder)

## License and copyright

This project is a **derivative work** of Manifold Bio's mBER, redistributed under the same **MIT License**.

- Upstream copyright: **Copyright (c) 2025 Manifold Bio** (see [LICENSE](./LICENSE) and [manifoldbio/mber-open](https://github.com/manifoldbio/mber-open))
- Theta Team modifications: **Copyright (c) 2026 Theta Team**
- The MIT license text and Manifold Bio copyright notice are retained as required
- `mBER`, Manifold Bio branding, and the scientific method remain attributed to the original authors; Theta-mBER does not claim ownership of the upstream project

**Third-party / vendored code** (must be preserved when redistributing):

| Component | Path | License |
|-----------|------|---------|
| AlphaFold (DeepMind) | [`src/mber/models/alphafold/`](./src/mber/models/alphafold/) | Apache-2.0 ([LICENSE](./src/mber/models/alphafold/LICENSE)) |
| BindCraft-adapted losses | [`src/mber/models/colabdesign/loss.py`](./src/mber/models/colabdesign/loss.py) | MIT |
| ColabDesign (dependency) | external package | Beerware (upstream `LICENSE.txt`) |

Full terms: [LICENSE](./LICENSE). Detailed notices: [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md). Model weights downloaded separately are **not** covered by this MIT license.

## Contributing

Internal contributions from the Theta Team are welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md). When contributing, preserve upstream copyright headers and do not remove Manifold Bio attribution from redistributed sources.
