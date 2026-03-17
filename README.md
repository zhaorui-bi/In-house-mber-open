# 🧬 mBER In-House

> ✨ A NeurIPS-style in-house research codebase for modular antibody binder design, fast protocol iteration, and reproducible internal experiments.

📄 Preprint with experimental validation: [mBER: Controllable de novo antibody design with million-scale experimental screening](https://www.biorxiv.org/content/10.1101/2025.09.26.678877v1)

## 🎨 Graphical Abstract

![Graphical Abstract](./assets/mBER_graphical_abstract.png)

## 📝 Abstract

`mBER` is a modular binder design framework built around a simple research abstraction: separate the design workflow into template preparation, trajectory optimization, and evaluation, then expose each stage through configurable protocols. This repository is the **in-house version** used for internal iteration, workflow tuning, and infrastructure customization. Compared with a minimal public-style release, this version emphasizes operational usability, configurable weight locations, protocol ergonomics, and experiment execution on shared GPU systems.

At a high level, mBER combines structure-aware template construction, sequence optimization with AlphaFold/ColabDesign-style objectives, and downstream binder evaluation into a unified research pipeline. The result is a codebase that is easy to adapt to new design tasks while still remaining explicit about where templates, losses, trajectories, and model backends enter the system.

## 🌟 Highlights

- 🧪 **Research-first design**: built for rapid experimentation, ablation, and protocol iteration.
- 🧩 **Modular pipeline**: clean separation between `Template`, `Trajectory`, and `Evaluation`.
- 🧠 **Multiple model backends**: AlphaFold-derived design, ESM-based language modeling, and nanobody folding support.
- 📦 **Internal usability**: custom weight directories, CLI workflows, Docker support, and resume-friendly outputs.
- 🚀 **Protocol-oriented structure**: the core engine stays reusable while task-specific logic lives under `protocols/`.

## 🏗️ Method Overview

The mBER pipeline is organized around three core stages:

1. **Template** 🧱
   Prepares target structures, identifies or ingests hotspots, builds truncations, and initializes binder templates.
2. **Trajectory** 🎯
   Runs binder design by optimizing sequence logits and sampling candidate sequences through iterative design trajectories.
3. **Evaluation** 📏
   Scores and filters resulting binders using structure-based and sequence-based metrics.

This architecture keeps the codebase close to the way we describe methods in a paper: each stage has a clear role, explicit inputs and outputs, and a protocol layer that defines task-specific defaults.

## 🗂️ Repository Map

- 📚 [Protocols Guide](./protocols/README.md) — how protocols are structured and extended
- 🧠 [Core Components](./src/mber/core/README.md) — internal architecture and module-level concepts
- 🐳 [Docker Guide](./docker/README.md) — containerized usage with GPU support
- 📓 [Notebooks](./notebooks) — exploratory examples and workflow demos

## ⚙️ Installation

mBER has been tested on modern NVIDIA datacenter GPUs, including A10G, A100, L4, L40S, and H100. We recommend at least 32 GB of VRAM for comfortable operation, although smaller targets may work on lower-memory GPUs.

```bash
# Clone this in-house repository
git clone <your-in-house-mber-repo>
cd In-house-mber-open

# Create the conda environment
conda env create -f environment.yml
conda activate mber

# Install protocol package
pip install -e protocols

# Download model weights (~9GB for AlphaFold2 + NanoBodyBuilder2 + ESM2)
bash download_weights.sh
```

## 💾 Weight Management

This in-house version supports **fully custom model weight locations** for both download-time and runtime. This is especially useful on shared clusters where `HOME` is small but scratch or project storage is large.

### ✅ Recommended: one shared root

```bash
export MBER_WEIGHTS_DIR=/data/mber_weights
bash download_weights.sh
```

### 🔧 Optional: per-model overrides

```bash
export MBER_AF_PARAMS_DIR=/data/mber_weights/af_params
export MBER_NBB2_WEIGHTS_DIR=/data/mber_weights/nbb2_weights
export MBER_HF_HOME=/data/mber_weights/huggingface
bash download_weights.sh
```

### 📄 Project-local `.env`

Use the included `.env.example` as a template:

```bash
cp .env.example .env
set -a
source .env
set +a
```

Supported variables:

- `MBER_WEIGHTS_DIR` — shared root for all weights
- `MBER_AF_PARAMS_DIR` — AlphaFold2 parameter directory
- `MBER_NBB2_WEIGHTS_DIR` — NanoBodyBuilder2 weight directory
- `MBER_HF_HOME` — HuggingFace / ESM cache directory

The same variables are used by `download_weights.sh` and by runtime model loading, so the system stays consistent end to end.

## 🚀 Quick Start

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

📘 Full CLI documentation lives in [VHH_CLI.md](./protocols/src/mber_protocols/stable/VHH_binder_design/VHH_CLI.md).

### 📓 Notebook Usage

See [notebooks](./notebooks) for exploratory and interactive workflows.

### 🐳 Docker Usage

For containerized execution with GPU support, see the [Docker guide](./docker/README.md).

## 🧪 Experiment Notes

- 🎯 If you manually pass `--hotspots`, those hotspot residues are used directly.
- 🎲 If hotspots are not provided, the template stage can automatically select them using configurable strategies such as `random`, `top_k`, or `none`.
- 🖥️ On shared machines, the most reliable way to choose a specific GPU is:

```bash
CUDA_VISIBLE_DEVICES=1 mber-vhh ...
```

Within the process, this makes physical GPU `1` appear as logical `cuda:0`.

## 🛠️ Troubleshooting

### `pkg_resources` import error

If you see `ModuleNotFoundError: No module named 'pkg_resources'`, your environment is missing `setuptools` at runtime.

```bash
conda install -n mber setuptools
```

or

```bash
pip install setuptools
```

### NumPy 2.x / OpenMM / ImmuneBuilder crash

If you see `_ARRAY_API`, `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`, or an OpenMM/ImmuneBuilder crash during refinement, downgrade NumPy:

```bash
conda install -n mber "numpy==1.26.4" --force-reinstall
```

Then reinstall dependent binary packages if needed:

```bash
conda install -n mber openmm==8.0.0 pdbfixer==1.9 --force-reinstall
python -m pip install --force-reinstall --no-cache-dir ImmuneBuilder
```

### Mixed NumPy / JAX ABI state

If you see errors such as `numpy._core.umath failed to import`, `ImportError: _multiarray_umath failed to import`, or `ml_dtypes` / `jaxlib` failing on import, reinstall the NumPy and JAX stack together:

```bash
python -m pip install --force-reinstall --no-cache-dir \
  "numpy==1.26.4" \
  "ml_dtypes>=0.4,<1" \
  "jax[cuda12]==0.5.2"
```

If the environment remains inconsistent after that, recreating the `mber` conda environment from scratch is usually the fastest fix.

## 📖 Citation

If you use this code in research, please cite the paper:

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

## 🙌 Acknowledgements

This in-house version builds on several excellent open-source tools and models:

- 🔬 [AlphaFold](https://github.com/deepmind/alphafold)
- 🧠 [ColabDesign](https://github.com/sokrypton/ColabDesign)
- 🧬 [ESM](https://github.com/facebookresearch/esm)
- 🧪 [AbLang](https://github.com/oxpig/AbLang)
- 🏗️ [ImmuneBuilder](https://github.com/oxpig/ImmuneBuilder)

## 📜 License

MIT License. See [LICENSE](./LICENSE) for details.

## 🤝 Contributing

We welcome contributions and internal improvements. See [CONTRIBUTING.md](./CONTRIBUTING.md) for contribution guidelines.
