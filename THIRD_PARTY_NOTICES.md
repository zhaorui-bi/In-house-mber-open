# Third-Party Notices

Theta-mBER is a derivative of Manifold Bio's mBER and redistributes or depends on
additional third-party software. This file summarizes license obligations that
apply **in addition to** the project [LICENSE](./LICENSE) (MIT).

When redistributing this repository, retain:

1. Root [LICENSE](./LICENSE) (Manifold Bio + Theta Team MIT terms)
2. This file
3. Vendored license texts referenced below (especially Apache-2.0 AlphaFold)

## Upstream project

| Component | Source | License | Notes |
|-----------|--------|---------|-------|
| mBER / mber-open | [manifoldbio/mber-open](https://github.com/manifoldbio/mber-open) | MIT | Copyright (c) 2025 Manifold Bio. Primary codebase this fork derives from. |

## Vendored / adapted code in this repository

| Component | Location | License | Notes |
|-----------|----------|---------|-------|
| AlphaFold | [`src/mber/models/alphafold/`](./src/mber/models/alphafold/) | Apache License 2.0 | Copyright DeepMind Technologies Limited. Full text: [`src/mber/models/alphafold/LICENSE`](./src/mber/models/alphafold/LICENSE). Do not remove Apache headers from source files. |
| BindCraft-adapted losses | [`src/mber/models/colabdesign/loss.py`](./src/mber/models/colabdesign/loss.py) | MIT | Functions adapted from [BindCraft](https://github.com/martinpacesa/BindCraft) (`functions/colabdesign_utils.py`). Copyright (c) 2024 Martin Pacesa. |

## Important runtime dependencies (not vendored)

These packages are installed via conda/pip/pixi and remain under their own licenses.
Installers must comply with each upstream license when redistributing binaries or wheels.

| Component | Typical source | License (as commonly published) | Role in mBER |
|-----------|----------------|----------------------------------|--------------|
| ColabDesign | [sokrypton/ColabDesign](https://github.com/sokrypton/ColabDesign) | Beerware License (see upstream `LICENSE.txt`) | Design / AF interface; Theta-mBER subclasses live under `src/mber/models/colabdesign/` |
| ImmuneBuilder / NanoBodyBuilder2 | [oxpig/ImmuneBuilder](https://github.com/oxpig/ImmuneBuilder) | See upstream repository | VHH monomer folding |
| AbLang2 | [oxpig/AbLang](https://github.com/oxpig/AbLang) | See upstream repository | Antibody language model utilities |
| ESM / transformers checkpoints | Meta / Hugging Face | See model cards and package licenses | PLM / optional ESMFold |
| OpenMM / pdbfixer / ANARCI | conda-forge / bioconda | See package metadata | Refinement and numbering |
| PyTorch / JAX / Flax | upstream wheels | See package metadata | ML backends |

Model **weights** downloaded by `download_weights.sh` (AlphaFold parameters, NanoBodyBuilder2, ESM/Hugging Face caches) are subject to their own terms of use and are **not** covered by the MIT license of this repository.

## Compliance checklist for contributors

- Do **not** strip DeepMind Apache-2.0 headers from files under `src/mber/models/alphafold/`.
- Do **not** remove or rewrite the Manifold Bio copyright in root `LICENSE`.
- When copying substantial snippets from other projects, keep attribution comments and update this file.
- Prefer documenting adaptations (source URL + license) next to the code, as in `colabdesign/loss.py`.
