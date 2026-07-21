# Contributing to Theta-mBER

This repository is a **Theta Team internal fork** of [manifoldbio/mber-open](https://github.com/manifoldbio/mber-open) (Copyright (c) 2025 Manifold Bio, MIT License). It is not an official Manifold Bio project.

For new protocol development, see the [Protocols Guide](./protocols/README.md), [Core Components](./src/mber/core/README.md), and the [VHH Binder Design Protocol](./protocols/src/mber_protocols/stable/VHH_binder_design/README.md).

## How to contribute

1. Keep changes focused: protocol logic under `protocols/`, shared engine under `src/mber/`.
2. Prefer environment variables and `.env` for cluster-specific weight paths; do not hard-code personal home directories in committed files.
3. Preserve license notices:
   - Do not strip Manifold Bio attribution from `LICENSE`, README, or docs.
   - Do not remove Apache-2.0 headers or [`src/mber/models/alphafold/LICENSE`](./src/mber/models/alphafold/LICENSE).
   - When adapting third-party code, keep attribution comments and update [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).
4. Open a PR with a short summary of the motivation and how you tested the change.
5. For bug reports that belong upstream, prefer filing them against [manifoldbio/mber-open](https://github.com/manifoldbio/mber-open) when appropriate; Theta-specific infra issues can go to [Theta-mBER](https://github.com/zhaorui-bi/Theta-mBER).

## Weight / environment notes

- Shared weight root on the Theta cluster: `/scratch2/mz32/share/mber_weights`
- Template: copy [`.env.example`](./.env.example) to `.env` (`.env` is gitignored)
- Verify weights with `bash download_weights.sh --check` (or `pixi run check-weights`)
- Environments: Pixi ([`pixi.toml`](./pixi.toml)), Micromamba/Conda ([`environment.yml`](./environment.yml))

## Upstream acknowledgements

- Original mBER authors: Erik Swanson, Michael Nichols, Supriya Ravichandran, Pierce Ogden (Manifold Bio)
- Upstream repository: https://github.com/manifoldbio/mber-open
- Community VHH CLI work in the upstream lineage includes contributions by Aaron Ring (@cytokineking)
