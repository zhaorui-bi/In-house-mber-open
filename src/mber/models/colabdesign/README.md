# Subclassing ColabDesign

We extend ColabDesign without rewriting most of its native functionality by
subclassing its main classes and overriding only the methods we need to change.

## License notes

- ColabDesign itself is an external dependency
  ([sokrypton/ColabDesign](https://github.com/sokrypton/ColabDesign)), distributed
  under the Beerware License (`LICENSE.txt` upstream). Retain that notice when
  redistributing ColabDesign sources.
- Code in this directory is part of the mBER / Theta-mBER tree (MIT), except
  BindCraft-adapted loss helpers in [`loss.py`](./loss.py) (MIT; see file header).
- Broader third-party obligations are summarized in
  [`THIRD_PARTY_NOTICES.md`](../../../../THIRD_PARTY_NOTICES.md).
