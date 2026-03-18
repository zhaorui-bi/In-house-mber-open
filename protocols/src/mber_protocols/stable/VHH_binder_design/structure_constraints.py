from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


DEFAULT_CONTACT_DISTANCE = 4.5


@dataclass(frozen=True)
class HotspotGroupConstraint:
    name: str
    residues: Tuple[str, ...]
    min_hits: int


@dataclass(frozen=True)
class StructuralConstraintConfig:
    groups: Tuple[HotspotGroupConstraint, ...]
    contact_distance: float = DEFAULT_CONTACT_DISTANCE
    min_total_hotspot_hits: Optional[int] = None


def parse_structural_constraint_config(raw: Optional[Dict[str, Any]]) -> Optional[StructuralConstraintConfig]:
    if not raw:
        return None

    raw_groups = raw.get("groups") or []
    groups: List[HotspotGroupConstraint] = []
    for idx, group in enumerate(raw_groups):
        residues = tuple(str(r).strip() for r in (group.get("residues") or []) if str(r).strip())
        if not residues:
            raise ValueError(f"structural_constraints.groups[{idx}] must define at least one residue")

        min_hits = int(group.get("min_hits", 1))
        if min_hits < 0:
            raise ValueError(f"structural_constraints.groups[{idx}].min_hits must be >= 0")

        groups.append(
            HotspotGroupConstraint(
                name=str(group.get("name") or f"group_{idx + 1}"),
                residues=residues,
                min_hits=min_hits,
            )
        )

    if not groups:
        raise ValueError("structural_constraints must define at least one group")

    min_total_hotspot_hits = raw.get("min_total_hotspot_hits")
    if min_total_hotspot_hits is not None:
        min_total_hotspot_hits = int(min_total_hotspot_hits)
        if min_total_hotspot_hits < 0:
            raise ValueError("structural_constraints.min_total_hotspot_hits must be >= 0")

    return StructuralConstraintConfig(
        groups=tuple(groups),
        contact_distance=float(raw.get("contact_distance", DEFAULT_CONTACT_DISTANCE)),
        min_total_hotspot_hits=min_total_hotspot_hits,
    )


def evaluate_structural_constraints(
    complex_pdb: Optional[str],
    config: Optional[StructuralConstraintConfig],
    target_chains: Optional[Iterable[str]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    if config is None:
        return True, {"enabled": False}
    if not complex_pdb:
        return False, {
            "enabled": True,
            "reason": "missing_complex_pdb",
        }

    contacts = _collect_contacted_target_residues(
        complex_pdb=complex_pdb,
        hotspot_residues={res for group in config.groups for res in group.residues},
        contact_distance=config.contact_distance,
        target_chains=_normalize_chain_ids(target_chains),
    )

    group_hits: Dict[str, List[str]] = {}
    failed_groups: List[str] = []
    for group in config.groups:
        hit_residues = sorted(res for res in group.residues if res in contacts)
        group_hits[group.name] = hit_residues
        if len(hit_residues) < group.min_hits:
            failed_groups.append(group.name)

    total_hits = len(contacts)
    if config.min_total_hotspot_hits is not None and total_hits < config.min_total_hotspot_hits:
        return False, {
            "enabled": True,
            "reason": "min_total_hotspot_hits",
            "contact_distance": config.contact_distance,
            "group_hits": group_hits,
            "total_hotspot_hits": total_hits,
            "required_total_hotspot_hits": config.min_total_hotspot_hits,
        }

    if failed_groups:
        return False, {
            "enabled": True,
            "reason": "group_min_hits",
            "contact_distance": config.contact_distance,
            "group_hits": group_hits,
            "failed_groups": failed_groups,
            "total_hotspot_hits": total_hits,
        }

    return True, {
        "enabled": True,
        "contact_distance": config.contact_distance,
        "group_hits": group_hits,
        "total_hotspot_hits": total_hits,
    }


def _collect_contacted_target_residues(
    complex_pdb: str,
    hotspot_residues: Set[str],
    contact_distance: float,
    target_chains: Optional[Set[str]] = None,
) -> Set[str]:
    target_atoms: Dict[str, List[np.ndarray]] = {residue: [] for residue in hotspot_residues}
    binder_atoms: List[np.ndarray] = []
    effective_target_chains = target_chains or {residue[0] for residue in hotspot_residues if residue}

    for line in complex_pdb.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue

        atom_name = line[12:16].strip()
        if atom_name.startswith("H"):
            continue

        chain_id = line[21].strip()
        residue_number = line[22:26].strip()
        insertion_code = line[26].strip()
        residue_id = f"{chain_id}{residue_number}{insertion_code}".strip()

        try:
            coords = np.array(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ],
                dtype=float,
            )
        except ValueError:
            continue

        if residue_id in target_atoms:
            target_atoms[residue_id].append(coords)
        elif chain_id and chain_id not in effective_target_chains:
            binder_atoms.append(coords)

    if not binder_atoms:
        return set()

    binder_coords = np.stack(binder_atoms, axis=0)
    threshold_sq = contact_distance ** 2

    contacted: Set[str] = set()
    for residue_id, atoms in target_atoms.items():
        if not atoms:
            continue
        target_coords = np.stack(atoms, axis=0)
        deltas = target_coords[:, None, :] - binder_coords[None, :, :]
        dist_sq = np.sum(deltas * deltas, axis=-1)
        if np.any(dist_sq <= threshold_sq):
            contacted.add(residue_id)

    return contacted


def _normalize_chain_ids(raw_chain_ids: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if raw_chain_ids is None:
        return None

    normalized = {str(chain_id).strip() for chain_id in raw_chain_ids if str(chain_id).strip()}
    return normalized or None
