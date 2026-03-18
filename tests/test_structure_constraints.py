import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "protocols"
    / "src"
    / "mber_protocols"
    / "stable"
    / "VHH_binder_design"
    / "structure_constraints.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "mber_vhh_structure_constraints",
    MODULE_PATH,
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)

evaluate_structural_constraints = MODULE.evaluate_structural_constraints
parse_structural_constraint_config = MODULE.parse_structural_constraint_config


def _atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_number: int,
    x: float,
    y: float,
    z: float,
    element: str = "C",
) -> str:
    return (
        f"{'ATOM':<6}{serial:>5} "
        f"{atom_name:>4}{' ':1}{residue_name:>3} {chain_id:1}"
        f"{residue_number:>4}{' ':1}   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"{1.00:>6.2f}{20.00:>6.2f}          {element:>2}"
    )


class StructuralConstraintTests(unittest.TestCase):
    def test_cxadr_like_primary_and_secondary_groups_must_both_hit(self) -> None:
        config = parse_structural_constraint_config(
            {
                "contact_distance": 4.5,
                "groups": [
                    {
                        "name": "primary_hotspots",
                        "residues": ["A21", "A23", "A24", "A25"],
                        "min_hits": 3,
                    },
                    {
                        "name": "secondary_hotspots",
                        "residues": ["A44", "A47"],
                        "min_hits": 1,
                    },
                ],
            }
        )

        complex_pdb = "\n".join(
            [
                _atom_line(1, "CA", "ALA", "A", 21, 0.0, 0.0, 0.0),
                _atom_line(2, "CA", "ALA", "A", 23, 10.0, 0.0, 0.0),
                _atom_line(3, "CA", "ALA", "A", 24, 20.0, 0.0, 0.0),
                _atom_line(4, "CA", "ALA", "A", 25, 30.0, 0.0, 0.0),
                _atom_line(5, "CA", "ALA", "A", 44, 40.0, 0.0, 0.0),
                _atom_line(6, "CA", "GLY", "C", 1, 1.0, 0.0, 0.0),
                _atom_line(7, "CA", "GLY", "C", 2, 10.5, 0.0, 0.0),
                _atom_line(8, "CA", "GLY", "C", 3, 19.5, 0.0, 0.0),
                _atom_line(9, "CA", "GLY", "C", 4, 40.5, 0.0, 0.0),
            ]
        )

        passed, details = evaluate_structural_constraints(
            complex_pdb=complex_pdb,
            config=config,
            target_chains={"A"},
        )

        self.assertTrue(passed)
        self.assertEqual(details["group_hits"]["primary_hotspots"], ["A21", "A23", "A24"])
        self.assertEqual(details["group_hits"]["secondary_hotspots"], ["A44"])

    def test_fap_like_dual_sided_rule_rejects_single_sided_contact(self) -> None:
        config = parse_structural_constraint_config(
            {
                "contact_distance": 4.5,
                "groups": [
                    {
                        "name": "catalytic_rim_north",
                        "residues": ["A122", "A123", "A124"],
                        "min_hits": 1,
                    },
                    {
                        "name": "catalytic_rim_south",
                        "residues": ["A732", "A733", "A735", "A737", "A741"],
                        "min_hits": 1,
                    },
                ],
            }
        )

        complex_pdb = "\n".join(
            [
                _atom_line(1, "CA", "ALA", "A", 122, 0.0, 0.0, 0.0),
                _atom_line(2, "CA", "ALA", "A", 123, 10.0, 0.0, 0.0),
                _atom_line(3, "CA", "ALA", "A", 732, 100.0, 0.0, 0.0),
                _atom_line(4, "CA", "GLY", "C", 1, 1.0, 0.0, 0.0),
                _atom_line(5, "CA", "GLY", "C", 2, 9.5, 0.0, 0.0),
            ]
        )

        passed, details = evaluate_structural_constraints(
            complex_pdb=complex_pdb,
            config=config,
            target_chains={"A"},
        )

        self.assertFalse(passed)
        self.assertEqual(details["reason"], "group_min_hits")
        self.assertIn("catalytic_rim_south", details["failed_groups"])

    def test_non_hotspot_target_chain_is_not_misclassified_as_binder(self) -> None:
        config = parse_structural_constraint_config(
            {
                "groups": [
                    {
                        "name": "interface_patch",
                        "residues": ["A21"],
                        "min_hits": 1,
                    }
                ]
            }
        )

        complex_pdb = "\n".join(
            [
                _atom_line(1, "CA", "ALA", "A", 21, 0.0, 0.0, 0.0),
                _atom_line(2, "CA", "SER", "B", 99, 1.0, 0.0, 0.0),
                _atom_line(3, "CA", "GLY", "C", 1, 20.0, 0.0, 0.0),
            ]
        )

        passed, details = evaluate_structural_constraints(
            complex_pdb=complex_pdb,
            config=config,
            target_chains={"A", "B"},
        )

        self.assertFalse(passed)
        self.assertEqual(details["reason"], "group_min_hits")
        self.assertEqual(details["group_hits"]["interface_patch"], [])


if __name__ == "__main__":
    unittest.main()
