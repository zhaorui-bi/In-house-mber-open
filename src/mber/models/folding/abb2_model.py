from .folding_model_bases import ProteinFoldingModel
from typing import Any
from pathlib import Path
import numpy as np
import torch


class ABB2Model(ProteinFoldingModel):
    """ABodyBuilder2 implementation of the ProteinFoldingModel interface."""

    @staticmethod
    def _ensure_numpy_compatibility() -> None:
        major_version = int(np.__version__.split(".", 1)[0])
        if major_version >= 2:
            raise RuntimeError(
                "ABodyBuilder2 currently depends on the OpenMM/pdbfixer stack, "
                "which is not compatible with NumPy 2.x in this environment. "
                "Please install `numpy<2` in the active environment."
            )

    def __init__(
        self,
        model: Any = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize ABodyBuilder2 model."""
        self._ensure_numpy_compatibility()
        if model is None:
            try:
                from ImmuneBuilder import ABodyBuilder2
            except ImportError as exc:
                raise ImportError(
                    "Failed to import ImmuneBuilder/ABodyBuilder2. "
                    "Please verify ImmuneBuilder and its runtime dependencies are installed "
                    "in the active environment."
                ) from exc

            model = ABodyBuilder2(numbering_scheme='raw')
        self.model = model
        self.device = device

    def predict_structure(
        self,
        sequence: str,
        output_pdb_path: Path,
    ) -> Path:
        """
        Predict the 3D structure of a paired heavy-chain light-chain antibody Fab domain and save it as a PDB file.
        
        Chains should be separated by the '|' character (e.g. "{Hchain_seq}|{Lchain_seq}")
        """
        # if sequence contains (G4S)3, assume it is scFv and split it into H and L chains
        if "GGGGSGGGGSGGGGS" in sequence:
            sequence = sequence.split("GGGGSGGGGSGGGGS")
            sequence = {"L": sequence[0], "H": sequence[1]}
        else:
            sequence = {"H": sequence.split("|")[1], "L": sequence.split("|")[0]}

        print(f"Predicting structure for H chain: {sequence['H']} and L chain: {sequence['L']} with ABodyBuilder2...")

        with torch.no_grad():
            antibody = self.model.predict(sequence)
            
        antibody.save(output_pdb_path)

        return output_pdb_path
