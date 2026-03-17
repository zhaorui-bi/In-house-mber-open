from .folding_model_bases import ProteinFoldingModel
import os
from typing import Any, Optional, Union
from pathlib import Path
import numpy as np
import torch
from mber.utils.model_paths import resolve_nbb2_weights_dir


class NBB2Model(ProteinFoldingModel):
    """NanoBodyBuilder2 implementation of the ProteinFoldingModel interface."""

    @staticmethod
    def _ensure_numpy_compatibility() -> None:
        major_version = int(np.__version__.split(".", 1)[0])
        if major_version >= 2:
            raise RuntimeError(
                "NanoBodyBuilder2 currently depends on the OpenMM/pdbfixer stack, "
                "which is not compatible with NumPy 2.x in this environment. "
                "Please install `numpy<2` in the active environment, for example:\n"
                "  conda install -n mber 'numpy<2' --force-reinstall\n"
                "or\n"
                "  python -m pip install --force-reinstall --no-cache-dir 'numpy<2'"
            )

    def __init__(
        self,
        model: Any = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        weights_dir: Optional[str] = None,
    ):
        """Initialize NanoBodyBuilder2 model."""
        # if weights dir does not exist, create it
        weights_dir = resolve_nbb2_weights_dir(weights_dir)
        os.makedirs(weights_dir, exist_ok=True)
        self._ensure_numpy_compatibility()

        if model is None:
            try:
                from ImmuneBuilder import NanoBodyBuilder2
            except ImportError as exc:
                raise ImportError(
                    "Failed to import ImmuneBuilder/NanoBodyBuilder2. "
                    "This usually means the runtime dependency chain for ImmuneBuilder "
                    "is incomplete or broken, often because `setuptools`/`pkg_resources` "
                    "is missing in the active environment. "
                    "Try `python -m pip install --force-reinstall --no-cache-dir 'setuptools<81'` "
                    "and then reinstall ImmuneBuilder if needed."
                ) from exc

            model = NanoBodyBuilder2(numbering_scheme='raw', weights_dir=weights_dir)
        self.model = model
        self.device = device

    def predict_structure(
        self,
        sequence: Union[str, dict],
        output_pdb_path: Union[str, Path],
    ) -> str:
        """
        Predict the 3D structure of a protein sequence and save it as a PDB file.
        
        Args:
            sequence: Amino acid sequence as string or dict ({"H": sequence})
            output_pdb_path: Path to save the PDB file
        
        Returns:
            Path to the saved PDB file
        """
        # Ensure output path is a string
        output_pdb_path = str(output_pdb_path)
        
        # Convert string sequence to dictionary format if needed
        if isinstance(sequence, str):
            sequence = {"H": sequence}
            
        # NBB2 expects dict format
        if not isinstance(sequence, dict):
            raise ValueError("Sequence must be a string or dictionary with chain as key")
            
        with torch.no_grad():
            nanobody = self.model.predict(sequence)
            
        nanobody.save(output_pdb_path)

        return output_pdb_path
