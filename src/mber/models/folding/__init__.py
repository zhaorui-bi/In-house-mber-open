from mber.models.folding.folding_model_bases import ProteinFoldingModel
from mber.models.folding.nbb2_model import NBB2Model
from mber.models.folding.abb2_model import ABB2Model
from mber.models.folding.esmfold_model import ESMFoldModel
# from .af2_model import AF2Model

FOLDING_MODELS = {
    'nbb2': NBB2Model,
    'abb2': ABB2Model,
    'esmfold': ESMFoldModel,
    # 'af2': AF2Model,
}


def get_folding_model_kwargs(
    model_name: str,
    af_params_dir: str | None = None,
    nbb2_weights_dir: str | None = None,
    hf_home: str | None = None,
) -> dict:
    if model_name == "nbb2":
        return {"weights_dir": nbb2_weights_dir}
    if model_name == "esmfold":
        return {"hf_home": hf_home}
    if model_name == "af2":
        return {"data_dir": af_params_dir}
    return {}
