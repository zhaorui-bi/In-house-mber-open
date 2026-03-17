from mber.models.plm.plm_model_bases import ProteinLanguageModel
from .esm2_model import ESM2Model
from .ablang2_model import AbLang2Model

PLM_MODELS = {
    "esm2-650M": lambda **kwargs: ESM2Model(pretrained_model_name_or_path="facebook/esm2_t33_650M_UR50D", **kwargs),
    "ablang2-paired": lambda **kwargs: AbLang2Model(model_to_use="ablang2-paired", **kwargs),
}


def get_plm_model_kwargs(model_name: str, hf_home: str | None = None) -> dict:
    if model_name.startswith("esm2"):
        return {"hf_home": hf_home}
    return {}
