import os
from dataclasses import dataclass
from typing import Mapping, Optional


DEFAULT_MBER_WEIGHTS_DIR = "~/.mber"
MBER_WEIGHTS_DIR_ENV = "MBER_WEIGHTS_DIR"
MBER_AF_PARAMS_DIR_ENV = "MBER_AF_PARAMS_DIR"
MBER_NBB2_WEIGHTS_DIR_ENV = "MBER_NBB2_WEIGHTS_DIR"
MBER_HF_HOME_ENV = "MBER_HF_HOME"


@dataclass(frozen=True)
class ModelPathConfig:
    """Resolved local filesystem paths for all mBER model weights."""

    weights_root_dir: str
    af_params_dir: str
    nbb2_weights_dir: str
    hf_home: str
    hf_hub_cache: str


def normalize_local_path(path: str) -> str:
    """Expand env vars and `~`, then normalize to an absolute path."""
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def resolve_weights_root_dir(
    weights_root_dir: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    env = os.environ if env is None else env
    candidate = weights_root_dir or env.get(MBER_WEIGHTS_DIR_ENV) or DEFAULT_MBER_WEIGHTS_DIR
    return normalize_local_path(candidate)


def resolve_af_params_dir(
    af_params_dir: Optional[str] = None,
    weights_root_dir: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    env = os.environ if env is None else env
    candidate = af_params_dir or env.get(MBER_AF_PARAMS_DIR_ENV)
    if candidate:
        return normalize_local_path(candidate)
    return os.path.join(resolve_weights_root_dir(weights_root_dir, env=env), "af_params")


def resolve_nbb2_weights_dir(
    nbb2_weights_dir: Optional[str] = None,
    weights_root_dir: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    env = os.environ if env is None else env
    candidate = nbb2_weights_dir or env.get(MBER_NBB2_WEIGHTS_DIR_ENV)
    if candidate:
        return normalize_local_path(candidate)
    return os.path.join(
        resolve_weights_root_dir(weights_root_dir, env=env),
        "nbb2_weights",
    )


def resolve_hf_home(
    hf_home: Optional[str] = None,
    weights_root_dir: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    env = os.environ if env is None else env
    candidate = hf_home or env.get(MBER_HF_HOME_ENV) or env.get("HF_HOME")
    if candidate:
        return normalize_local_path(candidate)
    return os.path.join(resolve_weights_root_dir(weights_root_dir, env=env), "huggingface")


def resolve_hf_hub_cache_dir(
    hf_home: Optional[str] = None,
    weights_root_dir: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    env = os.environ if env is None else env
    explicit_cache = env.get("HF_HUB_CACHE")
    if explicit_cache:
        return normalize_local_path(explicit_cache)
    return os.path.join(resolve_hf_home(hf_home, weights_root_dir=weights_root_dir, env=env), "hub")


def resolve_model_path_config(
    weights_root_dir: Optional[str] = None,
    af_params_dir: Optional[str] = None,
    nbb2_weights_dir: Optional[str] = None,
    hf_home: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> ModelPathConfig:
    env = os.environ if env is None else env

    resolved_weights_root_dir = resolve_weights_root_dir(weights_root_dir, env=env)
    resolved_af_params_dir = resolve_af_params_dir(
        af_params_dir,
        weights_root_dir=resolved_weights_root_dir,
        env=env,
    )
    resolved_nbb2_weights_dir = resolve_nbb2_weights_dir(
        nbb2_weights_dir,
        weights_root_dir=resolved_weights_root_dir,
        env=env,
    )
    resolved_hf_home = resolve_hf_home(
        hf_home,
        weights_root_dir=resolved_weights_root_dir,
        env=env,
    )
    resolved_hf_hub_cache = resolve_hf_hub_cache_dir(
        resolved_hf_home,
        weights_root_dir=resolved_weights_root_dir,
        env=env,
    )

    return ModelPathConfig(
        weights_root_dir=resolved_weights_root_dir,
        af_params_dir=resolved_af_params_dir,
        nbb2_weights_dir=resolved_nbb2_weights_dir,
        hf_home=resolved_hf_home,
        hf_hub_cache=resolved_hf_hub_cache,
    )


def configure_huggingface_environment(hf_home: Optional[str] = None) -> ModelPathConfig:
    """Apply the resolved Hugging Face cache path to the current process."""
    resolved = resolve_model_path_config(hf_home=hf_home)
    os.environ["HF_HOME"] = resolved.hf_home
    os.environ.setdefault("HF_HUB_CACHE", resolved.hf_hub_cache)
    return resolved
