import importlib.util
import os
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

from mber.utils.model_paths import resolve_model_path_config


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mber"
    / "core"
    / "modules"
    / "config.py"
)
CONFIG_SPEC = importlib.util.spec_from_file_location(
    "mber_core_modules_config",
    CONFIG_PATH,
)
assert CONFIG_SPEC is not None and CONFIG_SPEC.loader is not None
CONFIG_MODULE = importlib.util.module_from_spec(CONFIG_SPEC)
sys.modules[CONFIG_SPEC.name] = CONFIG_MODULE
CONFIG_SPEC.loader.exec_module(CONFIG_MODULE)
BaseEnvironmentConfig = CONFIG_MODULE.BaseEnvironmentConfig


class ModelPathResolutionTests(unittest.TestCase):
    def test_shared_root_env_resolves_all_subdirectories(self) -> None:
        resolved = resolve_model_path_config(
            env={
                "HOME": "/tmp/home",
                "MBER_WEIGHTS_DIR": "/mnt/mber_weights",
            }
        )

        self.assertEqual(resolved.weights_root_dir, "/mnt/mber_weights")
        self.assertEqual(resolved.af_params_dir, "/mnt/mber_weights/af_params")
        self.assertEqual(resolved.nbb2_weights_dir, "/mnt/mber_weights/nbb2_weights")
        self.assertEqual(resolved.hf_home, "/mnt/mber_weights/huggingface")
        self.assertEqual(resolved.hf_hub_cache, "/mnt/mber_weights/huggingface/hub")

    def test_individual_overrides_take_precedence(self) -> None:
        resolved = resolve_model_path_config(
            env={
                "HOME": "/tmp/home",
                "MBER_WEIGHTS_DIR": "/mnt/mber_weights",
                "MBER_AF_PARAMS_DIR": "/ssd/af_params",
                "MBER_NBB2_WEIGHTS_DIR": "/ssd/nbb2_weights",
                "MBER_HF_HOME": "/ssd/hf_home",
            }
        )

        self.assertEqual(resolved.af_params_dir, "/ssd/af_params")
        self.assertEqual(resolved.nbb2_weights_dir, "/ssd/nbb2_weights")
        self.assertEqual(resolved.hf_home, "/ssd/hf_home")
        self.assertEqual(resolved.hf_hub_cache, "/ssd/hf_home/hub")

    def test_environment_config_reads_env_and_exports_hf_paths(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HOME": "/tmp/home",
                "MBER_WEIGHTS_DIR": "/data/mber",
            },
            clear=True,
        ):
            env_cfg = BaseEnvironmentConfig()

            self.assertEqual(env_cfg.weights_root_dir, "/data/mber")
            self.assertEqual(env_cfg.af_params_dir, "/data/mber/af_params")
            self.assertEqual(env_cfg.nbb2_weights_dir, "/data/mber/nbb2_weights")
            self.assertEqual(env_cfg.hf_home, "/data/mber/huggingface")
            self.assertEqual(os.environ["HF_HOME"], "/data/mber/huggingface")
            self.assertEqual(
                os.environ["HF_HUB_CACHE"],
                "/data/mber/huggingface/hub",
            )


if __name__ == "__main__":
    unittest.main()
