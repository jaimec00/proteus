
import omegaconf as om
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from pathlib import Path
import importlib
from dataclasses import is_dataclass

from proteus.types import Any

def load_model_cls(model_cfg: Any):
    '''
    model cfgs have a model_cls attr to instantiate the cfg with
    '''
    if hasattr(model_cfg, "model_cls"):
        model_cls = model_cfg.model_cls
    else:
        raise RuntimeError

    module_str, model_cls = model_cls.rsplit(".", 1)
    module = importlib.import_module(module_str)
    return getattr(module, model_cls)

def add_trgt_to_cfg(cfg: DictConfig) -> DictConfig:
    '''recursively add _target_ to a structured config based on its type metadata.'''
    if not isinstance(cfg, DictConfig):
        return

    obj_type = OmegaConf.get_type(cfg)
    if obj_type is not None and is_dataclass(obj_type):
        if "_target_" not in cfg:
            OmegaConf.update(cfg, "_target_", f"{obj_type.__module__}.{obj_type.__qualname__}", force_add=True)

        for key in cfg:
            if key.startswith("_"):
                continue
            if isinstance(cfg[key], DictConfig):
                add_trgt_to_cfg(cfg[key])

def serialize_cfg(cfg: Any) -> DictConfig:
    '''
    convert typed dict to a yaml string for saving
    also adds target for easy instantiation
    '''
    structured_cfg = OmegaConf.structured(cfg)
    add_trgt_to_cfg(structured_cfg)
    yaml_cfg = OmegaConf.to_yaml(structured_cfg)
    return yaml_cfg

def save_cfg(cfg: Any, out_path: Path):
    serialized = serialize_cfg(cfg)
    with open(out_path, "w") as f:
        f.write(serialized)

def load_cfg(cfg_path: Path):
    return instantiate(OmegaConf.load(cfg_path))