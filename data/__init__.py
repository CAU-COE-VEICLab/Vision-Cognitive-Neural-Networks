from .build import build_loader as _build_loader


def build_loader(config, *cfg):
    return _build_loader(config)
