from .data_utils import RefLoader


def __getattr__(name):
    if name in ("Config", "load_config", "merge_best_epochs"):
        from .pipeline import Config, load_config, merge_best_epochs
        globals().update({"Config": Config, "load_config": load_config, "merge_best_epochs": merge_best_epochs})
        return globals()[name]
    raise AttributeError(f"module 'utils' has no attribute {name!r}")


__all__ = [
    "RefLoader",
    "Config",
    "load_config",
    "merge_best_epochs",
]