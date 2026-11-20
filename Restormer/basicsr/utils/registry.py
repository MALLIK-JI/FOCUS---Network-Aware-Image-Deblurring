
import functools


class Registry:
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _register(self, obj, name=None):
        if name is None:
            name = obj.__name__

        if name in self._obj_map:
            raise KeyError(
                f"{name} is already registered in {self._name}"
            )

        self._obj_map[name] = obj
        return obj

    def register(self, name=None):
        if callable(name):
            return self._register(name)

        def decorator(obj):
            return self._register(obj, name)

        return decorator

    def get(self, name):
        if name not in self._obj_map:
            raise KeyError(
                f"{name} is not registered in {self._name}"
            )
        return self._obj_map[name]

    def __contains__(self, name):
        return name in self._obj_map

    def __repr__(self):
        return f"Registry(name={self._name}, items={list(self._obj_map.keys())})"


def build_from_cfg(cfg, registry, default_args=None):
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dictionary")

    args = cfg.copy()

    if default_args is not None:
        for key, value in default_args.items():
            args.setdefault(key, value)

    obj_type = args.pop("type")

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
    elif callable(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError("type must be a string or callable")

    return obj_cls(**args)
ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')
DATASET_REGISTRY = Registry('dataset')
SCHEDULER_REGISTRY = Registry('scheduler')
OPTIMIZER_REGISTRY = Registry('optimizer')