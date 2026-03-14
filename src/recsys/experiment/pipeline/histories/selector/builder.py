from .selector.registry import SELECTOR_REGISTRY


def selector_builder(name, **kwargs):
    func = SELECTOR_REGISTRY[name]
    return func