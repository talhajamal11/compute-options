"""Source adapters. Importing this package registers every adapter in ``REGISTRY``."""

from compute_options.data.sources import (  # noqa: F401
    gpu_price_tracker,
    gpu_pricing_tracker,
    gpu_rental_prices,
    gpuwatch,
    ornn,
    vast,
)
from compute_options.data.sources.base import REGISTRY, Source

__all__ = ["REGISTRY", "Source", "get_source"]


def get_source(name: str, **kwargs) -> Source:
    try:
        return REGISTRY[name](**kwargs)
    except KeyError:
        raise KeyError(f"unknown source {name!r}; available: {sorted(REGISTRY)}") from None
