"""Form factor models for Radiosity."""
from .universal import (
    patch2patch_ff_universal,
)
from .kang import (
    patch2patch_ff_kang,
)

__all__ = [
    "patch2patch_ff_universal",
    "patch2patch_ff_kang",
]
