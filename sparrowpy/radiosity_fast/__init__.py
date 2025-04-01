"""Module of the fast radiosity solver."""
from .radiosity_class import (
    DirectionalRadiosityFast,
)

from . import geometry

__all__ = [
    'DirectionalRadiosityFast',
    'geometry',
]
