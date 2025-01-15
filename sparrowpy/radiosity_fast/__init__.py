"""Module of the fast radiosity solver."""
from .radiosity_class import (
    DRadiosityFast,
)

from . import geometry

__all__ = [
    'DRadiosityFast',
    'geometry',
]
