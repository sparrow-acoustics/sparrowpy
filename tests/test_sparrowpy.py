"""Test sparrowpy package."""
import pytest


def test_import_sparrowpy():
    """Test importing sparrowpy."""
    try:
        import sparrowpy           # noqa
    except ImportError:
        pytest.fail('import sparrowpy failed')
