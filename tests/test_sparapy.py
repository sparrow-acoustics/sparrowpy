"""Test sparrowpy package."""

def test_import_sparrowpy():
    """Test importing sparrowpy."""
    try:
        import sparrowpy           # noqa
    except ImportError:
        assert False
