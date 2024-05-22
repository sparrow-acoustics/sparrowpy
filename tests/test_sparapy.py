"""Test sparapy package."""

def test_import_sparapy():
    """Test importing sparapy."""
    try:
        import sparapy           # noqa
    except ImportError:
        assert False
