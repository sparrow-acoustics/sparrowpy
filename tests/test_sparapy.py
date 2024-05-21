def test_import_sparapy():
    try:
        import sparapy           # noqa
    except ImportError:
        assert False
