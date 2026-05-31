from pulsar_nav.catalog import load_bundled_catalog, load_catalog


def test_bundled_count():
    assert len(load_bundled_catalog()) == 5


def test_load_by_name():
    p = load_catalog(["J0437-4715"])[0]
    assert p.f0_hz > 170.0
