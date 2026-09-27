"""Live smoke tests against each public endpoint: ``uv run pytest -m network``."""

import pytest

from compute_options.data.sources import REGISTRY, get_source


@pytest.mark.network
@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_source_returns_rows(name):
    df = get_source(name).fetch()
    assert len(df) > 0
    assert (df["price"] > 0).all()
