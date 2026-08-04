"""Test-suite safety guards for opt-in external integrations."""

import pytest


@pytest.fixture(autouse=True)
def disable_pageindex_network_by_default(request, monkeypatch):
    """Unit/starter tests must not spend PageIndex quota or wait on polling."""

    if request.node.get_closest_marker("live"):
        return
    from src import task8_pageindex_vectorless as task8

    monkeypatch.setattr(task8, "PAGEINDEX_API_KEY", "")

