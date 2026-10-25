"""The server prices OSS traffic through metergraph-core.

The exhaustive pricing golden suite lives in core/tests. These tests only
assert that the server's catalog/prices modules are thin compatibility
re-exports of the shared core, with no second implementation or catalog copy.
"""

import pytest

from metergraph_core import CatalogSnapshot as CoreCatalogSnapshot

from metergraph_server import prices
from metergraph_server.catalog import Alias, CatalogSnapshot, CostResult, Price


def test_server_catalog_is_core_compatibility_export():
    assert CatalogSnapshot is CoreCatalogSnapshot


def test_server_catalog_reexports_core_types():
    from metergraph_core import Alias as CoreAlias
    from metergraph_core import CostResult as CoreCostResult
    from metergraph_core import Price as CorePrice

    assert Alias is CoreAlias
    assert CostResult is CoreCostResult
    assert Price is CorePrice


def test_server_loader_uses_core_bundled_catalog():
    version, document, snapshot = prices.load()
    assert version == "2026-10-25"
    assert document["models"]
    assert isinstance(snapshot, CoreCatalogSnapshot)


def test_server_prices_error_is_core_catalog_error():
    from metergraph_core import CatalogError

    assert prices.PricesError is CatalogError


def test_server_parse_alias_rejects_malformed_catalog():
    with pytest.raises(prices.PricesError):
        prices.parse({"models": []})
