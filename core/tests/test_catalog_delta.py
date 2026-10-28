"""The pricing delta schema and its pure, atomic application to a catalog
document."""

import copy
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metergraph_core import CatalogError, load_catalog_document
from metergraph_core.catalog_delta import (
    CATALOG_DELTA_SCHEMA_VERSION,
    CatalogDeltaConflict,
    CatalogDeltaError,
    PriceRowKey,
    apply_catalog_delta,
    parse_catalog_delta,
)

CURRENT_ROW = {
    "channel": "openai-api",
    "region": "global",
    "effective_from": "2026-01-01",
    "input_per_mtok": 1.0,
    "output_per_mtok": 4.0,
    "source_url": "https://example.com/pricing",
}


def _key(**overrides):
    key = {
        "canonical_id": "example/model-a",
        "channel": "openai-api",
        "region": "global",
        "effective_from": "2026-01-01",
    }
    key.update(overrides)
    return key


def _new_price(**overrides):
    price = {
        "channel": "openai-api",
        "region": "global",
        "effective_from": "2026-10-20",
        "input_per_mtok": "1.25",
        "output_per_mtok": "5",
        "source_url": "https://example.com/pricing",
    }
    price.update(overrides)
    return price


def _replace(**overrides):
    operation = {
        "op": "replace_price",
        "key": _key(),
        "expect": dict(CURRENT_ROW),
        "price": _new_price(),
    }
    operation.update(overrides)
    return operation


def _add(**overrides):
    operation = {
        "op": "add_price",
        "key": _key(channel="vercel-ai-gateway", effective_from="2026-10-20"),
        "expect": None,
        "price": _new_price(channel="vercel-ai-gateway"),
    }
    operation.update(overrides)
    return operation


def _close(**overrides):
    operation = {
        "op": "close_price",
        "key": _key(),
        "expect": dict(CURRENT_ROW),
        "effective_to": "2026-10-20",
    }
    operation.update(overrides)
    return operation


def _delta(*operations, **overrides):
    delta = {
        "schema_version": CATALOG_DELTA_SCHEMA_VERSION,
        "kind": "pricing_delta",
        "provenance": {
            "observations": ["observation:openai-api-pricing:2026-10-20"],
            "evidence": ["sha256:" + "0" * 64],
        },
        "operations": list(operations),
    }
    delta.update(overrides)
    return delta


def _add_model(**overrides):
    canonical_id = overrides.pop("canonical_id", "example/model-b")
    operation = {
        "op": "add_model",
        "key": {"canonical_id": canonical_id},
        "expect": None,
        "model": {
            "canonical_id": canonical_id,
            "publisher": "example",
            "aliases": [
                {
                    "provider": "openrouter",
                    "alias": canonical_id,
                    "channel": "openrouter-api",
                }
            ],
            "prices": [
                {
                    "channel": "openrouter-api",
                    "region": "global",
                    "effective_from": "2026-10-20",
                    "input_per_mtok": "1",
                    "output_per_mtok": "4",
                    "source_url": "https://openrouter.ai/api/v1/models",
                }
            ],
        },
    }
    operation.update(overrides)
    return operation


def _add_alias(**overrides):
    alias = {
        "provider": "openrouter",
        "alias": "example/model-a",
        "channel": "openrouter-api",
    }
    alias.update(overrides.pop("alias", {}))
    operation = {
        "op": "add_alias",
        "key": {
            "canonical_id": overrides.pop("canonical_id", "example/model-a"),
            "provider": alias["provider"],
            "alias": alias["alias"],
            "channel": alias["channel"],
        },
        "expect": None,
        "alias": alias,
    }
    operation.update(overrides)
    return operation


# --- Parsing -----------------------------------------------------------------


def test_parse_reads_each_operation_with_its_row_key():
    delta = parse_catalog_delta(_delta(_replace(), _add(), _close(key=_key(region="us"), expect=dict(CURRENT_ROW, region="us"))))
    assert [op.op for op in delta.operations] == ["replace_price", "add_price", "close_price"]
    assert delta.operations[0].key == PriceRowKey(
        "example/model-a", "openai-api", "global", "2026-01-01"
    )
    assert delta.operations[1].expect is None
    assert delta.operations[2].effective_to == "2026-10-20"
    assert delta.provenance["observations"] == [
        "observation:openai-api-pricing:2026-10-20"
    ]


def test_parser_rejects_obsolete_schema_version_one():
    with pytest.raises(CatalogDeltaError, match="unsupported delta schema_version 1"):
        parse_catalog_delta(_delta(schema_version=1))


def test_catalog_schema_parses_model_and_alias_keys():
    parsed = parse_catalog_delta(_delta(_add_model(), _add_alias()))

    assert parsed.operations[0].key.canonical_id == "example/model-b"
    assert parsed.operations[0].model["publisher"] == "example"
    assert (
        parsed.operations[1].key.canonical_id,
        parsed.operations[1].key.provider,
        parsed.operations[1].key.alias,
        parsed.operations[1].key.channel,
    ) == ("example/model-a", "openrouter", "example/model-a", "openrouter-api")
    assert parsed.operations[1].alias["provider"] == "openrouter"


def test_add_model_requires_a_complete_valid_model_entry():
    operation = _add_model(model={"canonical_id": "example/model-b"})
    with pytest.raises(CatalogDeltaError, match="operation 0: invalid model"):
        parse_catalog_delta(_delta(operation))


def test_add_alias_identity_must_match_its_key():
    operation = _add_alias()
    operation["alias"] = dict(operation["alias"], alias="example/other")
    with pytest.raises(CatalogDeltaError, match="operation 0: alias alias must match key"):
        parse_catalog_delta(_delta(operation))


def test_delta_errors_are_catalog_errors():
    assert issubclass(CatalogDeltaError, CatalogError)


@pytest.mark.parametrize(
    ("delta", "message"),
    [
        (["not", "a", "mapping"], "delta document must be a mapping"),
        (_delta(schema_version=3), "unsupported delta schema_version 3"),
        (_delta(kind="catalog"), "delta kind must be pricing_delta"),
        (_delta(operations="none"), "delta operations must be a list"),
        (_delta(provenance=None), "delta needs provenance"),
        (_delta(provenance={"evidence": []}), "provenance needs an observations list"),
        (_delta(_replace(op="rewrite_price")), "operation 0: unknown op 'rewrite_price'"),
        (_delta("replace"), "operation 0: must be a mapping"),
    ],
)
def test_parse_rejects_malformed_delta_documents(delta, message):
    with pytest.raises(CatalogDeltaError, match=message):
        parse_catalog_delta(delta)


@pytest.mark.parametrize("field", ["canonical_id", "channel", "region", "effective_from"])
def test_parse_requires_the_full_row_key(field):
    key = _key()
    del key[field]
    with pytest.raises(CatalogDeltaError, match=f"operation 0: key needs {field}"):
        parse_catalog_delta(_delta(_replace(key=key)))


def test_parse_rejects_a_non_iso_key_date():
    with pytest.raises(CatalogDeltaError, match="operation 0: invalid key effective_from"):
        parse_catalog_delta(_delta(_replace(key=_key(effective_from="Jan 1"))))


def test_add_price_requires_a_null_expect():
    with pytest.raises(CatalogDeltaError, match="operation 0: add_price expect must be null"):
        parse_catalog_delta(_delta(_add(expect=dict(CURRENT_ROW))))
    operation = _add()
    del operation["expect"]
    with pytest.raises(CatalogDeltaError, match="operation 0: add_price expect must be null"):
        parse_catalog_delta(_delta(operation))


@pytest.mark.parametrize("make", [_replace, _close])
def test_replace_and_close_require_the_expected_row(make):
    with pytest.raises(CatalogDeltaError, match="expect must be the full current row"):
        parse_catalog_delta(_delta(make(expect=None)))


def test_add_price_identity_must_match_its_key():
    with pytest.raises(CatalogDeltaError, match="operation 0: price effective_from must match key"):
        parse_catalog_delta(_delta(_add(price=_new_price(channel="vercel-ai-gateway", effective_from="2026-10-21"))))
    with pytest.raises(CatalogDeltaError, match="operation 0: price channel must match key"):
        parse_catalog_delta(_delta(_add(price=_new_price())))


@pytest.mark.parametrize("field", ["channel", "region"])
def test_replace_price_keeps_channel_and_region(field):
    with pytest.raises(CatalogDeltaError, match=f"operation 0: price {field} must match key"):
        parse_catalog_delta(_delta(_replace(price=_new_price(**{field: "other"}))))


@pytest.mark.parametrize("make", [_replace, _add])
def test_new_price_requires_a_source_url(make):
    operation = make()
    operation["price"] = dict(operation["price"], source_url="  ")
    with pytest.raises(CatalogDeltaError, match="operation 0: price needs source_url"):
        parse_catalog_delta(_delta(operation))


@pytest.mark.parametrize("value", [1.25, 1, None, "1.2.5", "-1", "NaN"])
def test_new_rates_must_be_non_negative_decimal_strings(value):
    with pytest.raises(
        CatalogDeltaError,
        match="operation 0: price input_per_mtok must be a non-negative decimal string",
    ):
        parse_catalog_delta(_delta(_replace(price=_new_price(input_per_mtok=value))))


def test_new_price_rejects_unknown_fields_and_a_closing_date():
    with pytest.raises(CatalogDeltaError, match="operation 0: price has unknown field 'per_call'"):
        parse_catalog_delta(_delta(_replace(price=_new_price(per_call="1"))))
    with pytest.raises(CatalogDeltaError, match="operation 0: a new price row is open"):
        parse_catalog_delta(_delta(_replace(price=_new_price(effective_to="2027-01-01"))))


def test_new_price_rules_must_be_a_mapping():
    with pytest.raises(CatalogDeltaError, match="operation 0: price rules must be a mapping"):
        parse_catalog_delta(_delta(_replace(price=_new_price(rules=["x"]))))


def test_parse_rejects_a_non_iso_price_date():
    with pytest.raises(CatalogDeltaError, match="operation 0: invalid price effective_from"):
        parse_catalog_delta(_delta(_replace(price=_new_price(effective_from="tomorrow"))))


def test_close_price_needs_a_later_iso_effective_to():
    with pytest.raises(CatalogDeltaError, match="operation 0: invalid effective_to"):
        parse_catalog_delta(_delta(_close(effective_to=None)))
    with pytest.raises(CatalogDeltaError, match="operation 0: effective_to must follow key effective_from"):
        parse_catalog_delta(_delta(_close(effective_to="2026-01-01")))


def test_close_price_carries_no_new_row():
    with pytest.raises(CatalogDeltaError, match="operation 0: close_price takes no price"):
        parse_catalog_delta(_delta(_close(price=_new_price())))


def test_parse_rejects_two_operations_on_one_row():
    with pytest.raises(CatalogDeltaError, match="operation 1: duplicate target"):
        parse_catalog_delta(_delta(_replace(), _close()))


def test_parse_does_not_mutate_the_delta():
    delta = _delta(_replace(), _add())
    before = copy.deepcopy(delta)
    parse_catalog_delta(delta)
    assert delta == before


# --- Application -------------------------------------------------------------


def _document():
    return {
        "version": "2026-10-19",
        "currency": "USD",
        "pricing_verified_at": "2026-10-19",
        "models": [
            {
                "canonical_id": "example/model-a",
                "publisher": "example",
                "aliases": [
                    {"provider": "openai", "alias": "model-a", "channel": "openai-api"},
                    {"provider": "vercel", "alias": "example/model-a", "channel": "vercel-ai-gateway"},
                ],
                "prices": [
                    {
                        "channel": "openai-api",
                        "region": "global",
                        "effective_from": "2025-06-01",
                        "effective_to": "2026-01-01",
                        "input_per_mtok": 2.0,
                        "output_per_mtok": 8.0,
                        "source_url": "https://example.com/pricing-old",
                    },
                    dict(CURRENT_ROW),
                    dict(CURRENT_ROW, region="us", input_per_mtok=1.1, output_per_mtok=4.4),
                ],
            }
        ],
    }


def _rows(document, **match):
    [model] = document["models"]
    return [
        row
        for row in model["prices"]
        if all(row.get(field) == value for field, value in match.items())
    ]


def test_replace_price_closes_the_current_row_and_appends_the_new_row():
    result = apply_catalog_delta(_document(), _delta(_replace()))
    rows = _rows(result, channel="openai-api", region="global")
    assert rows == [
        _document()["models"][0]["prices"][0],
        dict(CURRENT_ROW, effective_to="2026-10-20"),
        _new_price(),
    ]


def test_replace_price_keeps_unrelated_rows_unchanged():
    result = apply_catalog_delta(_document(), _delta(_replace()))
    assert _rows(result, region="us") == _rows(_document(), region="us")
    assert result["models"][0]["aliases"] == _document()["models"][0]["aliases"]


def test_add_price_appends_a_row_for_a_new_channel():
    result = apply_catalog_delta(_document(), _delta(_add()))
    assert _rows(result, channel="vercel-ai-gateway") == [
        _new_price(channel="vercel-ai-gateway")
    ]
    assert len(result["models"][0]["prices"]) == 4


def _resolved_after(document, at="2026-10-21"):
    return load_catalog_document(document).snapshot.resolve_price(
        model="example/model-a",
        channel="openai-api",
        at=datetime.fromisoformat(at).replace(tzinfo=timezone.utc),
    )


def test_a_new_price_keeps_its_batch_cache_rate():
    price = _new_price(
        batch_input_per_mtok="0.625",
        batch_output_per_mtok="2.5",
        batch_cache_read_per_mtok="0.05",
    )
    result = apply_catalog_delta(_document(), _delta(_replace(price=price)))
    assert _rows(result, effective_from="2026-10-20") == [price]
    assert str(_resolved_after(result).price.batch_cache_read_per_mtok) == "0.05"


def test_a_zero_batch_cache_rate_is_a_rate():
    price = _new_price(batch_cache_read_per_mtok="0")
    result = apply_catalog_delta(_document(), _delta(_replace(price=price)))
    assert str(_resolved_after(result).price.batch_cache_read_per_mtok) == "0"


def test_a_new_price_without_a_batch_cache_rate_has_none():
    result = apply_catalog_delta(_document(), _delta(_replace()))
    assert "batch_cache_read_per_mtok" not in _rows(result, effective_from="2026-10-20")[0]
    assert _resolved_after(result).price.batch_cache_read_per_mtok is None


@pytest.mark.parametrize("value", ["-0.05", 0.05, "NaN", None])
def test_a_batch_cache_rate_must_be_a_non_negative_decimal_string(value):
    with pytest.raises(
        CatalogDeltaError,
        match="operation 0: price batch_cache_read_per_mtok must be a non-negative decimal string",
    ):
        parse_catalog_delta(
            _delta(_replace(price=_new_price(batch_cache_read_per_mtok=value)))
        )


def test_add_model_rejects_a_negative_batch_cache_rate():
    operation = _add_model()
    operation["model"]["prices"][0]["batch_cache_read_per_mtok"] = -0.05
    with pytest.raises(CatalogDeltaError, match="invalid model: .*batch_cache_read_per_mtok"):
        parse_catalog_delta(_delta(operation))


def test_add_model_appends_a_complete_priceable_entry():
    result = apply_catalog_delta(_document(), _delta(_add_model()))

    added = result["models"][-1]
    assert added == _add_model()["model"]
    loaded = load_catalog_document(result)
    found = loaded.lookup_price(
        model="example/model-b",
        provider="openrouter",
        at=datetime(2026, 10, 20, tzinfo=timezone.utc),
    )
    assert found.status == "priced"
    assert str(found.price.input_per_mtok) == "1"


def test_add_alias_and_price_make_an_existing_model_priceable_on_a_new_channel():
    price = _add(
        key=_key(channel="openrouter-api", effective_from="2026-10-20"),
        price=_new_price(channel="openrouter-api"),
    )
    result = apply_catalog_delta(
        _document(), _delta(_add_alias(), price)
    )

    loaded = load_catalog_document(result)
    found = loaded.lookup_price(
        model="example/model-a",
        provider="openrouter",
        at=datetime(2026, 10, 20, tzinfo=timezone.utc),
    )
    assert found.status == "priced"


def test_close_price_sets_only_effective_to():
    result = apply_catalog_delta(_document(), _delta(_close()))
    assert _rows(result, channel="openai-api", region="global")[1] == dict(
        CURRENT_ROW, effective_to="2026-10-20"
    )
    assert len(result["models"][0]["prices"]) == 3


def test_apply_preserves_version_and_pricing_verified_at():
    result = apply_catalog_delta(_document(), _delta(_replace()))
    assert result["version"] == "2026-10-19"
    assert result["pricing_verified_at"] == "2026-10-19"


def test_apply_never_mutates_the_document_or_the_delta():
    document = _document()
    delta = _delta(_replace(), _add())
    document_before = copy.deepcopy(document)
    delta_before = copy.deepcopy(delta)
    result = apply_catalog_delta(document, delta)
    assert document == document_before
    assert delta == delta_before
    result["models"][0]["prices"][-1]["input_per_mtok"] = "9"
    assert delta == delta_before


def test_applied_result_prices_the_new_rate_from_its_effective_date():
    loaded = load_catalog_document(apply_catalog_delta(_document(), _delta(_replace())))

    def input_rate(day):
        found = loaded.lookup_price(
            model="model-a",
            provider="openai",
            region="global",
            at=datetime(2026, 10, day, tzinfo=timezone.utc),
        )
        return str(found.price.input_per_mtok)

    assert input_rate(19) == "1.0"
    assert input_rate(20) == "1.25"


def _conflicts(document, delta):
    with pytest.raises(CatalogDeltaConflict) as raised:
        apply_catalog_delta(document, delta)
    return [(c.index, c.reason) for c in raised.value.conflicts]


def _new_model_key():
    return _key(
        canonical_id="example/new-model",
        channel="vercel-ai-gateway",
        effective_from="2026-10-20",
    )


def test_an_unknown_model_is_a_conflict():
    assert _conflicts(_document(), _delta(_add(key=_new_model_key()))) == [
        (0, "unknown_model")
    ]


def test_adding_an_existing_model_is_a_conflict():
    operation = _add_model(canonical_id="example/model-a")
    operation["model"]["canonical_id"] = "example/model-a"
    assert _conflicts(_document(), _delta(operation)) == [
        (0, "model_exists")
    ]


def test_adding_an_alias_to_an_unknown_model_is_a_conflict():
    assert _conflicts(
        _document(),
        _delta(_add_alias(canonical_id="example/missing")),
    ) == [(0, "unknown_model")]


def test_adding_an_existing_alias_is_a_conflict():
    operation = _add_alias(
        alias={
            "provider": "vercel",
            "alias": "example/model-a",
            "channel": "vercel-ai-gateway",
        }
    )
    assert _conflicts(_document(), _delta(operation)) == [
        (0, "alias_exists")
    ]


def test_adding_an_alias_owned_by_another_model_is_a_conflict():
    document = _document()
    document["models"].append(
        {
            "canonical_id": "example/model-b",
            "publisher": "example",
            "aliases": [
                {
                    "provider": "openrouter",
                    "alias": "example/shared",
                    "channel": "openrouter-api",
                }
            ],
            "prices": [
                {
                    "channel": "openrouter-api",
                    "region": "global",
                    "effective_from": "2026-01-01",
                    "input_per_mtok": 1,
                    "output_per_mtok": 4,
                    "source_url": "https://example.com/pricing",
                }
            ],
        }
    )
    operation = _add_alias(alias={"alias": "example/shared"})
    assert _conflicts(document, _delta(operation)) == [
        (0, "alias_conflict")
    ]


def test_catalog_addition_conflicts_are_atomic():
    document = _document()
    before = copy.deepcopy(document)
    delta = _delta(
        _add_model(),
        _add_alias(
            alias={
                "provider": "vercel",
                "alias": "example/model-a",
                "channel": "vercel-ai-gateway",
            }
        ),
    )

    assert _conflicts(document, delta) == [(1, "alias_exists")]
    assert document == before


def test_a_missing_row_is_a_conflict():
    key = _key(effective_from="2026-02-01")
    expect = dict(CURRENT_ROW, effective_from="2026-02-01")
    assert _conflicts(_document(), _delta(_replace(key=key, expect=expect))) == [
        (0, "row_missing")
    ]


def test_adding_an_existing_row_is_a_conflict():
    operation = _add(key=_key(), price=_new_price(effective_from="2026-01-01"))
    assert _conflicts(_document(), _delta(operation)) == [(0, "row_exists")]


def test_an_expect_that_differs_from_the_row_is_a_conflict():
    stale = dict(CURRENT_ROW, input_per_mtok=1.05)
    assert _conflicts(_document(), _delta(_replace(expect=stale))) == [
        (0, "expect_mismatch")
    ]


def test_an_expect_missing_a_field_is_a_conflict():
    partial = dict(CURRENT_ROW)
    del partial["source_url"]
    assert _conflicts(_document(), _delta(_close(expect=partial))) == [
        (0, "expect_mismatch")
    ]


def test_replacing_a_closed_row_is_a_conflict():
    closed = _document()["models"][0]["prices"][0]
    key = _key(effective_from="2025-06-01")
    assert _conflicts(_document(), _delta(_replace(key=key, expect=closed))) == [
        (0, "row_not_open")
    ]


@pytest.mark.parametrize("effective_from", ["2026-01-01", "2025-12-31"])
def test_a_replacement_not_after_the_current_row_is_a_conflict(effective_from):
    operation = _replace(price=_new_price(effective_from=effective_from))
    assert _conflicts(_document(), _delta(operation)) == [(0, "not_after_current")]


def test_a_duplicated_model_entry_is_a_conflict():
    document = _document()
    document["models"].append(copy.deepcopy(document["models"][0]))
    assert _conflicts(document, _delta(_replace())) == [(0, "duplicate_model")]


def test_a_duplicated_price_row_is_a_conflict():
    document = _document()
    document["models"][0]["prices"].append(dict(CURRENT_ROW))
    assert _conflicts(document, _delta(_replace())) == [(0, "duplicate_row")]


def test_every_conflict_is_collected_and_nothing_is_applied():
    document = _document()
    before = copy.deepcopy(document)
    delta = _delta(
        _replace(expect=dict(CURRENT_ROW, input_per_mtok=3)),
        _add(),
        _add(key=_new_model_key()),
        _close(key=_key(region="us"), expect=dict(CURRENT_ROW, region="us")),
    )
    assert _conflicts(document, delta) == [
        (0, "expect_mismatch"),
        (2, "unknown_model"),
        (3, "expect_mismatch"),
    ]
    assert document == before


def test_the_conflict_message_names_each_failing_change():
    with pytest.raises(
        CatalogDeltaConflict,
        match=(
            r"operation 0 replace_price example/model-a "
            r"openai-api/global@2026-01-01: expect_mismatch"
        ),
    ):
        apply_catalog_delta(_document(), _delta(_replace(expect={})))


def test_a_result_that_is_not_a_valid_catalog_is_rejected():
    # The new row overlaps the open global row, which the loader rejects.
    operation = _add(
        key=_key(effective_from="2026-06-01"),
        price=_new_price(effective_from="2026-06-01"),
    )
    with pytest.raises(
        CatalogDeltaError,
        match="delta result is not a valid catalog: .*overlapping openai-api/global",
    ):
        apply_catalog_delta(_document(), _delta(operation))


def test_applying_the_same_delta_twice_conflicts():
    delta = _delta(_replace())
    once = apply_catalog_delta(_document(), delta)
    assert _conflicts(once, delta) == [(0, "row_not_open")]


def test_a_malformed_delta_is_rejected_before_application():
    with pytest.raises(CatalogDeltaError, match="unsupported delta schema_version"):
        apply_catalog_delta(_document(), _delta(_replace(), schema_version=3))


def test_an_empty_delta_returns_an_equal_copy():
    document = _document()
    result = apply_catalog_delta(document, _delta())
    assert result == document
    assert result is not document


# --- Mixed date and offset inputs --------------------------------------------


def test_a_close_date_with_an_offset_is_compared_as_an_instant():
    # 2026-01-01T00:30:00+02:00 is 2025-12-31T22:30Z, before the row starts.
    with pytest.raises(CatalogDeltaError, match="effective_to must follow key effective_from"):
        parse_catalog_delta(_delta(_close(effective_to="2026-01-01T00:30:00+02:00")))


def test_a_naive_key_and_an_aware_close_date_parse():
    [operation] = parse_catalog_delta(_delta(_close(effective_to="2026-10-20T12:00:00+00:00"))).operations
    assert operation.effective_to == "2026-10-20T12:00:00+00:00"


def test_dates_naming_the_same_instant_are_one_key():
    with pytest.raises(CatalogDeltaError, match="operation 1: duplicate target"):
        parse_catalog_delta(
            _delta(
                _replace(),
                _close(key=_key(effective_from="2026-01-01T00:00:00+00:00")),
            )
        )


def test_a_utc_midnight_key_matches_a_date_only_row():
    operation = _close(key=_key(effective_from="2026-01-01T00:00:00Z"))
    result = apply_catalog_delta(_document(), _delta(operation))
    assert _rows(result, channel="openai-api", region="global")[1]["effective_to"] == "2026-10-20"


@pytest.mark.parametrize("effective_from", ["2026-01-01T00:00:00+01:00", "2025-12-31T23:59:59-00:00"])
def test_a_replacement_before_the_row_with_an_offset_is_a_conflict(effective_from):
    operation = _replace(price=_new_price(effective_from=effective_from))
    assert _conflicts(_document(), _delta(operation)) == [(0, "not_after_current")]


@pytest.mark.parametrize(
    "effective_from",
    ["2026-01-01T12:00:00Z", "2026-01-01T00:00:01+00:00", "2026-01-02T01:00:00+02:00"],
    ids=["noon", "one-second-later", "offset-into-the-same-utc-date"],
)
def test_a_replacement_on_the_targets_utc_date_is_a_conflict(effective_from):
    # Price ids are date-granular, so both rows would share one public id.
    operation = _replace(price=_new_price(effective_from=effective_from))
    assert _conflicts(_document(), _delta(operation)) == [(0, "same_day_replacement")]


def test_a_replacement_on_the_next_utc_date_keeps_distinct_price_ids():
    operation = _replace(price=_new_price(effective_from="2026-01-02T00:30:00Z"))
    loaded = load_catalog_document(apply_catalog_delta(_document(), _delta(operation)))
    ids = sorted(p.id for prices in loaded.snapshot._prices.values() for p in prices)
    assert ids == [
        "example/model-a:openai-api:global:2025-06-01",
        "example/model-a:openai-api:global:2026-01-01",
        "example/model-a:openai-api:global:2026-01-02",
        "example/model-a:openai-api:us:2026-01-01",
    ]


def _with_finite_gateway_row():
    """A gateway row that ran from 00:00 to 10:00 UTC on 20 October."""
    document = _document()
    document["models"][0]["prices"].append(
        _new_price(
            channel="vercel-ai-gateway",
            effective_from="2026-10-20T00:00:00Z",
            effective_to="2026-10-20T10:00:00Z",
        )
    )
    return document


def _add_gateway_row(effective_from):
    return _add(
        key=_key(channel="vercel-ai-gateway", effective_from=effective_from),
        price=_new_price(channel="vercel-ai-gateway", effective_from=effective_from),
    )


def test_adding_a_row_on_an_existing_rows_utc_date_is_a_price_id_collision():
    # The windows do not overlap, but both rows would get the id
    # example/model-a:vercel-ai-gateway:global:2026-10-20.
    document = _with_finite_gateway_row()
    before = copy.deepcopy(document)
    assert _conflicts(document, _delta(_add_gateway_row("2026-10-20T12:00:00Z"))) == [
        (0, "price_id_collision")
    ]
    assert document == before


def test_a_price_id_collision_ignores_other_channels_and_regions():
    # Rows on 2026-01-01 exist for openai-api global and us, not the gateway.
    result = apply_catalog_delta(_document(), _delta(_add_gateway_row("2026-01-01T12:00:00Z")))
    assert len(_rows(result, channel="vercel-ai-gateway")) == 1


def test_adding_a_row_on_the_next_utc_date_keeps_distinct_price_ids():
    applied = apply_catalog_delta(_with_finite_gateway_row(), _delta(_add_gateway_row("2026-10-21")))
    loaded = load_catalog_document(applied)
    ids = sorted(p.id for prices in loaded.snapshot._prices.values() for p in prices if p.pricing_channel == "vercel-ai-gateway")
    assert ids == [
        "example/model-a:vercel-ai-gateway:global:2026-10-20",
        "example/model-a:vercel-ai-gateway:global:2026-10-21",
    ]


def _loaded_ids(document):
    loaded = load_catalog_document(document)
    return [p.id for prices in loaded.snapshot._prices.values() for p in prices]


def _offset_document(effective_from, effective_to=None):
    """model-a's only global openai-api row, starting at an offset time."""
    document = _document()
    row = dict(CURRENT_ROW, effective_from=effective_from)
    if effective_to:
        row["effective_to"] = effective_to
    document["models"][0]["prices"] = [row, document["models"][0]["prices"][2]]
    return document, row


def test_replacing_an_offset_row_on_the_same_local_date_keeps_ids_unique():
    # 00:00 at +14:00 on 2 January is 10:00 UTC on 1 January, so a
    # replacement at midnight UTC on 2 January starts a UTC date later.
    document, row = _offset_document("2025-01-02T00:00:00+14:00")
    operation = _replace(
        key=_key(effective_from="2025-01-02T00:00:00+14:00"),
        expect=row,
        price=_new_price(effective_from="2025-01-02T00:00:00Z"),
    )
    ids = _loaded_ids(apply_catalog_delta(document, _delta(operation)))
    assert len(ids) == len(set(ids))
    assert sorted(i for i in ids if ":global:" in i) == [
        "example/model-a:openai-api:global:2025-01-01",
        "example/model-a:openai-api:global:2025-01-02",
    ]


def test_adding_beside_an_offset_row_on_its_utc_date_is_a_collision():
    # The finite row ran 10:00-11:00 UTC on 1 January; 12:00 UTC that day
    # does not overlap it but shares its UTC date, and so its id.
    document, _ = _offset_document("2025-01-02T00:00:00+14:00", "2025-01-01T11:00:00Z")
    operation = _add(key=_key(effective_from="2025-01-01T12:00:00Z"), price=_new_price(effective_from="2025-01-01T12:00:00Z"))
    assert _conflicts(document, _delta(operation)) == [(0, "price_id_collision")]


def test_adding_beside_an_offset_row_on_its_local_date_keeps_ids_unique():
    document, _ = _offset_document("2025-01-02T00:00:00+14:00", "2025-01-01T11:00:00Z")
    operation = _add(key=_key(effective_from="2025-01-02T00:00:00Z"), price=_new_price(effective_from="2025-01-02T00:00:00Z"))
    ids = _loaded_ids(apply_catalog_delta(document, _delta(operation)))
    assert len(ids) == len(set(ids))


def test_a_replacement_stores_the_canonical_channel_region_and_source_url():
    padded = _new_price(channel=" openai-api ", region="global\t", source_url="  https://example.com/pricing  ")
    result = apply_catalog_delta(_document(), _delta(_replace(price=padded)))
    new_row = result["models"][0]["prices"][-2]
    assert (new_row["channel"], new_row["region"], new_row["source_url"]) == (
        "openai-api", "global", "https://example.com/pricing"
    )
    loaded = load_catalog_document(result)
    found = loaded.lookup_price(
        model="model-a", provider="openai", region="global",
        at=datetime(2026, 10, 21, tzinfo=timezone.utc),
    )
    assert str(found.price.input_per_mtok) == "1.25"


def test_an_added_row_stores_the_canonical_channel_and_region():
    operation = _add(price=_new_price(channel="vercel-ai-gateway ", region=" global"))
    result = apply_catalog_delta(_document(), _delta(operation))
    [row] = [r for r in result["models"][0]["prices"] if r["channel"].strip() == "vercel-ai-gateway"]
    assert (row["channel"], row["region"]) == ("vercel-ai-gateway", "global")


def test_a_replacement_with_an_offset_after_the_row_applies():
    operation = _replace(price=_new_price(effective_from="2026-10-20T00:00:00+00:00"))
    result = apply_catalog_delta(_document(), _delta(operation))
    rows = _rows(result, channel="openai-api", region="global")
    assert rows[1]["effective_to"] == "2026-10-20"
    assert rows[2]["effective_from"] == "2026-10-20"


# --- Moving an alias between models ------------------------------------------

GATEWAY_ALIAS = {
    "provider": "openrouter",
    "alias": "example-host/model-a-instruct",
    "channel": "openrouter-api",
}
GATEWAY_ROW = {
    "channel": "openrouter-api",
    "region": "global",
    "effective_from": "2026-01-01",
    "input_per_mtok": 0.5,
    "output_per_mtok": 2.0,
    "source_url": "https://example.com/gateway/models",
}
MOVE_SOURCE = "https://example.com/catalog-review"


def _with_duplicate():
    """model-a plus a duplicate entry a gateway spelling created for it."""
    document = _document()
    document["models"].append(
        {
            "canonical_id": "example-host/model-a-instruct",
            "publisher": "example-host",
            "aliases": [dict(GATEWAY_ALIAS)],
            "prices": [dict(GATEWAY_ROW)],
        }
    )
    return document


def _close_alias(**overrides):
    operation = {
        "op": "close_alias",
        "key": dict(GATEWAY_ALIAS, canonical_id="example-host/model-a-instruct"),
        "expect": dict(GATEWAY_ALIAS),
        "effective_to": "2026-10-20",
        "source_url": MOVE_SOURCE,
    }
    operation.update(overrides)
    return operation


def _move_operations():
    """The reviewed delta that folds the duplicate into model-a on 2026-10-20."""
    return [
        _close_alias(),
        _close(
            key=_key(
                canonical_id="example-host/model-a-instruct",
                channel="openrouter-api",
            ),
            expect=dict(GATEWAY_ROW),
        ),
        _add_alias(
            alias=dict(
                GATEWAY_ALIAS,
                effective_from="2026-10-20",
                source_url=MOVE_SOURCE,
            )
        ),
        _add(
            key=_key(channel="openrouter-api", effective_from="2026-10-20"),
            price=_new_price(
                channel="openrouter-api",
                input_per_mtok="0.5",
                output_per_mtok="2",
                source_url="https://example.com/gateway/models",
            ),
        ),
    ]


def _resolved(document, at):
    result = load_catalog_document(document).snapshot.cost(
        provider="openrouter",
        model="example-host/model-a-instruct",
        at=at,
        input_tokens=1_000_000,
        output_tokens=0,
    )
    return result.canonical_model, result.cost_usd


def test_a_gateway_spelling_moves_to_another_model_from_a_date():
    document = _with_duplicate()
    before = copy.deepcopy(document)
    moved = apply_catalog_delta(document, _delta(*_move_operations()))
    assert document == before

    duplicate = moved["models"][1]
    assert duplicate["aliases"] == [
        dict(GATEWAY_ALIAS, effective_to="2026-10-20", source_url=MOVE_SOURCE)
    ]
    assert duplicate["prices"] == [dict(GATEWAY_ROW, effective_to="2026-10-20")]
    # Calls before the move keep the model and price they were billed under;
    # calls from the move resolve to the surviving model.
    assert _resolved(moved, datetime(2026, 10, 19, tzinfo=timezone.utc)) == (
        "example-host/model-a-instruct",
        Decimal("0.5"),
    )
    assert _resolved(moved, datetime(2026, 10, 21, tzinfo=timezone.utc)) == (
        "example/model-a",
        Decimal("0.5"),
    )


def test_a_spelling_another_model_holds_stays_a_conflict_without_a_close():
    operations = _move_operations()
    assert _conflicts(_with_duplicate(), _delta(*operations[2:])) == [
        (0, "alias_conflict")
    ]


def test_a_spelling_whose_alias_already_ended_can_be_added_from_that_date():
    document = _with_duplicate()
    document["models"][1]["aliases"] = [
        dict(GATEWAY_ALIAS, effective_to="2026-10-20", source_url=MOVE_SOURCE)
    ]
    added = apply_catalog_delta(
        document, _delta(_move_operations()[2])
    )
    assert added["models"][0]["aliases"][-1]["effective_from"] == "2026-10-20"
    # Starting before the old alias ends would give one spelling two models.
    early = _add_alias(
        alias=dict(
            GATEWAY_ALIAS, effective_from="2026-10-19", source_url=MOVE_SOURCE
        )
    )
    assert _conflicts(document, _delta(early)) == [(0, "alias_conflict")]


def test_close_alias_sets_only_the_end_and_a_missing_source():
    cited = _with_duplicate()
    cited["models"][1]["aliases"] = [dict(GATEWAY_ALIAS, source_url=MOVE_SOURCE)]
    operation = _close_alias(expect=dict(GATEWAY_ALIAS, source_url=MOVE_SOURCE))
    del operation["source_url"]
    closed = apply_catalog_delta(cited, _delta(operation))
    assert closed["models"][1]["aliases"] == [
        dict(GATEWAY_ALIAS, source_url=MOVE_SOURCE, effective_to="2026-10-20")
    ]


@pytest.mark.parametrize(
    ("document_alias", "overrides", "reason"),
    [
        (GATEWAY_ALIAS, {"key": dict(GATEWAY_ALIAS, canonical_id="example/missing")}, "unknown_model"),
        (GATEWAY_ALIAS, {"key": dict(GATEWAY_ALIAS, canonical_id="example-host/model-a-instruct", alias="example-host/other")}, "alias_missing"),
        (GATEWAY_ALIAS, {"expect": dict(GATEWAY_ALIAS, channel="vercel-ai-gateway")}, "expect_mismatch"),
        (
            dict(GATEWAY_ALIAS, effective_to="2026-09-01", source_url=MOVE_SOURCE),
            {"expect": dict(GATEWAY_ALIAS, effective_to="2026-09-01", source_url=MOVE_SOURCE), "source_url": None},
            "alias_not_open",
        ),
        (
            dict(GATEWAY_ALIAS, effective_from="2026-11-01", source_url=MOVE_SOURCE),
            {"expect": dict(GATEWAY_ALIAS, effective_from="2026-11-01", source_url=MOVE_SOURCE), "source_url": None},
            "not_after_start",
        ),
        (GATEWAY_ALIAS, {"source_url": None}, "source_url_needed"),
        (
            dict(GATEWAY_ALIAS, source_url=MOVE_SOURCE),
            {"expect": dict(GATEWAY_ALIAS, source_url=MOVE_SOURCE), "source_url": "https://example.com/other"},
            "source_url_exists",
        ),
    ],
)
def test_close_alias_preconditions(document_alias, overrides, reason):
    document = _with_duplicate()
    document["models"][1]["aliases"] = [dict(document_alias)]
    operation = _close_alias(**overrides)
    if operation.get("source_url") is None:
        operation.pop("source_url")
    assert _conflicts(document, _delta(operation)) == [(0, reason)]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expect": None}, "expect must be the full current row"),
        ({"effective_to": "soon"}, "invalid effective_to"),
        ({"source_url": "  "}, "invalid source_url"),
        ({"price": _new_price()}, "close_alias takes only effective_to and source_url"),
        ({"key": dict(GATEWAY_ALIAS, canonical_id="example/model-a", region="global")}, "close_alias key has unknown fields"),
    ],
)
def test_close_alias_schema(overrides, message):
    with pytest.raises(CatalogDeltaError, match=message):
        parse_catalog_delta(_delta(_close_alias(**overrides)))


def _reopened():
    """The duplicate's spelling closed on 2026-09-01 and reopened on
    2026-09-15: two windows under one close_alias key."""
    document = _with_duplicate()
    document["models"][1]["aliases"] = [
        dict(GATEWAY_ALIAS, effective_to="2026-09-01", source_url=MOVE_SOURCE),
        dict(GATEWAY_ALIAS, effective_from="2026-09-15", source_url=MOVE_SOURCE),
    ]
    return document


def test_close_alias_ends_the_window_its_expect_names():
    reopened = dict(GATEWAY_ALIAS, effective_from="2026-09-15", source_url=MOVE_SOURCE)
    operation = _close_alias(expect=reopened)
    del operation["source_url"]
    closed = apply_catalog_delta(_reopened(), _delta(operation))
    assert closed["models"][1]["aliases"] == [
        dict(GATEWAY_ALIAS, effective_to="2026-09-01", source_url=MOVE_SOURCE),
        dict(reopened, effective_to="2026-10-20"),
    ]


def test_a_move_applies_with_the_add_before_the_close():
    reopened = dict(GATEWAY_ALIAS, effective_from="2026-09-15", source_url=MOVE_SOURCE)
    close = _close_alias(expect=reopened)
    del close["source_url"]
    # The add lands on the same model and key as the close it follows.
    reopen = _add_alias(
        canonical_id="example-host/model-a-instruct",
        alias=dict(GATEWAY_ALIAS, effective_from="2026-11-01", source_url=MOVE_SOURCE),
    )
    applied = apply_catalog_delta(_reopened(), _delta(reopen, close))
    assert applied["models"][1]["aliases"][1:] == [
        dict(reopened, effective_to="2026-10-20"),
        dict(GATEWAY_ALIAS, effective_from="2026-11-01", source_url=MOVE_SOURCE),
    ]


def test_a_close_alias_expect_matching_no_window_is_a_mismatch():
    operation = _close_alias(expect=dict(GATEWAY_ALIAS, source_url=MOVE_SOURCE))
    del operation["source_url"]
    assert _conflicts(_reopened(), _delta(operation)) == [(0, "expect_mismatch")]
