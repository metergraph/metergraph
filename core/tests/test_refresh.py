"""Pure low-risk refresh planning and discovery evidence over supplied
content. Nothing here fetches or applies a delta."""

import copy
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from metergraph_core import load_catalog_document
from metergraph_core.catalog_delta import apply_catalog_delta, parse_catalog_delta
from metergraph_core.extractors import structural_fingerprint
from metergraph_core.refresh import discovery_evidence, plan_low_risk_refresh
from metergraph_core.sources import parse_source_registry

FIXTURES = Path(__file__).parent / "fixtures" / "sources"
OBSERVED_ON = date(2026, 10, 20)
OPENAI_URL = "https://example.com/openai-pricing"


def _fixture(source_id):
    return (FIXTURES / f"{source_id}.html").read_text()


def _source(source_id, parser, channel, provider, *, low_risk=True, fingerprint="fixture", dimensions=None, low_risk_dimensions=None, url=None):
    dimensions = dimensions or ["input_per_mtok", "cache_read_per_mtok", "cache_write_5m_per_mtok", "output_per_mtok"]
    if fingerprint == "fixture":
        fingerprint = structural_fingerprint(parser, _fixture(source_id))
    return {
        "id": source_id,
        "provider": provider,
        "channel": channel,
        "catalog_channel": True,
        "url": url or f"https://example.com/{parser}-pricing",
        "parser": parser,
        "dimensions": dimensions,
        "fingerprint": fingerprint,
        "low_risk": (
            {"eligible": True, "dimensions": low_risk_dimensions or dimensions}
            if low_risk
            else {"eligible": False, "dimensions": []}
        ),
    }


def _registry(*sources):
    return parse_source_registry({"schema_version": 1, "sources": list(sources) or [_openai_source()]})


def _openai_source(**overrides):
    return _source("openai-api-pricing", "openai", "openai-api", "openai", url=OPENAI_URL, **overrides)


def _row(**rates):
    row = {"channel": "openai-api", "region": "global", "effective_from": "2026-01-01"}
    row.update(rates)
    row["source_url"] = "https://example.com/model-page"
    return row


def _model(canonical_id, alias, *rows, channel="openai-api", provider="openai"):
    return {
        "canonical_id": canonical_id,
        "publisher": "example",
        "aliases": [{"provider": provider, "alias": alias, "channel": channel}],
        "prices": list(rows),
    }


def _document(*models):
    # Every price in this catalog matches the OpenAI fixture.
    return {
        "version": "2026-10-19",
        "currency": "USD",
        "pricing_verified_at": "2026-10-19",
        "models": list(models)
        or [
            _model("example/gpt-x", "gpt-x", _row(input_per_mtok=2.0, cache_read_per_mtok=0.2, cache_write_5m_per_mtok=2.5, output_per_mtok=8.0)),
            _model("example/gpt-x-mini", "gpt-x-mini", _row(input_per_mtok=0.4, cache_read_per_mtok=0.04, output_per_mtok=1.6)),
            _model("example/gpt-legacy", "gpt-legacy", _row(input_per_mtok=2.5, cache_read_per_mtok=1.25, output_per_mtok=10.0)),
        ],
    }


def _with_gpt_x(**rates):
    document = _document()
    document["models"][0]["prices"][0] = _row(**rates)
    return document


GPT_X = dict(input_per_mtok=2.0, cache_read_per_mtok=0.2, cache_write_5m_per_mtok=2.5, output_per_mtok=8.0)


def _plan(document=None, registry=None, contents=None, observed_on=OBSERVED_ON):
    return plan_low_risk_refresh(
        document if document is not None else _document(),
        registry or _registry(),
        contents if contents is not None else {"openai-api-pricing": _fixture("openai-api-pricing")},
        observed_on=observed_on,
    )


def _reasons(result):
    # A row-level escalation names its catalog row; a page-only one its model.
    return [(e.source_id, e.reason, e.key.canonical_id if e.key else e.model_key) for e in result.escalations]


def test_unchanged_content_yields_an_empty_delta_and_an_unchanged_observation():
    result = _plan()
    assert result.delta["operations"] == []
    assert result.escalations == ()
    [observation] = result.observations
    assert observation.source_id == "openai-api-pricing"
    assert observation.result == "unchanged"
    assert observation.rows_compared == 3
    assert observation.changes == 0


def test_a_change_replaces_the_row_with_its_exact_current_state():
    document = _with_gpt_x(**dict(GPT_X, input_per_mtok=2.5))
    result = _plan(document)
    assert result.escalations == ()
    assert result.delta["operations"] == [
        {
            "op": "replace_price",
            "key": {"canonical_id": "example/gpt-x", "channel": "openai-api", "region": "global", "effective_from": "2026-01-01"},
            "expect": _row(**dict(GPT_X, input_per_mtok=2.5)),
            "price": {
                "channel": "openai-api",
                "region": "global",
                "effective_from": "2026-10-20",
                "input_per_mtok": "2",
                "cache_read_per_mtok": "0.2",
                "cache_write_5m_per_mtok": "2.5",
                "output_per_mtok": "8",
                "source_url": OPENAI_URL,
            },
        }
    ]
    assert result.observations[0].result == "changed"
    assert result.observations[0].changes == 1


@pytest.mark.parametrize(("current", "proposed"), [(1.0, "2"), (10.0, "2")], ids=["doubled", "cut-by-80-percent"])
def test_a_large_change_is_proposed_when_every_invariant_holds(current, proposed):
    # No relative-change threshold: a reviewed source on an existing row and
    # dimension is trusted for the size of the change.
    result = _plan(_with_gpt_x(**dict(GPT_X, input_per_mtok=current)))
    assert result.escalations == ()
    [operation] = result.delta["operations"]
    assert operation["price"]["input_per_mtok"] == proposed
    assert result.observations[0].result == "changed"


def test_one_escalated_rate_holds_back_the_whole_row():
    # input changes within the allowed dimensions; cache_read is not allowed.
    registry = _registry(_openai_source(low_risk_dimensions=["input_per_mtok", "output_per_mtok"]))
    result = _plan(_with_gpt_x(**dict(GPT_X, input_per_mtok=1.0, cache_read_per_mtok=0.25)), registry)
    assert result.delta["operations"] == []
    assert [e.reason for e in result.escalations] == ["dimension_not_allowed"]


def test_rows_are_independent():
    document = _with_gpt_x(**dict(GPT_X, input_per_mtok=-1.0))
    document["models"][1]["prices"][0]["output_per_mtok"] = 2.0
    result = _plan(document)
    assert [op["key"]["canonical_id"] for op in result.delta["operations"]] == ["example/gpt-x-mini"]
    assert [e.reason for e in result.escalations] == ["invalid_current_rate"]
    assert result.observations[0].result == "changed_and_escalated"


def test_the_new_row_starts_on_the_utc_observation_date():
    result = _plan(_with_gpt_x(**dict(GPT_X, output_per_mtok=9.0)), observed_on=date(2026, 11, 3))
    [operation] = result.delta["operations"]
    assert operation["price"]["effective_from"] == "2026-11-03"


def test_a_row_that_started_on_the_observation_date_escalates():
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0))
    document["models"][0]["prices"][0]["effective_from"] = "2026-10-20"
    result = _plan(document)
    assert result.delta["operations"] == []
    assert _reasons(result) == [("openai-api-pricing", "not_after_current", "example/gpt-x")]


def test_a_second_change_to_one_row_on_the_same_utc_day_escalates():
    first = _plan(_with_gpt_x(**dict(GPT_X, output_per_mtok=9.0)))
    applied = apply_catalog_delta(_with_gpt_x(**dict(GPT_X, output_per_mtok=9.0)), first.delta)
    changed_again = _fixture("openai-api-pricing").replace("[0,2.5],[0,8]", "[0,2.5],[0,8.5]")
    second = _plan(applied, contents={"openai-api-pricing": changed_again})
    assert second.delta["operations"] == []
    assert [e.reason for e in second.escalations] == ["not_after_current"]


def test_a_dimension_the_row_lacks_escalates_as_new():
    document = _document()
    del document["models"][1]["prices"][0]["cache_read_per_mtok"]
    result = _plan(document)
    assert result.delta["operations"] == []
    assert [(e.reason, e.dimension) for e in result.escalations] == [("new_dimension", "cache_read_per_mtok")]


def test_a_change_outside_the_low_risk_dimensions_escalates():
    registry = _registry(_openai_source(low_risk_dimensions=["input_per_mtok", "output_per_mtok"]))
    result = _plan(_with_gpt_x(**dict(GPT_X, cache_read_per_mtok=0.25)), registry)
    assert result.delta["operations"] == []
    assert [(e.reason, e.dimension) for e in result.escalations] == [("dimension_not_allowed", "cache_read_per_mtok")]


def test_a_change_to_a_row_with_rates_the_source_does_not_represent_escalates():
    result = _plan(_with_gpt_x(**dict(GPT_X, output_per_mtok=9.0, batch_input_per_mtok=1.0)))
    assert result.delta["operations"] == []
    assert [(e.reason, e.dimension) for e in result.escalations] == [("unrepresented_dimension", "batch_input_per_mtok")]


def test_a_change_to_a_row_with_a_batch_cache_rate_escalates_instead_of_dropping_it():
    # No source reads batch cache rates, so an automated replacement could
    # not carry the rate forward with the others.
    result = _plan(_with_gpt_x(**dict(GPT_X, output_per_mtok=9.0, batch_cache_read_per_mtok=0.05)))
    assert result.delta["operations"] == []
    assert [(e.reason, e.dimension) for e in result.escalations] == [("unrepresented_dimension", "batch_cache_read_per_mtok")]


def test_an_unchanged_row_with_unrepresented_rates_is_fine():
    result = _plan(_with_gpt_x(**dict(GPT_X, batch_input_per_mtok=1.0)))
    assert result.escalations == ()


def test_a_model_the_catalog_lacks_escalates_as_unknown():
    document = _document()
    del document["models"][2]
    result = _plan(document)
    assert result.delta["operations"] == []
    assert _reasons(result) == [("openai-api-pricing", "unknown_model", "gpt-legacy")]


def test_a_catalog_row_the_page_no_longer_lists_escalates():
    document = _document()
    document["models"].append(_model("example/gpt-gone", "gpt-gone", _row(input_per_mtok=1.0, output_per_mtok=2.0)))
    result = _plan(document)
    assert _reasons(result) == [("openai-api-pricing", "model_not_listed", "example/gpt-gone")]


def test_a_region_the_catalog_lacks_escalates_as_new():
    source = _source("xai-api-pricing", "xai", "xai-api", "xai", dimensions=["input_per_mtok", "output_per_mtok", "cache_read_per_mtok"])
    document = _document(
        _model("example/grok-x", "grok-x", dict(_row(input_per_mtok=2.0, cache_read_per_mtok=0.5, output_per_mtok=6.0), channel="xai-api"), channel="xai-api", provider="x-ai")
    )
    result = plan_low_risk_refresh(document, _registry(source), {"xai-api-pricing": _fixture("xai-api-pricing")}, observed_on=OBSERVED_ON)
    assert result.delta["operations"] == []
    [escalation] = result.escalations
    assert (escalation.reason, escalation.model_key, escalation.region) == ("new_region", "grok-x", "us")


def test_a_source_that_is_not_low_risk_escalates_without_comparing():
    registry = _registry(_openai_source(low_risk=False))
    result = _plan(_with_gpt_x(**dict(GPT_X, output_per_mtok=9.0)), registry)
    assert result.delta["operations"] == []
    assert _reasons(result) == [("openai-api-pricing", "not_low_risk", None)]


def test_a_missing_fingerprint_escalates():
    result = _plan(_with_gpt_x(**dict(GPT_X, output_per_mtok=9.0)), _registry(_openai_source(fingerprint=None)))
    assert result.delta["operations"] == []
    assert [e.reason for e in result.escalations] == ["fingerprint_missing"]


def test_a_mismatched_fingerprint_escalates():
    registry = _registry(_openai_source(fingerprint="sha256:" + "0" * 64))
    result = _plan(_with_gpt_x(**dict(GPT_X, output_per_mtok=9.0)), registry)
    assert result.delta["operations"] == []
    [escalation] = result.escalations
    assert escalation.reason == "fingerprint_mismatch"
    assert result.observations[0].fingerprint == structural_fingerprint("openai", _fixture("openai-api-pricing"))


def test_reordered_openai_columns_block_a_low_risk_refresh():
    page = _fixture("openai-api-pricing")
    reordered = page.replace("<th>Cached input</th>", "\0").replace("<th>Cache writes</th>", "<th>Cached input</th>").replace("\0", "<th>Cache writes</th>")
    result = _plan(_with_gpt_x(**dict(GPT_X, output_per_mtok=9.0)), contents={"openai-api-pricing": reordered})
    assert result.delta["operations"] == []
    assert [e.reason for e in result.escalations] == ["fingerprint_mismatch"]


def test_missing_content_for_a_low_risk_source_is_observed_and_escalated():
    result = _plan(contents={})
    assert [e.reason for e in result.escalations] == ["content_missing"]
    assert result.observations[0].result == "missing"


def test_every_source_in_the_given_registry_is_observed_even_without_content():
    # The registry passed in is the selected scope, low-risk or not.
    registry = _registry(_openai_source(), _openai_source(low_risk=False) | {"id": "other-pricing", "url": "https://example.com/other"})
    result = _plan(registry=registry)
    assert [(o.source_id, o.result) for o in result.observations] == [
        ("openai-api-pricing", "unchanged"),
        ("other-pricing", "missing"),
    ]
    assert [(e.source_id, e.reason) for e in result.escalations] == [("other-pricing", "content_missing")]


MIRROR_URL = "https://example.com/openai-mirror"


def _two_openai_sources():
    mirror = dict(_openai_source(), id="openai-mirror-pricing", url=MIRROR_URL)
    return _registry(_openai_source(), mirror)


def _plan_two(document, mirror_page):
    return _plan(
        document,
        _two_openai_sources(),
        {"openai-api-pricing": _fixture("openai-api-pricing"), "openai-mirror-pricing": mirror_page},
    )


def test_two_sources_agreeing_on_one_row_change_yield_no_duplicate_operation():
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0))
    result = _plan_two(document, _fixture("openai-api-pricing"))
    assert result.delta["operations"] == []
    assert [(e.source_id, e.reason, e.key.canonical_id) for e in result.escalations] == [
        ("openai-api-pricing", "conflicting_sources", "example/gpt-x"),
        ("openai-mirror-pricing", "conflicting_sources", "example/gpt-x"),
    ]
    assert result.escalations[0].detail == (
        "openai-api-pricing and openai-mirror-pricing both propose this row, with the same values"
    )
    assert [(o.source_id, o.result, o.changes) for o in result.observations] == [
        ("openai-api-pricing", "escalated", 0),
        ("openai-mirror-pricing", "escalated", 0),
    ]
    parse_catalog_delta(result.delta)


def test_two_sources_disagreeing_on_one_row_change_yield_no_operation():
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0))
    mirror = _fixture("openai-api-pricing").replace("[0,2],[0,0.2],[0,2.5],[0,8]", "[0,2],[0,0.2],[0,2.5],[0,8.5]")
    result = _plan_two(document, mirror)
    assert result.delta["operations"] == []
    assert [(e.source_id, e.reason) for e in result.escalations] == [
        ("openai-api-pricing", "conflicting_sources"),
        ("openai-mirror-pricing", "conflicting_sources"),
    ]
    assert result.escalations[1].detail == (
        "openai-api-pricing and openai-mirror-pricing both propose this row, with different values"
    )
    assert result.escalations[0].observed == "8"
    assert result.escalations[1].observed == "8.5"


def test_a_conflict_on_one_row_leaves_other_rows_operations():
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0))
    document["models"][1]["prices"][0]["output_per_mtok"] = 2.0
    mirror = _fixture("openai-api-pricing").replace("[0,0.4],[0,0.04],[0,1.6]", "[0,0.4],[0,0.04],[0,2]")
    # The mirror agrees gpt-x-mini is unchanged, so only openai-api-pricing
    # proposes that row; both propose gpt-x.
    result = _plan_two(document, mirror)
    assert [op["key"]["canonical_id"] for op in result.delta["operations"]] == ["example/gpt-x-mini"]
    assert [(o.source_id, o.result, o.changes) for o in result.observations] == [
        ("openai-api-pricing", "changed_and_escalated", 1),
        ("openai-mirror-pricing", "escalated", 0),
    ]
    applied = apply_catalog_delta(document, result.delta)
    assert load_catalog_document(applied).version == "2026-10-19"


def test_content_for_a_source_outside_the_registry_is_rejected():
    with pytest.raises(ValueError, match="unknown source 'stray-pricing'"):
        _plan(contents={"openai-api-pricing": _fixture("openai-api-pricing"), "stray-pricing": "<p></p>"})


def test_a_non_positive_rate_escalates():
    content = _fixture("openai-api-pricing").replace("[0,2],[0,0.2],[0,2.5],[0,8]", "[0,0],[0,0.2],[0,2.5],[0,8]")
    result = _plan(contents={"openai-api-pricing": content})
    assert result.delta["operations"] == []
    assert [(e.reason, e.dimension) for e in result.escalations] == [("non_positive_rate", "input_per_mtok")]


@pytest.mark.parametrize(
    "current",
    [-1.0, 0.0, "-0.5", float("nan"), float("inf"), "NaN", "-Infinity"],
    ids=["negative", "zero", "negative-text", "nan", "infinite", "nan-text", "negative-infinite-text"],
)
def test_an_invalid_current_rate_escalates(current):
    # The catalog loader accepts these; the planner must not continue them.
    document = _with_gpt_x(**dict(GPT_X, input_per_mtok=current))
    result = _plan(document)
    assert result.delta["operations"] == []
    [escalation] = result.escalations
    assert (escalation.reason, escalation.dimension, escalation.observed) == (
        "invalid_current_rate", "input_per_mtok", "2"
    )


def test_a_missing_required_rate_escalates():
    source = _source("anthropic-api-pricing", "anthropic", "anthropic-api", "anthropic")
    document = _document(
        _model("example/claude-example-5", "claude-example-5", dict(_row(input_per_mtok=5.0, output_per_mtok=25.0, cache_write_5m_per_mtok=6.25, cache_read_per_mtok=0.5), channel="anthropic-api"), channel="anthropic-api", provider="anthropic"),
        _model("example/claude-example-5-5", "claude-example-5-5", dict(_row(input_per_mtok=4.0, output_per_mtok=20.0, cache_write_5m_per_mtok=5.0, cache_read_per_mtok=0.4), channel="anthropic-api"), channel="anthropic-api", provider="anthropic"),
    )
    content = _fixture("anthropic-api-pricing").replace("<td>$25 / MTok</td>", "<td>Contact us</td>")
    result = plan_low_risk_refresh(document, _registry(source), {"anthropic-api-pricing": content}, observed_on=OBSERVED_ON)
    assert [(e.reason, e.dimension) for e in result.escalations] == [("missing_rate", "output_per_mtok")]


def test_an_ambiguous_parse_escalates_the_row():
    content = _fixture("openai-api-pricing").replace(
        "[1,[[0,&quot;gpt-x-mini&quot;]",
        "[1,[[0,&quot;gpt-x&quot;],[0,2.2],[0,0.2],[0,2.5],[0,8.8]]][1,[[0,&quot;gpt-x-mini&quot;]",
    )
    result = _plan(_with_gpt_x(**dict(GPT_X, output_per_mtok=9.0)), contents={"openai-api-pricing": content})
    assert result.delta["operations"] == []
    assert _reasons(result) == [("openai-api-pricing", "ambiguous_parse", "example/gpt-x")]


def test_a_catalog_with_a_duplicated_model_entry_is_rejected():
    document = _document()
    duplicate = _model("example/gpt-x", "gpt-x-alt", dict(_row(input_per_mtok=1.0, output_per_mtok=1.0), channel="vercel-ai-gateway"), channel="vercel-ai-gateway")
    document["models"].append(duplicate)
    with pytest.raises(ValueError, match="duplicate model entry 'example/gpt-x'"):
        _plan(document)


def test_the_delta_references_its_observations_and_evidence():
    result = _plan(_with_gpt_x(**dict(GPT_X, output_per_mtok=9.0)))
    assert result.delta["schema_version"] == 2
    assert result.delta["kind"] == "pricing_delta"
    assert result.delta["provenance"]["observations"] == ["openai-api-pricing@2026-10-20"]
    assert result.delta["provenance"]["evidence"] == [
        {
            "source_id": "openai-api-pricing",
            "url": OPENAI_URL,
            "fingerprint": structural_fingerprint("openai", _fixture("openai-api-pricing")),
        }
    ]
    assert result.observations[0].id == "openai-api-pricing@2026-10-20"


def test_every_low_risk_source_gets_an_observation_even_without_changes():
    anthropic = _source("anthropic-api-pricing", "anthropic", "anthropic-api", "anthropic")
    result = _plan(registry=_registry(_openai_source(), anthropic))
    assert [(o.source_id, o.result) for o in result.observations] == [
        ("openai-api-pricing", "unchanged"),
        ("anthropic-api-pricing", "missing"),
    ]


def test_the_refresh_delta_applies_cleanly():
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0))
    applied = apply_catalog_delta(document, _plan(document).delta)
    rows = applied["models"][0]["prices"]
    assert [(row["effective_from"], row.get("effective_to")) for row in rows] == [
        ("2026-01-01", "2026-10-20"),
        ("2026-10-20", None),
    ]
    assert load_catalog_document(applied).version == "2026-10-19"


def test_refresh_does_not_mutate_its_inputs():
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0))
    contents = {"openai-api-pricing": _fixture("openai-api-pricing")}
    before = (copy.deepcopy(document), dict(contents))
    _plan(document, contents=contents)
    assert (document, contents) == before


def test_refresh_rejects_an_invalid_catalog():
    with pytest.raises(ValueError, match="currency must be USD"):
        _plan(dict(_document(), currency="EUR"))


def test_refresh_is_deterministic():
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0, input_per_mtok=1.0))
    assert _plan(document) == _plan(document)


def test_the_result_serializes_to_json_data():
    data = _plan(_with_gpt_x(**dict(GPT_X, input_per_mtok=-1.0))).as_json()
    assert data["observations"][0]["result"] == "escalated"
    assert data["escalations"][0] == {
        "result": "escalated",
        "source_id": "openai-api-pricing",
        "reason": "invalid_current_rate",
        "key": {"canonical_id": "example/gpt-x", "channel": "openai-api", "region": "global", "effective_from": "2026-01-01"},
        "model_key": "gpt-x",
        "region": "global",
        "dimension": "input_per_mtok",
        "current": "-1.0",
        "observed": "2",
        "detail": "the catalog rate is not a positive finite number",
    }


# --- Discovery evidence ---------------------------------------------------------


def test_discovery_evidence_lists_candidates_with_prior_values_and_reasons():
    source = _registry(_openai_source(low_risk=False)).get("openai-api-pricing")
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0))
    del document["models"][2]
    evidence = discovery_evidence(source, _fixture("openai-api-pricing"), document, observed_on=OBSERVED_ON)
    assert evidence["source_id"] == "openai-api-pricing"
    assert evidence["observed_on"] == "2026-10-20"
    assert evidence["fingerprint_matches"] is True
    assert evidence["applicable"] is True
    assert evidence["candidates"] == [
        {"model_key": "gpt-legacy", "region": "global", "canonical_id": None, "reason": "unknown_model", "prior": None, "candidate": {"input_per_mtok": "2.5", "cache_read_per_mtok": "1.25", "output_per_mtok": "10"}, "missing_dimensions": [], "invalid_dimensions": []},
        {"model_key": "gpt-x", "region": "global", "canonical_id": "example/gpt-x", "reason": "changed", "prior": {"input_per_mtok": "2", "cache_read_per_mtok": "0.2", "cache_write_5m_per_mtok": "2.5", "output_per_mtok": "9"}, "candidate": {"input_per_mtok": "2", "cache_read_per_mtok": "0.2", "cache_write_5m_per_mtok": "2.5", "output_per_mtok": "8"}, "missing_dimensions": [], "invalid_dimensions": []},
    ]
    # Raw page content is never emitted: no retention settings, no excerpt.
    assert "retention" not in evidence and "excerpt" not in evidence


def test_discovery_evidence_proposes_operations_only_for_changed_catalog_rows():
    source = _registry(_openai_source(low_risk=False)).get("openai-api-pricing")
    # gpt-x changed, and the catalog lacks gpt-legacy.
    document = _with_gpt_x(**dict(GPT_X, input_per_mtok=1.0))
    del document["models"][2]
    evidence = discovery_evidence(source, _fixture("openai-api-pricing"), document, observed_on=OBSERVED_ON)
    assert evidence["operations"] == [
        {
            "op": "replace_price",
            "key": {"canonical_id": "example/gpt-x", "channel": "openai-api", "region": "global", "effective_from": "2026-01-01"},
            "expect": _row(**dict(GPT_X, input_per_mtok=1.0)),
            "price": {
                "channel": "openai-api",
                "region": "global",
                "effective_from": "2026-10-20",
                "input_per_mtok": "2",
                "cache_read_per_mtok": "0.2",
                "cache_write_5m_per_mtok": "2.5",
                "output_per_mtok": "8",
                "source_url": OPENAI_URL,
            },
        }
    ]


def test_discovery_evidence_proposes_no_operation_for_a_row_starting_that_day():
    source = _registry(_openai_source(low_risk=False)).get("openai-api-pricing")
    document = _with_gpt_x(**dict(GPT_X, input_per_mtok=1.0))
    document["models"][0]["prices"][0]["effective_from"] = "2026-10-20"
    evidence = discovery_evidence(source, _fixture("openai-api-pricing"), document, observed_on=OBSERVED_ON)
    assert evidence["operations"] == []
    assert [c["reason"] for c in evidence["candidates"]] == ["changed"]


def _gateway_source(provider="openrouter", parser="openrouter-json"):
    return _registry(
        _source(
            "gateway-models",
            parser,
            f"{provider}-api" if provider == "openrouter" else "vercel-ai-gateway",
            provider,
            low_risk=False,
            fingerprint=None,
            url=f"https://example.com/{provider}/models",
        )
    ).get("gateway-models")


def _openrouter_page(*models):
    return json.dumps({"data": list(models)})


def _openrouter_model(model_id="example/model-new"):
    return {
        "id": model_id,
        "architecture": {"output_modalities": ["text"]},
        "pricing": {"prompt": "0.000001", "completion": "0.000004"},
    }


def test_gateway_discovery_proposes_a_complete_new_model():
    source = _gateway_source()
    evidence = discovery_evidence(
        source,
        _openrouter_page(_openrouter_model()),
        _document(),
        observed_on=OBSERVED_ON,
    )

    assert evidence["operations"] == [
        {
            "op": "add_model",
            "key": {"canonical_id": "example/model-new"},
            "expect": None,
            "model": {
                "canonical_id": "example/model-new",
                "publisher": "example",
                "aliases": [
                    {
                        "provider": "openrouter",
                        "alias": "example/model-new",
                        "channel": "openrouter-api",
                        "rules": {
                            "input_includes_cache_read": True,
                            "input_includes_cache_write": True,
                        },
                    }
                ],
                "prices": [
                    {
                        "channel": "openrouter-api",
                        "region": "global",
                        "effective_from": "2026-10-20",
                        "input_per_mtok": "1",
                        "output_per_mtok": "4",
                        "rules": {"varies_by_provider": True},
                        "source_url": "https://example.com/openrouter/models",
                    }
                ],
            },
        }
    ]


def test_gateway_discovery_adds_an_exact_alias_and_channel_price():
    source = _gateway_source()
    document = _document(
        _model(
            "example/model-new",
            "model-new",
            _row(input_per_mtok=2, output_per_mtok=8),
        )
    )

    operations = discovery_evidence(
        source,
        _openrouter_page(_openrouter_model()),
        document,
        observed_on=OBSERVED_ON,
    )["operations"]
    assert [operation["op"] for operation in operations] == ["add_alias", "add_price"]

    delta = {
        "schema_version": 2,
        "kind": "pricing_delta",
        "provenance": {"observations": ["gateway-models@2026-10-20"]},
        "operations": operations,
    }
    updated = apply_catalog_delta(document, delta)
    repeated = discovery_evidence(
        source,
        _openrouter_page(_openrouter_model()),
        updated,
        observed_on=OBSERVED_ON,
    )
    assert repeated["operations"] == []


def test_gateway_alias_reuses_existing_channel_rules():
    source = _gateway_source(provider="vercel", parser="vercel-gateway-json")
    document = _document(
        {
            "canonical_id": "example/model-new",
            "publisher": "example",
            "aliases": [
                {
                    "provider": "example",
                    "alias": "example/model-new",
                    "channel": "vercel-ai-gateway",
                    "rules": {"input_includes_cache_read": False},
                }
            ],
            "prices": [
                {
                    "channel": "vercel-ai-gateway",
                    "region": "global",
                    "effective_from": "2026-01-01",
                    "input_per_mtok": 1,
                    "output_per_mtok": 4,
                    "source_url": "https://example.com/models",
                }
            ],
        }
    )
    page = json.dumps(
        {
            "data": [
                {
                    "id": "example/model-new",
                    "owned_by": "example",
                    "type": "language",
                    "pricing": {"input": "0.000001", "output": "0.000004"},
                }
            ]
        }
    )

    [operation] = discovery_evidence(
        source, page, document, observed_on=OBSERVED_ON
    )["operations"]
    assert operation["op"] == "add_alias"
    assert operation["alias"]["rules"] == {"input_includes_cache_read": False}
    apply_catalog_delta(
        document,
        {
            "schema_version": 2,
            "kind": "pricing_delta",
            "provenance": {"observations": ["gateway@2026-10-20"]},
            "operations": [operation],
        },
    )


def test_gateway_discovery_replaces_a_changed_exact_route():
    source = _gateway_source(provider="vercel", parser="vercel-gateway-json")
    document = _document(
        _model(
            "example/model-new",
            "example/model-new",
            {
                "channel": "vercel-ai-gateway",
                "region": "global",
                "effective_from": "2026-01-01",
                "input_per_mtok": 0.5,
                "output_per_mtok": 2,
                "source_url": "https://example.com/models",
            },
            channel="vercel-ai-gateway",
            provider="vercel",
        )
    )
    page = json.dumps(
        {
            "data": [
                {
                    "id": "example/model-new",
                    "owned_by": "example",
                    "type": "language",
                    "pricing": {"input": "0.000001", "output": "0.000004"},
                }
            ]
        }
    )

    [operation] = discovery_evidence(
        source, page, document, observed_on=OBSERVED_ON
    )["operations"]
    assert operation["op"] == "replace_price"
    assert operation["price"]["input_per_mtok"] == "1"
    assert operation["price"]["output_per_mtok"] == "4"


def test_gateway_discovery_reports_exclusion_counts_without_raw_rows():
    source = _gateway_source()
    evidence = discovery_evidence(
        source,
        _openrouter_page(
            _openrouter_model("example/model-new:free"),
            _openrouter_model("example/model-new"),
        ),
        _document(),
        observed_on=OBSERVED_ON,
    )

    assert evidence["exclusions"] == {"free_route": 1}
    assert "model-new:free" not in json.dumps(evidence)


def test_gateway_discovery_maps_reviewed_publisher_prefixes():
    source = _gateway_source()
    document = _document(
        _model(
            "zai/glm-5.1",
            "glm-5.1",
            _row(input_per_mtok=1, output_per_mtok=4),
        )
    )
    page = _openrouter_page(_openrouter_model("z-ai/glm-5.1"))

    operations = discovery_evidence(
        source, page, document, observed_on=OBSERVED_ON
    )["operations"]
    assert [operation["op"] for operation in operations] == [
        "add_alias",
        "add_price",
    ]
    assert {operation["key"]["canonical_id"] for operation in operations} == {
        "zai/glm-5.1"
    }


def test_gateway_discovery_matches_a_same_publisher_short_alias():
    source = _gateway_source()
    document = _document(
        {
            "canonical_id": "openai/gpt-4o",
            "publisher": "openai",
            "aliases": [
                {
                    "provider": "openai",
                    "alias": "gpt-4o-2024-08-06",
                    "channel": "openai-api",
                }
            ],
            "prices": [
                _row(input_per_mtok=1, output_per_mtok=4)
            ],
        }
    )
    page = _openrouter_page(
        _openrouter_model("openai/gpt-4o-2024-08-06")
    )

    operations = discovery_evidence(
        source, page, document, observed_on=OBSERVED_ON
    )["operations"]
    assert [operation["op"] for operation in operations] == [
        "add_alias",
        "add_price",
    ]
    assert {operation["key"]["canonical_id"] for operation in operations} == {
        "openai/gpt-4o"
    }


def test_gateway_discovery_inherits_existing_channel_alias_rules():
    source = _gateway_source(provider="vercel", parser="vercel-gateway-json")
    document = _document(
        {
            "canonical_id": "deepseek/deepseek-v4.1-flash",
            "publisher": "deepseek",
            "aliases": [
                {
                    "provider": "deepseek",
                    "alias": "deepseek-v4-flash-vision-exp",
                    "channel": "deepseek-api",
                },
                {
                    "provider": "deepseek",
                    "alias": "deepseek/deepseek-v4.1-flash",
                    "channel": "vercel-ai-gateway",
                    "rules": {"input_includes_cache_read": True},
                }
            ],
            "prices": [
                {
                    "channel": "vercel-ai-gateway",
                    "region": "global",
                    "effective_from": "2026-01-01",
                    "input_per_mtok": 1,
                    "output_per_mtok": 4,
                    "source_url": "https://example.com/models",
                }
            ],
        }
    )

    operations = discovery_evidence(
        source,
        json.dumps(
            {
                "data": [
                    {
                        "id": "deepseek/deepseek-v4-flash-vision-exp",
                        "owned_by": "deepseek",
                        "type": "language",
                        "pricing": {"input": "0.000001", "output": "0.000004"},
                    }
                ]
            }
        ),
        document,
        observed_on=OBSERVED_ON,
    )["operations"]

    alias = next(
        operation["alias"]
        for operation in operations
        if operation["op"] == "add_alias"
    )
    assert alias["rules"] == {"input_includes_cache_read": True}


def test_gateway_discovery_rejects_a_new_alias_with_a_different_channel_price():
    source = _gateway_source(provider="vercel", parser="vercel-gateway-json")
    document = _document(
        {
            "canonical_id": "deepseek/deepseek-v4.1-flash",
            "publisher": "deepseek",
            "aliases": [
                {
                    "provider": "deepseek",
                    "alias": "deepseek-v4-flash-vision-exp",
                    "channel": "deepseek-api",
                },
                {
                    "provider": "deepseek",
                    "alias": "deepseek/deepseek-v4.1-flash",
                    "channel": "vercel-ai-gateway",
                },
            ],
            "prices": [
                {
                    "channel": "vercel-ai-gateway",
                    "region": "global",
                    "effective_from": "2026-01-01",
                    "input_per_mtok": 1,
                    "output_per_mtok": 4,
                    "source_url": "https://example.com/models",
                }
            ],
        }
    )
    page = json.dumps(
        {
            "data": [
                {
                    "id": "deepseek/deepseek-v4-flash-vision-exp",
                    "owned_by": "deepseek",
                    "type": "language",
                    "pricing": {"input": "0.000002", "output": "0.000004"},
                }
            ]
        }
    )

    evidence = discovery_evidence(
        source, page, document, observed_on=OBSERVED_ON
    )

    assert evidence["operations"] == []
    assert evidence["identity_exclusions"] == [
        {
            "model_key": "deepseek/deepseek-v4-flash-vision-exp",
            "reason": "channel_price_conflict",
            "candidates": ["deepseek/deepseek-v4.1-flash"],
        }
    ]


def test_gateway_discovery_allows_a_direct_identity_price_change():
    source = _gateway_source(provider="vercel", parser="vercel-gateway-json")
    document = _document(
        _model(
            "example/model-new",
            "example/model-new",
            {
                "channel": "vercel-ai-gateway",
                "region": "global",
                "effective_from": "2026-01-01",
                "input_per_mtok": 0.5,
                "output_per_mtok": 2,
                "source_url": "https://example.com/models",
            },
            channel="vercel-ai-gateway",
            provider="example",
        )
    )
    page = json.dumps(
        {
            "data": [
                {
                    "id": "example/model-new",
                    "owned_by": "example",
                    "type": "language",
                    "pricing": {"input": "0.000001", "output": "0.000004"},
                }
            ]
        }
    )

    operations = discovery_evidence(
        source, page, document, observed_on=OBSERVED_ON
    )["operations"]

    assert sorted(operation["op"] for operation in operations) == [
        "add_alias",
        "replace_price",
    ]


def test_gateway_discovery_reports_near_identity_instead_of_adding_a_model():
    source = _gateway_source()
    document = _document(
        _model(
            "mistral/mistral-medium-3.5",
            "mistral-medium-3.5",
            _row(input_per_mtok=1, output_per_mtok=4),
        )
    )
    page = _openrouter_page(
        _openrouter_model("mistralai/mistral-medium-3-5")
    )

    evidence = discovery_evidence(
        source, page, document, observed_on=OBSERVED_ON
    )
    assert evidence["operations"] == []
    assert evidence["identity_exclusions"] == [
        {
            "model_key": "mistralai/mistral-medium-3-5",
            "reason": "ambiguous_identity",
            "candidates": ["mistral/mistral-medium-3.5"],
        }
    ]


def test_gateway_discovery_reports_colliding_operations():
    source = _gateway_source()
    document = _document(
        {
            "canonical_id": "example/model",
            "publisher": "example",
            "aliases": [
                {
                    "provider": "example",
                    "alias": "one",
                    "channel": "example-api",
                },
                {
                    "provider": "example",
                    "alias": "two",
                    "channel": "example-api",
                },
            ],
            "prices": [_row(input_per_mtok=1, output_per_mtok=4)],
        }
    )
    page = _openrouter_page(
        _openrouter_model("example/one"),
        _openrouter_model("example/two"),
    )

    evidence = discovery_evidence(
        source, page, document, observed_on=OBSERVED_ON
    )
    assert sorted(operation["op"] for operation in evidence["operations"]) == [
        "add_alias",
        "add_alias",
        "add_price",
    ]
    assert evidence["operation_collisions"] == [
        {
            "canonical_id": "example/model",
            "operation": "add_price",
            "reason": "identical_operation_deduplicated",
            "candidate_count": 2,
        }
    ]


def test_gateway_discovery_blocks_conflicting_operations():
    source = _gateway_source()
    document = _document(
        {
            "canonical_id": "example/model",
            "publisher": "example",
            "aliases": [
                {"provider": "example", "alias": "one", "channel": "example-api"},
                {"provider": "example", "alias": "two", "channel": "example-api"},
            ],
            "prices": [_row(input_per_mtok=1, output_per_mtok=4)],
        }
    )
    second = _openrouter_model("example/two")
    second["pricing"]["prompt"] = "0.000002"

    evidence = discovery_evidence(
        source,
        _openrouter_page(_openrouter_model("example/one"), second),
        document,
        observed_on=OBSERVED_ON,
    )

    assert evidence["operations"] == []
    assert evidence["operation_collisions"] == [
        {
            "canonical_id": "example/model",
            "operation": "add_price",
            "reason": "conflicting_operation",
            "candidate_count": 2,
        }
    ]


def test_gateway_discovery_blocks_operations_on_a_fingerprint_change():
    source = _registry(
        _source(
            "gateway-models",
            "openrouter-json",
            "openrouter-api",
            "openrouter",
            low_risk=False,
            fingerprint="sha256:" + "0" * 64,
            url="https://example.com/openrouter/models",
        )
    ).get("gateway-models")

    evidence = discovery_evidence(
        source,
        _openrouter_page(_openrouter_model()),
        _document(),
        observed_on=OBSERVED_ON,
    )
    assert evidence["fingerprint_matches"] is False
    assert evidence["operations"] == []
    assert evidence["operations_blocked_reason"] == "fingerprint_mismatch"


def test_gateway_discovery_accepts_a_zero_rate_on_one_paid_dimension():
    source = _gateway_source()
    page = _openrouter_page(
        {
            "id": "scaledown/input-only",
            "architecture": {"output_modalities": ["text"]},
            "pricing": {"prompt": "0.00000004", "completion": "0"},
        }
    )
    initial = discovery_evidence(
        source, page, _document(), observed_on=OBSERVED_ON
    )
    delta = {
        "schema_version": 2,
        "kind": "pricing_delta",
        "provenance": {"observations": ["gateway-models@2026-10-20"]},
        "operations": initial["operations"],
    }
    updated = apply_catalog_delta(_document(), delta)

    repeated = discovery_evidence(
        source, page, updated, observed_on=OBSERVED_ON
    )
    assert repeated["operations"] == []
    assert repeated["candidates"] == []


def test_discovery_evidence_for_gpu_licensing_carries_no_token_candidates():
    source = _discovery_source("gpu-license-prose")
    evidence = discovery_evidence(source, _generic("gpu-license-prose"), _document(), observed_on=OBSERVED_ON)
    assert evidence["applicable"] is False
    assert evidence["candidates"] == []
    assert evidence["evidence"]["candidates"] == {"license_per_gpu_year_usd": "1000", "gpu_hour_usd": "0.50"}


# --- Several aliases of one model on the page ----------------------------------------


def _gpt_x_with_aliases(*aliases, **rates):
    """gpt-x answering to several page names; gpt-x-mini is not its own model."""
    document = _with_gpt_x(**rates)
    del document["models"][1]
    document["models"][0]["aliases"] = [
        {"provider": "openai", "alias": alias, "channel": "openai-api"} for alias in aliases
    ]
    return document


def test_aliases_whose_page_prices_disagree_escalate_as_ambiguous():
    # The page prices gpt-x and gpt-x-mini differently; both name one model.
    document = _gpt_x_with_aliases("gpt-x", "gpt-x-mini", **dict(GPT_X, output_per_mtok=9.0))
    result = _plan(document)
    assert result.delta["operations"] == []
    assert _reasons(result) == [("openai-api-pricing", "ambiguous_parse", "example/gpt-x")]
    assert "gpt-x-mini" in result.escalations[0].detail


_GPT_X_LATEST = _fixture("openai-api-pricing").replace(
    "[1,[[0,&quot;gpt-x-mini&quot;]",
    "[1,[[0,&quot;gpt-x-latest&quot;],[0,2],[0,0.2],[0,2.5],[0,8]]][1,[[0,&quot;gpt-x-mini&quot;]",
)


def test_aliases_whose_page_prices_agree_are_one_match():
    document = _gpt_x_with_aliases("gpt-x", "gpt-x-latest", **dict(GPT_X, output_per_mtok=9.0))
    document["models"].insert(
        1, _model("example/gpt-x-mini", "gpt-x-mini", _row(input_per_mtok=0.4, cache_read_per_mtok=0.04, output_per_mtok=1.6))
    )
    result = _plan(document, contents={"openai-api-pricing": _GPT_X_LATEST})
    assert result.escalations == ()
    [operation] = result.delta["operations"]
    assert (operation["key"]["canonical_id"], operation["price"]["output_per_mtok"]) == ("example/gpt-x", "8")


def test_discovery_proposes_no_operation_when_aliases_disagree():
    source = _registry(_openai_source(low_risk=False)).get("openai-api-pricing")
    document = _gpt_x_with_aliases("gpt-x", "gpt-x-mini", **dict(GPT_X, output_per_mtok=9.0))
    evidence = discovery_evidence(source, _fixture("openai-api-pricing"), document, observed_on=OBSERVED_ON)
    assert evidence["operations"] == []
    assert "gpt-x-mini" not in [c["model_key"] for c in evidence["candidates"] if c["reason"] == "unknown_model"]


# --- Rows with a scheduled end -------------------------------------------------------


def _scheduled(**rates):
    """gpt-x's current row ends on 2027-01-01, where a scheduled row begins."""
    document = _with_gpt_x(**rates)
    document["models"][0]["prices"][0]["effective_to"] = "2027-01-01"
    document["models"][0]["prices"].append(
        _row(**dict(GPT_X, output_per_mtok=12.0), effective_from="2027-01-01")
    )
    return document


def test_a_change_to_a_row_with_a_scheduled_end_escalates():
    result = _plan(_scheduled(**dict(GPT_X, output_per_mtok=9.0)))
    assert result.delta["operations"] == []
    [escalation] = result.escalations
    assert (escalation.reason, escalation.key.canonical_id) == ("scheduled_window", "example/gpt-x")
    assert escalation.detail == "the current row already ends 2027-01-01"


def test_an_unchanged_row_with_a_scheduled_end_is_fine():
    result = _plan(_scheduled(**GPT_X))
    assert result.escalations == ()
    assert result.observations[0].result == "unchanged"


def test_no_low_risk_operation_conflicts_when_applied():
    document = _scheduled(**dict(GPT_X, output_per_mtok=9.0))
    document["models"][1]["prices"][0]["output_per_mtok"] = 2.0
    result = _plan(document)
    applied = apply_catalog_delta(document, result.delta)
    assert [op["key"]["canonical_id"] for op in result.delta["operations"]] == ["example/gpt-x-mini"]
    assert load_catalog_document(applied).version == "2026-10-19"


def test_discovery_proposes_no_operation_for_a_row_with_a_scheduled_end():
    source = _registry(_openai_source(low_risk=False)).get("openai-api-pricing")
    evidence = discovery_evidence(source, _fixture("openai-api-pricing"), _scheduled(**dict(GPT_X, output_per_mtok=9.0)), observed_on=OBSERVED_ON)
    assert [(c["model_key"], c["reason"]) for c in evidence["candidates"]] == [("gpt-x", "changed")]
    assert evidence["operations"] == []


# --- Effective dates in other forms --------------------------------------------------


@pytest.mark.parametrize(
    "effective_from",
    [
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 1),
        "2026-01-01T00:00:00Z",
        "2026-01-01T09:30:00+05:30",
    ],
    ids=["aware-datetime", "naive-datetime", "z-suffix", "offset"],
)
def test_rows_dated_as_datetimes_or_offsets_plan_normally(effective_from):
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0))
    document["models"][0]["prices"][0]["effective_from"] = effective_from
    result = _plan(document)
    assert result.escalations == ()
    [operation] = result.delta["operations"]
    assert operation["key"]["effective_from"] == "2026-01-01" or operation["key"]["effective_from"].startswith("2026-01-01T")
    assert operation["price"]["effective_from"] == "2026-10-20"
    apply_catalog_delta(document, result.delta)


def test_an_offset_start_is_compared_on_its_utc_date():
    # 01:00 at +02:00 on 20 October is 23:00 UTC on 19 October, so the row
    # started before the observation date and may be replaced.
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0))
    document["models"][0]["prices"][0]["effective_from"] = "2026-10-20T01:00:00+02:00"
    [operation] = _plan(document).delta["operations"]
    assert operation["price"]["effective_from"] == "2026-10-20"


def test_a_row_starting_at_utc_midnight_of_the_observation_date_escalates():
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0))
    document["models"][0]["prices"][0]["effective_from"] = "2026-10-19T22:00:00-02:00"
    result = _plan(document)
    assert result.delta["operations"] == []
    assert [e.reason for e in result.escalations] == ["not_after_current"]


def test_a_row_starting_after_utc_midnight_is_not_yet_in_effect():
    # 23:00 at -02:00 on 19 October is 01:00 UTC on 20 October, after the
    # observation instant, so gpt-x has no current row to compare or replace.
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0))
    document["models"][0]["prices"][0]["effective_from"] = "2026-10-19T23:00:00-02:00"
    result = _plan(document)
    assert result.delta["operations"] == []
    assert result.observations[0].rows_compared == 2


# --- Effective windows at the observation instant ------------------------------------
# A run observed on day D sees the catalog as of 00:00 UTC on D.


def test_a_row_ending_at_noon_is_current_at_the_observation_instant():
    document = _with_gpt_x(**GPT_X)
    document["models"][0]["prices"][0]["effective_to"] = "2026-10-20T14:00:00+02:00"
    document["models"][0]["prices"].append(
        _row(**dict(GPT_X, output_per_mtok=12.0), effective_from="2026-10-20T12:00:00Z")
    )
    result = _plan(document)
    # The page matches the row in effect at midnight, not the noon successor.
    assert result.escalations == ()
    assert result.observations[0].result == "unchanged"
    assert result.observations[0].rows_compared == 3


def test_date_only_windows_keep_their_meaning():
    document = _with_gpt_x(**GPT_X)
    document["models"][0]["prices"][0]["effective_to"] = "2026-10-20"
    document["models"][0]["prices"].append(
        _row(**dict(GPT_X, output_per_mtok=9.0), effective_from="2026-10-20")
    )
    result = _plan(document)
    # The 20 October row is current, and it started on the observation date.
    assert [e.reason for e in result.escalations] == ["not_after_current"]


def _dated_alias_document(alias_from, alias_to):
    document = _document()
    entry = document["models"][0]
    entry["canonical_id"] = "example/gpt-x-model"
    entry["aliases"] = [
        {
            "provider": "openai",
            "alias": "gpt-x",
            "channel": "openai-api",
            "effective_from": alias_from,
            "effective_to": alias_to,
            "source_url": "https://example.com/aliases",
        }
    ]
    return document


def test_an_alias_ending_later_on_the_observation_date_still_matches():
    result = _plan(_dated_alias_document("2026-01-01", "2026-10-20T06:00:00-04:00"))
    assert result.escalations == ()


def test_an_alias_starting_later_on_the_observation_date_does_not_match_yet():
    result = _plan(_dated_alias_document("2026-10-20T12:00:00Z", None))
    assert _reasons(result) == [
        ("openai-api-pricing", "model_not_listed", "example/gpt-x-model"),
        ("openai-api-pricing", "unknown_model", "gpt-x"),
    ]


def _tail_alias_document(alias, alias_from, alias_to):
    """gpt-x whose only openai-api alias is its own canonical tail, dated."""
    document = _document()
    window = {"effective_from": alias_from, "source_url": "https://example.com/aliases"}
    if alias_to:
        window["effective_to"] = alias_to
    document["models"][0]["aliases"] = [dict({"provider": "openai", "alias": alias, "channel": "openai-api"}, **window)]
    return document


@pytest.mark.parametrize(
    ("alias", "alias_from", "alias_to"),
    [("gpt-x", "2026-11-01", None), ("gpt-x", "2026-01-01", "2026-10-01"), ("GPT-X", "2026-01-01", "2026-10-01")],
    ids=["not-yet-effective", "expired", "expired-other-case"],
)
def test_an_inactive_alias_is_not_revived_by_the_canonical_tail(alias, alias_from, alias_to):
    result = _plan(_tail_alias_document(alias, alias_from, alias_to))
    assert _reasons(result) == [
        ("openai-api-pricing", "model_not_listed", "example/gpt-x"),
        ("openai-api-pricing", "unknown_model", "gpt-x"),
    ]


def test_the_canonical_tail_still_matches_without_an_alias_on_the_channel():
    document = _document()
    document["models"][0]["aliases"] = [
        {"provider": "openai", "alias": "gpt-x", "channel": "vercel-ai-gateway"}
    ]
    result = _plan(document)
    assert result.escalations == ()
    assert result.observations[0].rows_compared == 3


def test_current_rows_use_the_observation_instant():
    from metergraph_core.refresh import current_rows

    document = _with_gpt_x(**GPT_X)
    document["models"][0]["prices"][0]["effective_to"] = "2026-10-20T12:00:00Z"
    document["models"][0]["prices"].append(_row(**dict(GPT_X, output_per_mtok=12.0), effective_from="2026-10-20T12:00:00Z"))
    rows = current_rows(document, "openai-api", OBSERVED_ON)
    assert [(r["canonical_id"], r["output_per_mtok"]) for r in rows] == [
        ("example/gpt-x", 8.0),
        ("example/gpt-x-mini", 1.6),
        ("example/gpt-legacy", 10.0),
    ]
    assert rows[0]["effective_to"] == "2026-10-20T12:00:00Z"


def test_a_result_with_datetime_rows_serializes_to_json():
    import json

    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0))
    document["models"][0]["prices"][0]["effective_from"] = datetime(2026, 1, 1, tzinfo=timezone.utc)
    data = _plan(document).as_json()
    json.dumps(data)
    apply_catalog_delta(document, data["delta"])


def test_discovery_evidence_with_datetime_rows_serializes_to_json():
    import json

    source = _registry(_openai_source(low_risk=False)).get("openai-api-pricing")
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0))
    document["models"][0]["prices"][0]["effective_from"] = datetime(2026, 1, 1, tzinfo=timezone.utc)
    evidence = discovery_evidence(source, _fixture("openai-api-pricing"), document, observed_on=OBSERVED_ON)
    json.dumps(evidence)
    assert evidence["operations"][0]["expect"]["effective_from"] == "2026-01-01T00:00:00+00:00"


# --- Raw page content is never emitted ----------------------------------------------

# Generic synthetic pages for the discovery parsers, standing for no
# provider's layout: a token-price table beside a non-token table, and GPU
# or license pricing stated in prose.
TOKEN_TABLE_PAGE = (
    "<table><tr><th>Model</th><th>Input (per 1M tokens)</th><th>Cached input (per 1M tokens)</th>"
    "<th>Output (per 1M tokens)</th></tr>"
    "<tr><td>Example Model Large</td><td>$2.00</td><td>$0.20</td><td>$8.00</td></tr>"
    "<tr><td>Example Model Small</td><td>$0.40</td><td>Free</td><td>$1.60</td></tr></table>"
    "<table><tr><th>Accelerator</th><th>Price per hour</th></tr>"
    "<tr><td>Example Accelerator</td><td>$2.90</td></tr></table>"
)
GPU_LICENSE_PAGE = (
    "<p>Licenses start at $1000 per GPU per year, or about $0.50 per GPU per hour in the cloud.</p>"
    "<p>Developers can prototype at no cost.</p>"
)
# Each generic synthetic page with the discovery parser and dimensions
# of a source that would be read with it.
GENERIC_SOURCES = {
    "token-table": ("html-tables", ["input_per_mtok", "output_per_mtok"]),
    "gpu-license-prose": ("gpu-license-prose", ["gpu_hour_usd", "license_per_gpu_year_usd"]),
}


def _generic(name):
    return {"token-table": TOKEN_TABLE_PAGE, "gpu-license-prose": GPU_LICENSE_PAGE}[name]


def _discovery_source(page):
    parser, dimensions = GENERIC_SOURCES[page]
    entry = _source("example-discovery-pricing", parser, "example-channel", "example", low_risk=False, fingerprint=None, dimensions=dimensions)
    entry.update(catalog_channel=False)
    return _registry(entry).get("example-discovery-pricing")


def _verbatim_cells(content):
    """Page text a forbidden report must not repeat: each table cell and
    paragraph that reads as words rather than a bare amount."""
    import re

    cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>|<p>(.*?)</p>", content, flags=re.S)
    texts = [a or b for a, b in cells]
    return [t for t in texts if re.search(r"[A-Za-z]{2}", t) and (" " in t or re.search(r"[A-Z]", t))]


@pytest.mark.parametrize("page", list(GENERIC_SOURCES))
def test_raw_page_content_never_appears_in_discovery_evidence(page):
    import json

    content = _generic(page)
    evidence = discovery_evidence(_discovery_source(page), content, _document(), observed_on=OBSERVED_ON)
    text = json.dumps(evidence)
    cells = _verbatim_cells(content)
    assert cells
    assert [cell for cell in cells if cell in text] == []
    assert evidence["evidence"]["redacted"] is True
    assert evidence["fingerprint"] == structural_fingerprint(GENERIC_SOURCES[page][0], content)


def test_redacted_table_evidence_keeps_normalized_labels_and_amounts():
    evidence = discovery_evidence(_discovery_source("token-table"), _generic("token-table"), _document(), observed_on=OBSERVED_ON)
    assert evidence["evidence"] == {
        "kind": "tables",
        "redacted": True,
        "tables": [
            {
                "columns": 4,
                "rows": [
                    {"label": "example-model-large", "amounts": ["2.00", "0.20", "8.00"]},
                    {"label": "example-model-small", "amounts": ["0.40", "1.60"]},
                ],
            },
            {"columns": 2, "rows": [{"label": "example-accelerator", "amounts": ["2.90"]}]},
        ],
    }


def test_redacted_amounts_are_plain_decimals_even_when_tiny():
    page = "<table><tr><th>Model</th><th>Per token</th></tr><tr><td>Tiny</td><td>$0.0000001</td></tr></table>"
    evidence = discovery_evidence(_discovery_source("token-table"), page, _document(), observed_on=OBSERVED_ON)
    assert evidence["evidence"]["tables"] == [{"columns": 2, "rows": [{"label": "tiny", "amounts": ["0.0000001"]}]}]


def test_redacted_license_evidence_keeps_candidates_and_drops_statements():
    evidence = discovery_evidence(_discovery_source("gpu-license-prose"), _generic("gpu-license-prose"), _document(), observed_on=OBSERVED_ON)
    assert evidence["evidence"] == {
        "kind": "gpu_or_license_statements",
        "redacted": True,
        "statement_count": 1,
        "candidates": {"license_per_gpu_year_usd": "1000", "gpu_hour_usd": "0.50"},
    }


# --- Dimensions that disappear and regional ambiguity -------------------------

# The gpt-x row without its cache-write column. Dropping the column changes the
# page's row layout, so the registry records this content's fingerprint.
_NO_CACHE_WRITE = _fixture("openai-api-pricing").replace(
    "[0,2],[0,0.2],[0,2.5],[0,8]", "[0,2],[0,0.2],[0,9]"
)


def _no_cache_write_registry(**overrides):
    return _registry(_openai_source(fingerprint=structural_fingerprint("openai", _NO_CACHE_WRITE), **overrides))


def test_a_represented_dimension_missing_from_the_page_escalates_a_changing_row():
    result = _plan(_document(), _no_cache_write_registry(), {"openai-api-pricing": _NO_CACHE_WRITE})
    assert result.delta["operations"] == []
    assert [(e.reason, e.dimension, e.current) for e in result.escalations] == [
        ("missing_rate", "cache_write_5m_per_mtok", "2.5")
    ]


def test_a_represented_dimension_missing_from_the_page_escalates_an_unchanged_row():
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0))
    result = _plan(document, _no_cache_write_registry(), {"openai-api-pricing": _NO_CACHE_WRITE})
    assert result.delta["operations"] == []
    assert [(e.reason, e.dimension) for e in result.escalations] == [
        ("missing_rate", "cache_write_5m_per_mtok")
    ]


def test_a_dimension_the_source_does_not_represent_is_not_required():
    registry = _no_cache_write_registry(
        dimensions=["input_per_mtok", "cache_read_per_mtok", "output_per_mtok"]
    )
    document = _document()
    del document["models"][0]["prices"][0]["cache_write_5m_per_mtok"]
    result = _plan(document, registry, {"openai-api-pricing": _NO_CACHE_WRITE})
    [operation] = result.delta["operations"]
    assert operation["price"]["output_per_mtok"] == "9"
    assert result.escalations == ()


def test_discovery_proposes_no_operation_that_would_copy_a_missing_rate_forward():
    source = _no_cache_write_registry(low_risk=False).get("openai-api-pricing")
    evidence = discovery_evidence(source, _NO_CACHE_WRITE, _document(), observed_on=OBSERVED_ON)
    assert [(c["model_key"], c["reason"]) for c in evidence["candidates"]] == [("gpt-x", "changed")]
    assert evidence["operations"] == []


def test_discovery_reports_a_cache_rate_that_disappears_with_nothing_else_changed():
    page = _fixture("openai-api-pricing").replace("[0,2],[0,0.2],[0,2.5],[0,8]", "[0,2],[0,0.2],[0,8]")
    source = _registry(_openai_source(low_risk=False)).get("openai-api-pricing")
    evidence = discovery_evidence(source, page, _document(), observed_on=OBSERVED_ON)
    assert evidence["candidates"] == [
        {
            "model_key": "gpt-x",
            "region": "global",
            "canonical_id": "example/gpt-x",
            "reason": "missing_rate",
            "prior": {"input_per_mtok": "2", "cache_read_per_mtok": "0.2", "cache_write_5m_per_mtok": "2.5", "output_per_mtok": "8"},
            "candidate": {"input_per_mtok": "2", "cache_read_per_mtok": "0.2", "output_per_mtok": "8"},
            "missing_dimensions": ["cache_write_5m_per_mtok"],
            "invalid_dimensions": [],
        }
    ]
    assert evidence["operations"] == []


def test_discovery_reports_an_output_rate_that_disappears_with_nothing_else_changed():
    entry = _source("anthropic-api-pricing", "anthropic", "anthropic-api", "anthropic", low_risk=False)
    source = _registry(entry).get("anthropic-api-pricing")
    document = _document(
        _model("example/claude-example-5", "claude-example-5", dict(_row(input_per_mtok=5.0, output_per_mtok=25.0, cache_write_5m_per_mtok=6.25, cache_read_per_mtok=0.5), channel="anthropic-api"), channel="anthropic-api", provider="anthropic"),
        _model("example/claude-example-5-5", "claude-example-5-5", dict(_row(input_per_mtok=4.0, output_per_mtok=20.0, cache_write_5m_per_mtok=5.0, cache_read_per_mtok=0.4), channel="anthropic-api"), channel="anthropic-api", provider="anthropic"),
    )
    page = _fixture("anthropic-api-pricing").replace("<td>$25 / MTok</td>", "<td>Contact us</td>")
    evidence = discovery_evidence(source, page, document, observed_on=OBSERVED_ON)
    assert [(c["model_key"], c["reason"], c["missing_dimensions"]) for c in evidence["candidates"]] == [
        ("claude-example-5", "missing_rate", ["output_per_mtok"])
    ]
    assert evidence["operations"] == []


def test_discovery_lists_a_missing_rate_alongside_a_changed_one():
    source = _no_cache_write_registry(low_risk=False).get("openai-api-pricing")
    evidence = discovery_evidence(source, _NO_CACHE_WRITE, _document(), observed_on=OBSERVED_ON)
    [candidate] = evidence["candidates"]
    assert (candidate["reason"], candidate["missing_dimensions"]) == ("changed", ["cache_write_5m_per_mtok"])


@pytest.mark.parametrize(
    ("current", "shown"),
    [("not-a-rate", "not-a-rate"), (float("nan"), "nan"), ("Infinity", "Infinity"), (0.0, "0.0"), (-2.0, "-2.0")],
    ids=["unparsable", "nan", "infinite-text", "zero", "negative"],
)
def test_discovery_reports_an_invalid_current_rate_and_proposes_nothing(current, shown):
    # The loader accepts each of these; an unparsable rate loads as absent.
    document = _with_gpt_x(**dict(GPT_X, input_per_mtok=current))
    load_catalog_document(document)
    source = _registry(_openai_source(low_risk=False)).get("openai-api-pricing")
    evidence = discovery_evidence(source, _fixture("openai-api-pricing"), document, observed_on=OBSERVED_ON)
    [candidate] = evidence["candidates"]
    assert (candidate["model_key"], candidate["reason"]) == ("gpt-x", "invalid_current_rate")
    assert candidate["invalid_dimensions"] == ["input_per_mtok"]
    assert candidate["prior"]["input_per_mtok"] == shown
    assert candidate["candidate"]["input_per_mtok"] == "2"
    assert evidence["operations"] == []


@pytest.mark.parametrize("carried", ["-1", "n/a", float("nan"), "Infinity"], ids=["negative", "unparsable", "nan", "infinite"])
def test_discovery_proposes_nothing_when_a_carried_rate_is_invalid(carried):
    # batch_input is outside the source's dimensions, so a replacement would
    # copy it forward: invalid, it would break the delta or vanish from it.
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0, batch_input_per_mtok=carried))
    load_catalog_document(document)
    source = _registry(_openai_source(low_risk=False)).get("openai-api-pricing")
    evidence = discovery_evidence(source, _fixture("openai-api-pricing"), document, observed_on=OBSERVED_ON)
    [candidate] = evidence["candidates"]
    assert (candidate["reason"], candidate["invalid_dimensions"]) == ("invalid_current_rate", ["batch_input_per_mtok"])
    assert evidence["operations"] == []


def test_discovery_carries_a_zero_rate_outside_its_dimensions():
    document = _with_gpt_x(**dict(GPT_X, output_per_mtok=9.0, batch_input_per_mtok=0.0))
    source = _registry(_openai_source(low_risk=False)).get("openai-api-pricing")
    evidence = discovery_evidence(source, _fixture("openai-api-pricing"), document, observed_on=OBSERVED_ON)
    [operation] = evidence["operations"]
    assert operation["price"]["batch_input_per_mtok"] == "0"


@pytest.mark.parametrize("carried", ["-1", "n/a", float("nan")], ids=["negative", "unparsable", "nan"])
def test_a_refresh_never_copies_an_invalid_carried_rate(carried):
    result = _plan(_with_gpt_x(**dict(GPT_X, output_per_mtok=9.0, batch_input_per_mtok=carried)))
    assert result.delta["operations"] == []
    assert [(e.reason, e.dimension) for e in result.escalations] == [("unrepresented_dimension", "batch_input_per_mtok")]


def test_an_unparsable_current_rate_escalates_as_invalid_in_a_refresh():
    result = _plan(_with_gpt_x(**dict(GPT_X, input_per_mtok="not-a-rate")))
    assert result.delta["operations"] == []
    [escalation] = result.escalations
    assert (escalation.reason, escalation.dimension, escalation.current, escalation.observed) == (
        "invalid_current_rate", "input_per_mtok", "not-a-rate", "2"
    )


def test_an_unparsable_current_rate_the_page_drops_escalates_without_crashing():
    document = _with_gpt_x(**dict(GPT_X, cache_write_5m_per_mtok="n/a"))
    result = _plan(document, _no_cache_write_registry(), {"openai-api-pricing": _NO_CACHE_WRITE})
    assert result.delta["operations"] == []
    assert [(e.reason, e.dimension, e.current) for e in result.escalations] == [
        ("missing_rate", "cache_write_5m_per_mtok", "n/a")
    ]


def test_discovery_proposes_no_operation_for_an_ambiguous_row():
    source = _registry(dict(_xai_source(), low_risk={"eligible": False, "dimensions": []})).get("xai-api-pricing")
    document = _xai_document(global_output=6.0)
    document["models"][0]["prices"][1]["input_per_mtok"] = 2.0
    evidence = discovery_evidence(source, _XAI_TWO_US_PRICES, document, observed_on=OBSERVED_ON)
    assert evidence["ambiguities"] == ["grok-x@us"]
    assert evidence["operations"] == []


def _xai_source():
    return _source(
        "xai-api-pricing", "xai", "xai-api", "xai",
        dimensions=["input_per_mtok", "output_per_mtok", "cache_read_per_mtok"],
    )


def _xai_document(global_output):
    rows = [
        dict(_row(input_per_mtok=2.0, cache_read_per_mtok=0.5, output_per_mtok=global_output), channel="xai-api"),
        dict(_row(input_per_mtok=2.2, cache_read_per_mtok=0.55, output_per_mtok=6.6), channel="xai-api", region="us"),
    ]
    return _document(_model("example/grok-x", "grok-x", *rows, channel="xai-api", provider="x-ai"))


_SECOND_US_ENTRY = (
    "{&quot;name&quot;:&quot;grok-x&quot;,&quot;version&quot;:&quot;1.0&quot;,"
    "&quot;promptTextTokenPrice&quot;:&quot;23000&quot;,&quot;cachedPromptTokenPrice&quot;:&quot;5500&quot;,"
    "&quot;completionTextTokenPrice&quot;:&quot;66000&quot;,&quot;cluster&quot;:&quot;us-central-1&quot;}"
)
# A second US-cluster entry for grok-x with a different input price.
_XAI_TWO_US_PRICES = _fixture("xai-api-pricing").replace(
    "{&quot;name&quot;:&quot;grok-imagine-image&quot;",
    _SECOND_US_ENTRY + "{&quot;name&quot;:&quot;grok-imagine-image&quot;",
)


def _plan_xai(global_output):
    return plan_low_risk_refresh(
        _xai_document(global_output),
        _registry(_xai_source()),
        {"xai-api-pricing": _XAI_TWO_US_PRICES},
        observed_on=OBSERVED_ON,
    )


def test_a_regional_ambiguity_escalates_only_that_regions_row():
    result = _plan_xai(global_output=6.5)
    assert [(op["key"]["region"], op["price"]["output_per_mtok"]) for op in result.delta["operations"]] == [
        ("global", "6")
    ]
    assert [(e.reason, e.region) for e in result.escalations] == [("ambiguous_parse", "us")]


def test_a_regional_ambiguity_escalates_even_an_unchanged_row():
    result = _plan_xai(global_output=6.0)
    assert result.delta["operations"] == []
    assert [(e.reason, e.region, e.key.canonical_id) for e in result.escalations] == [
        ("ambiguous_parse", "us", "example/grok-x")
    ]


def test_gateway_discovery_maps_meta_llama_and_bytedance_seed_prefixes():
    source = _gateway_source()
    document = _document(
        _model(
            "meta/llama-x",
            "meta/llama-x",
            _row(channel="vercel-ai-gateway", input_per_mtok=1, output_per_mtok=4),
            channel="vercel-ai-gateway",
            provider="vercel",
        )
    )
    page = _openrouter_page(
        _openrouter_model("meta-llama/llama-x"),
        _openrouter_model("bytedance-seed/seed-x"),
    )

    operations = discovery_evidence(
        source, page, document, observed_on=OBSERVED_ON
    )["operations"]
    assert sorted(
        (operation["op"], operation["key"]["canonical_id"])
        for operation in operations
    ) == [
        ("add_alias", "meta/llama-x"),
        ("add_model", "bytedance/seed-x"),
        ("add_price", "meta/llama-x"),
    ]
    [added] = [o for o in operations if o["op"] == "add_model"]
    assert added["model"]["publisher"] == "bytedance"
    assert added["model"]["aliases"][0]["alias"] == "bytedance-seed/seed-x"


def _moved_spelling_document(gateway_input=1):
    """meta/llama-x took over the gateway spelling a duplicate entry held
    until 2026-10-01."""
    moved = {"effective_from": "2026-10-01", "source_url": "https://example.com/review"}
    gateway_row = _row(channel="openrouter-api", input_per_mtok=gateway_input, output_per_mtok=4)
    return _document(
        {
            "canonical_id": "meta/llama-x",
            "publisher": "meta",
            "aliases": [
                {"provider": "vercel", "alias": "meta/llama-x", "channel": "vercel-ai-gateway"},
                {"provider": "openrouter", "alias": "meta-llama/llama-x-instruct", "channel": "openrouter-api", **moved},
            ],
            "prices": [
                _row(channel="vercel-ai-gateway", input_per_mtok=1, output_per_mtok=4),
                dict(gateway_row, effective_from="2026-10-01"),
            ],
        },
        {
            "canonical_id": "meta-llama/llama-x-instruct",
            "publisher": "meta-llama",
            "aliases": [
                {
                    "provider": "openrouter",
                    "alias": "meta-llama/llama-x-instruct",
                    "channel": "openrouter-api",
                    "effective_to": "2026-10-01",
                    "source_url": "https://example.com/review",
                }
            ],
            "prices": [dict(gateway_row, effective_to="2026-10-01")],
        },
    )


def test_gateway_discovery_follows_a_spelling_to_the_model_it_moved_to():
    source = _gateway_source()
    page = _openrouter_page(_openrouter_model("meta-llama/llama-x-instruct"))

    unchanged = discovery_evidence(
        source, page, _moved_spelling_document(), observed_on=OBSERVED_ON
    )
    assert unchanged["operations"] == []
    assert unchanged["identity_exclusions"] == []

    # A later price change lands on the model the spelling now names.
    changed = discovery_evidence(
        source, page, _moved_spelling_document(gateway_input=2), observed_on=OBSERVED_ON
    )
    assert [(o["op"], o["key"]["canonical_id"]) for o in changed["operations"]] == [
        ("replace_price", "meta/llama-x")
    ]
    assert changed["identity_exclusions"] == []


def test_gateway_discovery_follows_a_moved_spelling_another_gateway_keeps():
    # The duplicate keeps the spelling on Vercel, while OpenRouter's own
    # alias for it now names meta/llama-x.
    document = _moved_spelling_document(gateway_input=2)
    document["models"][1]["aliases"].append(
        {"provider": "vercel", "alias": "meta-llama/llama-x-instruct", "channel": "vercel-ai-gateway"}
    )
    page = _openrouter_page(_openrouter_model("meta-llama/llama-x-instruct"))

    evidence = discovery_evidence(_gateway_source(), page, document, observed_on=OBSERVED_ON)

    assert [(o["op"], o["key"]["canonical_id"]) for o in evidence["operations"]] == [
        ("replace_price", "meta/llama-x")
    ]
    assert evidence["identity_exclusions"] == []


def test_google_multimodal_output_escalates_instead_of_reverting_verified_rate():
    page = (
        "<h2>Example video</h2><p>gemini-example-video</p>"
        "<a>Try it in Google AI Studio</a><table>"
        "<tr><td>Input price</td><td>Not available</td><td>$1.50 (text / video)</td></tr>"
        "<tr><td>Output price</td><td>Not available</td>"
        "<td>$9.00 (text)<br>$17.50 (video)</td></tr></table>"
    )
    dimensions = ["input_per_mtok", "output_per_mtok"]
    source = _source(
        "google-api-pricing", "google", "google-api", "google",
        dimensions=dimensions, fingerprint=structural_fingerprint("google", page),
    )
    row = {
        "channel": "google-api", "region": "global", "effective_from": "2026-10-29",
        "input_per_mtok": 1.5, "output_per_mtok": 17.5,
        "source_url": "https://example.com/google-pricing",
    }
    document = _document(_model(
        "example/video", "gemini-example-video", row, channel="google-api", provider="google",
    ))
    result = _plan(document, _registry(source), {"google-api-pricing": page}, date(2026, 10, 30))
    assert result.delta["operations"] == []
    assert [(e.reason, e.dimension) for e in result.escalations] == [("missing_rate", "output_per_mtok")]
