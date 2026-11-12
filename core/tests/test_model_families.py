"""Stable model product-line families declared in the prices document."""

import copy
import dataclasses

import pytest

from metergraph_core import (
    CatalogDeltaError,
    CatalogError,
    ModelFamily,
    apply_catalog_delta,
    load_catalog,
    load_catalog_document,
    parse_catalog,
    parse_catalog_delta,
)

SONNET = {"id": "example/sonnet-line", "display_name": "Example Sonnet"}
HAIKU = {"id": "example/haiku-line", "display_name": "Example Haiku"}


def _model(canonical_id, *aliases, family=None, publisher="example"):
    entry = {
        "canonical_id": canonical_id,
        "publisher": publisher,
        "aliases": [
            dict({"provider": "example-api", "channel": "example-api"}, **alias)
            for alias in aliases
        ],
        "prices": [
            {
                "channel": "example-api",
                "region": "global",
                "effective_from": "2026-01-01",
                "input_per_mtok": 1.0,
                "output_per_mtok": 4.0,
                "source_url": "https://example.com/pricing",
            }
        ],
    }
    if family is not None:
        entry["family"] = copy.deepcopy(family)
    return entry


def _document(*models):
    return {
        "version": "test",
        "currency": "USD",
        "pricing_verified_at": "2026-10-01",
        "models": list(models),
    }


def _moving(name, start, end=None):
    alias = {"alias": name, "effective_from": start, "source_url": "https://example.com/changelog"}
    if end is not None:
        alias["effective_to"] = end
    return alias


def _family_document():
    return _document(
        _model(
            "example/sonnet-1",
            {"alias": "sonnet-1"},
            _moving("sonnet-latest", "2026-01-01", "2026-06-01"),
            _moving("chat-latest", "2026-01-01", "2026-06-01"),
            family=SONNET,
        ),
        _model(
            "example/sonnet-2",
            {"alias": "sonnet-2"},
            _moving("sonnet-latest", "2026-06-01"),
            family=SONNET,
        ),
        _model(
            "example/haiku-1",
            {"alias": "haiku-1"},
            _moving("chat-latest", "2026-06-01"),
            _moving("fast-latest", "2026-01-01", "2026-06-01"),
            family=HAIKU,
        ),
        _model(
            "example/unlabeled",
            {"alias": "unlabeled"},
            _moving("fast-latest", "2026-06-01"),
        ),
    )


# --- Canonical lookup ----------------------------------------------------------


def test_model_family_returns_the_declared_family():
    catalog = load_catalog_document(_family_document())

    family = catalog.model_family("example/sonnet-2")

    assert family == ModelFamily(id="example/sonnet-line", display_name="Example Sonnet")
    assert catalog.model_family("example/sonnet-1") == family
    assert catalog.model_family("example/haiku-1").display_name == "Example Haiku"


def test_model_family_is_frozen():
    family = load_catalog_document(_family_document()).model_family("example/sonnet-1")

    with pytest.raises(dataclasses.FrozenInstanceError):
        family.id = "example/other"


def test_model_without_family_and_unknown_ids_have_none():
    catalog = load_catalog_document(_family_document())

    assert catalog.model_family("example/unlabeled") is None
    assert catalog.model_family("example/missing") is None
    assert catalog.model_family("sonnet-1") is None  # aliases are not canonical ids
    assert catalog.model_family(None) is None


def test_catalog_without_any_family_metadata_stays_valid():
    catalog = load_catalog_document(
        _document(_model("example/a", {"alias": "a"}), _model("example/b", {"alias": "b"}))
    )

    assert catalog.model_family("example/a") is None
    assert catalog.family_for_model("example-api", "a") is None
    assert dict(catalog.families) == {}


def test_family_metadata_changes_no_resolution_or_price():
    with_family = load_catalog_document(_family_document())
    stripped = _family_document()
    for entry in stripped["models"]:
        entry.pop("family", None)
    without_family = load_catalog_document(stripped)

    for name in ("sonnet-1", "sonnet-latest", "chat-latest", "fast-latest"):
        for at in ("2026-02-01", "2026-07-01"):
            assert with_family.canonical_model_id(
                "example-api", name, at=at
            ) == without_family.canonical_model_id("example-api", name, at=at)
            assert with_family.snapshot.cost(
                provider="example-api", model=name, at=at, input_tokens=1000, output_tokens=10
            ) == without_family.snapshot.cost(
                provider="example-api", model=name, at=at, input_tokens=1000, output_tokens=10
            )
    assert dict(with_family.canonical_ids) == dict(without_family.canonical_ids)


# --- Raw provider/model lookup -------------------------------------------------


def test_family_for_model_reads_aliases_and_canonical_ids():
    catalog = load_catalog_document(_family_document())

    assert catalog.family_for_model("example-api", "sonnet-1").id == "example/sonnet-line"
    assert catalog.family_for_model("example-api", "example/sonnet-2").id == "example/sonnet-line"
    assert catalog.family_for_model(" Example-API ", " SONNET-2 ").id == "example/sonnet-line"


def test_family_for_model_spans_every_alias_date_when_all_agree():
    catalog = load_catalog_document(_family_document())

    # sonnet-latest moved from sonnet-1 to sonnet-2; both are one product line.
    assert catalog.family_for_model("example-api", "sonnet-latest").id == "example/sonnet-line"


def test_family_for_model_is_none_when_bindings_mix_families():
    catalog = load_catalog_document(_family_document())

    # chat-latest moved across product lines, so no family is safe for it.
    assert catalog.family_for_model("example-api", "chat-latest") is None


def test_family_for_model_is_none_when_any_binding_lacks_a_family():
    catalog = load_catalog_document(_family_document())

    # fast-latest moved from a labeled model to an unlabeled one.
    assert catalog.family_for_model("example-api", "fast-latest") is None
    assert catalog.family_for_model("example-api", "unlabeled") is None


def test_family_for_model_needs_an_explicit_binding_for_the_provider():
    catalog = load_catalog_document(_family_document())

    assert catalog.family_for_model("other-provider", "sonnet-1") is None
    assert catalog.family_for_model("other-provider", "example/sonnet-1") is None
    assert catalog.family_for_model("example-api", "sonnet-9") is None
    assert catalog.family_for_model("example-api", None) is None
    assert catalog.family_for_model(None, "sonnet-1") is None


def test_family_for_model_uses_the_providers_normalized_spelling():
    document = _family_document()
    document["models"][0]["aliases"].append(
        {"provider": "bedrock", "alias": "example.sonnet-1", "channel": "aws-bedrock"}
    )
    catalog = load_catalog_document(document)

    assert catalog.family_for_model("aws", "example.sonnet-1").id == "example/sonnet-line"
    assert catalog.family_for_model("amazon-bedrock", "example.sonnet-1").id == "example/sonnet-line"


def _without_families(document):
    stripped = copy.deepcopy(document)
    for entry in stripped["models"]:
        entry.pop("family", None)
    return stripped


def _synonym_document(first_provider, second_provider, *, second_family, dated=False):
    # One spelling declared under two spellings of one provider, bound to two
    # models. Dated windows keep keys that a provider shares from overlapping.
    first = {"provider": first_provider, "alias": "moving"}
    second = {"provider": second_provider, "alias": "moving"}
    if dated:
        first.update(_moving("moving", "2026-01-01", "2026-06-01"))
        second.update(_moving("moving", "2026-06-01"))
    return _document(
        _model("example/sonnet-1", first, family=SONNET),
        _model("example/other-1", second, family=second_family),
    )


@pytest.mark.parametrize(
    ("first", "second", "dated"),
    [("google", "google-genai", True), ("bedrock", "aws", True)],
)
@pytest.mark.parametrize("second_family", [HAIKU, None], ids=["mixed", "missing"])
def test_family_for_model_combines_bindings_across_provider_spellings(
    first, second, dated, second_family
):
    document = _synonym_document(first, second, second_family=second_family, dated=dated)
    catalog = load_catalog_document(document)

    for provider in (first, second):
        assert catalog.family_for_model(provider, "moving") is None


@pytest.mark.parametrize(
    ("first", "second", "dated"),
    [("google", "google-genai", True), ("bedrock", "aws", True)],
)
def test_family_for_model_agrees_across_provider_spellings(first, second, dated):
    document = _synonym_document(first, second, second_family=SONNET, dated=dated)
    catalog = load_catalog_document(document)

    for provider in (first, second):
        assert catalog.family_for_model(provider, "moving").id == "example/sonnet-line"


@pytest.mark.parametrize(
    ("first", "second", "dated"),
    [("google", "google-genai", True), ("bedrock", "aws", True)],
)
def test_provider_spelling_families_leave_resolution_and_price_unchanged(first, second, dated):
    document = _synonym_document(first, second, second_family=HAIKU, dated=dated)
    with_family = load_catalog_document(document)
    without_family = load_catalog_document(_without_families(document))

    for provider in (first, second):
        for at in ("2026-02-01", "2026-07-01"):
            assert with_family.canonical_model_id(
                provider, "moving", at=at
            ) == without_family.canonical_model_id(provider, "moving", at=at)
            assert with_family.snapshot.cost(
                provider=provider, model="moving", at=at, input_tokens=1000, output_tokens=10
            ) == without_family.snapshot.cost(
                provider=provider, model="moving", at=at, input_tokens=1000, output_tokens=10
            )
    assert dict(with_family.canonical_ids) == dict(without_family.canonical_ids)


def test_family_for_model_never_strips_a_bedrock_version_suffix():
    document = _family_document()
    document["models"][0]["aliases"].append(
        {"provider": "bedrock", "alias": "example.sonnet-1", "channel": "aws-bedrock"}
    )
    catalog = load_catalog_document(document)

    assert catalog.family_for_model("bedrock", "example.sonnet-1-v1:0") is None


# --- Validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "message"),
    [
        (None, "family must be a mapping"),
        ("example/sonnet-line", "family must be a mapping"),
        (["example/sonnet-line"], "family must be a mapping"),
        ({"display_name": "Example Sonnet"}, "family needs id"),
        ({"id": "example/sonnet-line"}, "family needs display_name"),
        ({"id": "", "display_name": "Example Sonnet"}, "family needs id"),
        ({"id": 7, "display_name": "Example Sonnet"}, "family needs id"),
        ({"id": "example/sonnet-line", "display_name": "  "}, "family needs display_name"),
        ({"id": "example/sonnet-line", "display_name": 7}, "family needs display_name"),
        ({"id": "example/sonnet-line", "display_name": "S", "successor": "x"}, "unknown field"),
        ({"id": "sonnet-line", "display_name": "Example Sonnet"}, "publisher/name"),
        ({"id": "example/", "display_name": "Example Sonnet"}, "publisher/name"),
        ({"id": "example/a/b", "display_name": "Example Sonnet"}, "publisher/name"),
        ({"id": "Example/Sonnet-Line", "display_name": "Example Sonnet"}, "publisher/name"),
        ({"id": " example/sonnet-line", "display_name": "Example Sonnet"}, "publisher/name"),
        ({"id": "example/sonnet line", "display_name": "Example Sonnet"}, "publisher/name"),
        ({"id": "other/sonnet-line", "display_name": "Example Sonnet"}, "publisher 'example'"),
    ],
)
def test_malformed_family_is_rejected(family, message):
    entry = _model("example/sonnet-1", {"alias": "sonnet-1"})
    entry["family"] = family

    with pytest.raises(CatalogError, match=message):
        load_catalog_document(_document(entry))


def test_family_needs_a_model_publisher():
    entry = _model("example/sonnet-1", {"alias": "sonnet-1"}, family=SONNET)
    del entry["publisher"]

    with pytest.raises(CatalogError, match="family needs the model's publisher"):
        load_catalog_document(_document(entry))


def test_one_family_id_cannot_carry_two_display_names():
    document = _document(
        _model("example/sonnet-1", {"alias": "sonnet-1"}, family=SONNET),
        _model(
            "example/sonnet-2",
            {"alias": "sonnet-2"},
            family={"id": "example/sonnet-line", "display_name": "Example Sonnet Next"},
        ),
    )

    with pytest.raises(CatalogError, match="conflicting display_name"):
        load_catalog_document(document)


@pytest.mark.parametrize(
    "display_name",
    [" Example Sonnet", "Example Sonnet ", "Example\nSonnet", "Example Sonnet\n", "Example\tSonnet"],
)
def test_display_name_with_padding_or_line_breaks_is_rejected_not_trimmed(display_name):
    entry = _model(
        "example/sonnet-1",
        {"alias": "sonnet-1"},
        family={"id": "example/sonnet-line", "display_name": display_name},
    )

    with pytest.raises(CatalogError, match="must be one line without surrounding whitespace"):
        load_catalog_document(_document(entry))


def test_display_name_may_be_ordinary_unicode():
    entry = _model(
        "example/sonnet-1",
        {"alias": "sonnet-1"},
        family={"id": "example/sonnet-line", "display_name": "Exämple Sönnet · Ünïcode"},
    )

    family = load_catalog_document(_document(entry)).model_family("example/sonnet-1")
    assert family.display_name == "Exämple Sönnet · Ünïcode"


def test_family_needs_canonical_id_under_the_publisher():
    entry = _model("other/sonnet-1", {"alias": "sonnet-1"}, family=SONNET)

    with pytest.raises(CatalogError, match="canonical id under publisher 'example'"):
        load_catalog_document(_document(entry))


def test_canonical_publisher_mismatch_without_family_stays_valid():
    entry = _model("other/sonnet-1", {"alias": "sonnet-1"})

    assert load_catalog_document(_document(entry)).model_family("other/sonnet-1") is None


# --- parse_catalog --------------------------------------------------------------


@pytest.mark.parametrize(
    ("models", "message"),
    [
        ([dict(_model("example/a", {"alias": "a"}), family=None)], "family must be a mapping"),
        (
            [_model("example/a", {"alias": "a"}, family={"id": "other/x", "display_name": "X"})],
            "must start with publisher",
        ),
        (
            [
                _model("example/a", {"alias": "a"}, family=SONNET),
                _model(
                    "example/b",
                    {"alias": "b"},
                    family={"id": "example/sonnet-line", "display_name": "Other"},
                ),
            ],
            "conflicting display_name",
        ),
    ],
)
def test_parse_catalog_rejects_invalid_family_as_load_does(models, message):
    with pytest.raises(CatalogError, match=message):
        parse_catalog(_document(*models))
    with pytest.raises(CatalogError, match=message):
        load_catalog_document(_document(*models))


def test_parse_catalog_returns_the_same_tuple_with_or_without_families():
    with_family = _family_document()
    stripped = _family_document()
    for entry in stripped["models"]:
        entry.pop("family", None)

    assert parse_catalog(with_family) == parse_catalog(stripped)


# --- Catalog deltas -----------------------------------------------------------


def _delta(*operations):
    return {
        "schema_version": 2,
        "kind": "pricing_delta",
        "provenance": {"observations": ["observation:example:2026-10-20"]},
        "operations": list(operations),
    }


def _add_model(model):
    return {
        "op": "add_model",
        "key": {"canonical_id": model["canonical_id"]},
        "expect": None,
        "model": model,
    }


def test_add_model_carries_a_valid_family():
    model = _model("example/sonnet-3", {"alias": "sonnet-3"}, family=SONNET)
    result = apply_catalog_delta(_family_document(), _delta(_add_model(model)))

    assert load_catalog_document(result).model_family("example/sonnet-3").id == (
        "example/sonnet-line"
    )


def test_add_model_rejects_a_malformed_family():
    model = _model("example/sonnet-3", {"alias": "sonnet-3"})
    model["family"] = {"id": "other/sonnet-line", "display_name": "Example Sonnet"}

    with pytest.raises(CatalogDeltaError, match="invalid model: .*family"):
        parse_catalog_delta(_delta(_add_model(model)))


def test_add_model_rejects_a_family_label_conflict_with_the_catalog():
    model = _model(
        "example/sonnet-3",
        {"alias": "sonnet-3"},
        family={"id": "example/sonnet-line", "display_name": "Example Sonnet Next"},
    )

    with pytest.raises(CatalogDeltaError, match="conflicting display_name"):
        apply_catalog_delta(_family_document(), _delta(_add_model(model)))


# --- Bundled catalog ------------------------------------------------------------


def test_bundled_catalog_declares_stable_product_lines():
    catalog = load_catalog()

    expected = {
        "anthropic/claude-sonnet-4.6": ("anthropic/claude-sonnet", "Claude Sonnet"),
        "anthropic/claude-sonnet-5": ("anthropic/claude-sonnet", "Claude Sonnet"),
        "anthropic/claude-sonnet-5.5": ("anthropic/claude-sonnet", "Claude Sonnet"),
        "anthropic/claude-opus-4.6": ("anthropic/claude-opus", "Claude Opus"),
        "anthropic/claude-opus-5.5": ("anthropic/claude-opus", "Claude Opus"),
        "anthropic/claude-haiku-4.5": ("anthropic/claude-haiku", "Claude Haiku"),
        "anthropic/claude-haiku-5.5": ("anthropic/claude-haiku", "Claude Haiku"),
        "google/gemini-2.5-pro": ("google/gemini-pro", "Gemini Pro"),
        "google/gemini-3.5-flash": ("google/gemini-flash", "Gemini Flash"),
        "google/gemini-3.5-flash-lite": ("google/gemini-flash-lite", "Gemini Flash-Lite"),
        "openai/gpt-5.4-mini": ("openai/gpt-mini", "GPT mini"),
        "openai/gpt-5.4-nano": ("openai/gpt-nano", "GPT nano"),
        "openai/gpt-6.1-sol": ("openai/gpt-sol", "GPT Sol"),
        "openai/gpt-6-luna": ("openai/gpt-luna", "GPT Luna"),
        "openai/gpt-6-astra": ("openai/gpt-astra", "GPT Astra"),
        "openai/gpt-6.1-terra": ("openai/gpt-terra", "GPT Terra"),
        "openai/gpt-5.5": ("openai/gpt", "GPT"),
        "deepseek/v4-flash": ("deepseek/deepseek-flash", "DeepSeek Flash"),
        "deepseek/v4-pro": ("deepseek/deepseek-pro", "DeepSeek Pro"),
        "meta/muse-spark-1.3": ("meta/muse-spark", "Muse Spark"),
        "meta/muse-wonder-1.0": ("meta/muse-wonder", "Muse Wonder"),
        "perplexity/sonar": ("perplexity/sonar", "Sonar"),
        "xai/grok-4.7": ("xai/grok", "Grok"),
    }
    actual = {
        canonical_id: (family.id, family.display_name)
        for canonical_id in expected
        if (family := catalog.model_family(canonical_id)) is not None
    }
    assert actual == expected


def test_bundled_catalog_leaves_mixed_aliases_without_a_family():
    catalog = load_catalog()

    # deepseek-chat has served models from more than one product line.
    assert catalog.family_for_model("deepseek", "deepseek-chat") is None
    assert catalog.family_for_model("anthropic", "claude-sonnet-4-6").id == (
        "anthropic/claude-sonnet"
    )


def test_every_bundled_family_id_is_scoped_to_its_models_publisher():
    catalog = load_catalog()
    publishers = {
        entry["canonical_id"]: entry.get("publisher") for entry in catalog.document["models"]
    }

    assert catalog.families
    for canonical_id, family in catalog.families.items():
        assert family.id.split("/", 1)[0] == publishers[canonical_id]
