import copy
import dataclasses
from types import MappingProxyType

import pytest

from metergraph_core.models import (
    EffectiveTrainingPolicy,
    ModelDefinition,
    ModelRegistry,
    ModelRegistryError,
    ROUTE_PROVIDER_CHANNELS,
    TrainingPolicy,
    load_model_registry,
    parse_model_registry,
)
MODEL = "example/model"
GATEWAY = "gateway:example/model"
DIRECT = "openai-direct:openai:example-model"
SOURCE = "https://example.com/terms"


def document():
    """A minimal synthetic registry with one model on a gateway route and a
    direct OpenAI route. It is only parsed, never validated against prices."""
    return {
        "version": "2026-09-25",
        "models": [
            {
                "canonical_id": MODEL,
                "display_name": "Example Model",
                "publisher": "example",
                "routes": [
                    {
                        "key": GATEWAY,
                        "id": MODEL,
                        "provider": "vercel",
                        "model_id": MODEL,
                        "pricing_channel": "vercel-ai-gateway",
                        "execution_profiles": ["default"],
                    },
                    {
                        "key": DIRECT,
                        "id": "openai:example-model",
                        "provider": "openai",
                        "model_id": "example-model",
                        "pricing_channel": "openai-api",
                        "execution_profiles": ["default"],
                    },
                ],
            }
        ],
        "execution_profiles": [{"id": "default", "routes": [GATEWAY, DIRECT]}],
        "offer_groups": [
            {"id": "gateway", "credential": "AI_GATEWAY_API_KEY", "routes": [GATEWAY]},
            {"id": "openai-direct", "credential": "OPENAI_API_KEY", "routes": [DIRECT]},
        ],
    }


def prohibited(**extra):
    return {"training_on_customer_data": "prohibited", "source": SOURCE, **extra}


def with_defaults(value=None, **pairs):
    value = value or document()
    value["provider_defaults"] = {
        provider: {ROUTE_PROVIDER_CHANNELS[provider]: {"training_policy": policy}}
        for provider, policy in pairs.items()
    }
    return value


def test_old_documents_parse_and_resolve_every_route_to_unknown():
    registry = parse_model_registry(document())
    for key in (GATEWAY, DIRECT):
        effective = registry.effective_training_policy(key)
        assert effective == EffectiveTrainingPolicy(
            policy=TrainingPolicy("unknown"),
            origin="unspecified",
            registry_version="2026-09-25",
        )
    assert registry.route(GATEWAY).training_policy is None
    assert registry.model(MODEL).training_policy_overrides == {}
    assert registry.training_policy_defaults == {}


def test_unknown_route_key_is_an_error():
    with pytest.raises(KeyError):
        parse_model_registry(document()).effective_training_policy("missing:route")


def test_provider_default_applies_only_to_its_serving_provider_channel():
    registry = parse_model_registry(with_defaults(openai=prohibited()))
    direct = registry.effective_training_policy(DIRECT)
    assert direct.origin == "provider"
    assert direct.policy == TrainingPolicy("prohibited", source=SOURCE)
    # The gateway route serves an OpenAI-published model, but the publisher
    # never supplies a policy for another serving provider.
    gateway = registry.effective_training_policy(GATEWAY)
    assert (gateway.policy.training_on_customer_data, gateway.origin) == ("unknown", "unspecified")


def test_model_override_beats_provider_default_and_is_scoped_to_its_provider():
    value = with_defaults(openai=prohibited())
    value["models"][0]["training_policy_overrides"] = {
        "vercel": {"vercel-ai-gateway": {"training_policy": prohibited(source="https://example.com/gw")}},
        "openai": {"openai-api": {"training_policy": {
            "training_on_customer_data": "conditional",
            "source": "https://example.com/model",
            "conditions": "Requires an opted-out organization.",
        }}},
    }
    registry = parse_model_registry(value)
    direct = registry.effective_training_policy(DIRECT)
    assert direct.origin == "model"
    # Whole-object replacement: nothing is inherited from the default.
    assert direct.policy == TrainingPolicy(
        "conditional",
        source="https://example.com/model",
        conditions="Requires an opted-out organization.",
    )
    gateway = registry.effective_training_policy(GATEWAY)
    assert gateway.origin == "model"
    assert gateway.policy == TrainingPolicy("prohibited", source="https://example.com/gw")


def test_route_policy_beats_model_override_and_provider_default():
    value = with_defaults(openai=prohibited())
    value["models"][0]["training_policy_overrides"] = {
        "openai": {"openai-api": {"training_policy": prohibited()}},
    }
    value["models"][0]["routes"][1]["training_policy"] = {
        "training_on_customer_data": "permitted",
        "source": "https://example.com/route",
    }
    effective = parse_model_registry(value).effective_training_policy(DIRECT)
    assert effective.origin == "route"
    assert effective.policy == TrainingPolicy("permitted", source="https://example.com/route")


def test_explicit_unknown_blocks_inheritance():
    value = with_defaults(openai=prohibited())
    value["models"][0]["routes"][1]["training_policy"] = {
        "training_on_customer_data": "unknown",
    }
    effective = parse_model_registry(value).effective_training_policy(DIRECT)
    assert effective.origin == "route"
    assert effective.policy == TrainingPolicy("unknown")


def test_parsed_policy_maps_and_values_are_immutable():
    value = with_defaults(openai=prohibited())
    value["models"][0]["training_policy_overrides"] = {
        "vercel": {"vercel-ai-gateway": {"training_policy": prohibited()}},
    }
    registry = parse_model_registry(value)
    assert isinstance(registry.training_policy_defaults, MappingProxyType)
    assert isinstance(registry.training_policy_defaults["openai"], MappingProxyType)
    overrides = registry.model(MODEL).training_policy_overrides
    assert isinstance(overrides, MappingProxyType)
    assert isinstance(overrides["vercel"], MappingProxyType)
    with pytest.raises(TypeError):
        registry.training_policy_defaults["openai"]["x"] = TrainingPolicy("unknown")
    with pytest.raises(dataclasses.FrozenInstanceError):
        registry.training_policy_defaults["openai"]["openai-api"].source = "https://x"
    effective = registry.effective_training_policy(DIRECT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        effective.origin = "route"


def test_parsing_does_not_retain_the_input_document():
    value = with_defaults(openai=prohibited())
    registry = parse_model_registry(value)
    value["provider_defaults"]["openai"]["openai-api"]["training_policy"]["source"] = "https://x"
    assert registry.training_policy_defaults["openai"]["openai-api"].source == SOURCE


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ({"training_on_customer_data": "forbidden", "source": SOURCE}, "training_on_customer_data"),
        ({"training_on_customer_data": None}, "training_on_customer_data"),
        ({"source": SOURCE}, "training_on_customer_data"),
        ({"training_on_customer_data": "prohibited"}, "source"),
        ({"training_on_customer_data": "permitted", "source": ""}, "source"),
        ({"training_on_customer_data": "prohibited", "source": "http://example.com/terms"}, "HTTPS"),
        ({"training_on_customer_data": "prohibited", "source": "https://"}, "HTTPS"),
        ({"training_on_customer_data": "prohibited", "source": "example.com"}, "HTTPS"),
        ({"training_on_customer_data": "prohibited", "source": "https://[broken"}, "HTTPS"),
        ({"training_on_customer_data": "prohibited", "source": "https://:443/terms"}, "HTTPS"),
        ({"training_on_customer_data": "unknown", "source": "https://[broken"}, "HTTPS"),
        ({"training_on_customer_data": "unknown", "source": "ftp://example.com"}, "HTTPS"),
        ({"training_on_customer_data": "conditional", "source": SOURCE}, "conditions"),
        ({"training_on_customer_data": "conditional", "source": SOURCE, "conditions": "  "}, "conditions"),
        (prohibited(conditions="Paid tier only."), "conditions"),
        ({"training_on_customer_data": "unknown", "conditions": "x"}, "conditions"),
        (prohibited(notes="legacy prose"), "notes"),
        (prohibited(notes=3), "notes"),
        (prohibited(verified_at="2026-11-17"), "unknown field 'verified_at'"),
        (prohibited(sorce=SOURCE), "unknown field 'sorce'"),
        ("prohibited", "must be a mapping"),
    ],
)
def test_invalid_policy_objects_fail_closed(policy, message):
    with pytest.raises(ModelRegistryError, match=message):
        parse_model_registry(with_defaults(openai=policy))
    value = document()
    value["models"][0]["routes"][1]["training_policy"] = policy
    with pytest.raises(ModelRegistryError, match=message):
        parse_model_registry(value)


@pytest.mark.parametrize("conditions", [None, ""])
def test_conditions_may_be_absent_or_empty_outside_conditional(conditions):
    registry = parse_model_registry(with_defaults(openai=prohibited(conditions=conditions)))
    assert registry.training_policy_defaults["openai"]["openai-api"].conditions is None


def test_unknown_needs_no_source_but_may_cite_one():
    registry = parse_model_registry(with_defaults(
        openai={"training_on_customer_data": "unknown"},
        vercel={"training_on_customer_data": "unknown", "source": SOURCE},
    ))
    assert registry.training_policy_defaults["openai"]["openai-api"] == TrainingPolicy("unknown")
    assert registry.training_policy_defaults["vercel"]["vercel-ai-gateway"].source == SOURCE


def test_text_fields_are_trimmed():
    registry = parse_model_registry(with_defaults(openai={
        "training_on_customer_data": " conditional ",
        "source": f" {SOURCE} ",
        "conditions": " Paid tier. ",
    }))
    assert registry.training_policy_defaults["openai"]["openai-api"] == TrainingPolicy(
        "conditional", source=SOURCE, conditions="Paid tier."
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("defaults_not_mapping", "provider_defaults must be a mapping"),
        ("unknown_provider", "unknown route provider 'vercle'"),
        ("wrong_channel", "cannot use pricing channel"),
        ("channels_not_mapping", "must be a mapping"),
        ("entry_not_mapping", "must be a mapping"),
        ("entry_missing_policy", "training_policy"),
        ("entry_unknown_field", "unknown field 'training_polcy'"),
        ("override_unknown_provider", "unknown route provider 'vercle'"),
        ("override_wrong_channel", "cannot use pricing channel"),
        ("override_without_route", "no route"),
        ("override_unknown_field", "unknown field 'policy'"),
        ("overrides_not_mapping", "training_policy_overrides must be a mapping"),
        ("route_policy_typo", "unknown field 'training_polcy'"),
        ("model_override_typo", "unknown field 'training_policy_override'"),
        ("root_defaults_typo", "unknown field 'provider_default'"),
    ],
)
def test_invalid_policy_placement_fails_closed(mutation, message):
    value = with_defaults(openai=prohibited())
    defaults = value["provider_defaults"]
    model = value["models"][0]
    entry = {"training_policy": prohibited()}
    if mutation == "defaults_not_mapping":
        value["provider_defaults"] = []
    elif mutation == "unknown_provider":
        defaults["vercle"] = {"vercel-ai-gateway": entry}
    elif mutation == "wrong_channel":
        defaults["openai"] = {"vercel-ai-gateway": entry}
    elif mutation == "channels_not_mapping":
        defaults["openai"] = [entry]
    elif mutation == "entry_not_mapping":
        defaults["openai"]["openai-api"] = "prohibited"
    elif mutation == "entry_missing_policy":
        defaults["openai"]["openai-api"] = {}
    elif mutation == "entry_unknown_field":
        defaults["openai"]["openai-api"]["training_polcy"] = prohibited()
    elif mutation == "override_unknown_provider":
        model["training_policy_overrides"] = {"vercle": {"vercel-ai-gateway": entry}}
    elif mutation == "override_wrong_channel":
        model["training_policy_overrides"] = {"vercel": {"openai-api": entry}}
    elif mutation == "override_without_route":
        model["training_policy_overrides"] = {"anthropic": {"anthropic-api": entry}}
    elif mutation == "override_unknown_field":
        model["training_policy_overrides"] = {"vercel": {"vercel-ai-gateway": {"policy": prohibited()}}}
    elif mutation == "overrides_not_mapping":
        model["training_policy_overrides"] = [entry]
    elif mutation == "route_policy_typo":
        model["routes"][0]["training_polcy"] = prohibited()
    elif mutation == "model_override_typo":
        model["training_policy_override"] = {}
    else:
        value["provider_default"] = value.pop("provider_defaults")
    with pytest.raises(ModelRegistryError, match=message):
        parse_model_registry(value)


def test_empty_maps_are_valid():
    value = document()
    value["provider_defaults"] = {}
    value["models"][0]["training_policy_overrides"] = {}
    registry = parse_model_registry(value)
    assert registry.effective_training_policy(DIRECT).origin == "unspecified"


def test_training_policy_validates_direct_construction():
    assert TrainingPolicy("unknown").source is None
    with pytest.raises(ModelRegistryError, match="source"):
        TrainingPolicy("prohibited")
    with pytest.raises(ModelRegistryError, match="training_on_customer_data"):
        TrainingPolicy("maybe")


def test_bundled_registry_resolves_a_policy_for_every_route():
    registry = load_model_registry()
    for key, route in registry.routes.items():
        effective = registry.effective_training_policy(key)
        assert effective.registry_version == registry.version
        assert effective.origin in {"route", "model", "provider", "unspecified"}
        if effective.policy.training_on_customer_data != "unknown":
            assert effective.policy.source.startswith("https://"), key


def test_bundled_gateway_routes_never_inherit_publisher_terms():
    """A gateway route is only non-unknown when the gateway's own default or
    a specific gateway override says so, never through the publisher."""
    registry = load_model_registry()
    gateway_default = registry.training_policy_defaults.get("vercel", {}).get("vercel-ai-gateway")
    for key, route in registry.routes.items():
        if route.provider != "vercel":
            continue
        effective = registry.effective_training_policy(key)
        if effective.origin == "provider":
            assert effective.policy == gateway_default
        elif effective.origin == "unspecified":
            assert effective.policy.training_on_customer_data == "unknown"


def test_every_serving_provider_has_a_sourced_bundled_default():
    registry = load_model_registry()
    policies = {
        (provider, channel): policy
        for provider, channels in registry.training_policy_defaults.items()
        for channel, policy in channels.items()
    }
    assert set(policies) == set(ROUTE_PROVIDER_CHANNELS.items())
    # Sources live in the catalog, not in a second test fixture that must be
    # edited whenever a provider moves its policy page.
    for policy in policies.values():
        assert policy.source and policy.source.startswith("https://")


def test_bundled_gateway_default_is_unknown_despite_its_optional_filter():
    policy = load_model_registry().training_policy_defaults["vercel"]["vercel-ai-gateway"]
    assert policy.training_on_customer_data == "unknown"
    assert policy.conditions is None


def test_bundled_mistral_default_names_its_prerequisites():
    policy = load_model_registry().training_policy_defaults["mistral"]["mistral-api"]
    assert "opt-out" in policy.conditions
    assert "Labs or Preview" in policy.conditions
    assert "feedback" in policy.conditions


def test_policy_map_defaults_are_empty_and_immutable_on_direct_construction():
    """Regression: a mappingproxy default must not be a dataclass default
    value, which Python 3.11 rejects at import time."""
    registry = parse_model_registry(document())
    model = registry.model(MODEL)
    direct = ModelDefinition(model.canonical_id, model.display_name, model.publisher, ())
    assert direct.training_policy_overrides == {}
    assert isinstance(direct.training_policy_overrides, MappingProxyType)
    bare = ModelRegistry(
        registry.version, registry.models, registry.routes,
        registry.execution_profiles, registry.offer_groups,
    )
    assert isinstance(bare.training_policy_defaults, MappingProxyType)
    assert bare.effective_training_policy(DIRECT).origin == "unspecified"


def test_models_routes_and_policies_stay_hashable():
    value = with_defaults(openai=prohibited())
    value["models"][0]["training_policy_overrides"] = {
        "vercel": {"vercel-ai-gateway": {"training_policy": prohibited()}},
    }
    value["models"][0]["routes"][1]["training_policy"] = prohibited()
    with_overrides = parse_model_registry(value).model(MODEL)
    plain = parse_model_registry(document()).model(MODEL)
    for item in (with_overrides, plain, *with_overrides.routes):
        assert hash(item) == hash(item)
    assert {with_overrides, plain} == {with_overrides, plain}
    # Overrides still take part in equality, so these remain distinct.
    assert with_overrides != dataclasses.replace(
        with_overrides, training_policy_overrides=MappingProxyType({})
    )
    assert plain == parse_model_registry(document()).model(MODEL)
    assert len({plain, parse_model_registry(document()).model(MODEL)}) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"training_on_customer_data": "conditional", "source": SOURCE, "conditions": ""}, "conditions"),
        ({"training_on_customer_data": "conditional", "source": SOURCE, "conditions": " "}, "conditions"),
        ({"training_on_customer_data": "prohibited", "source": 3}, "source"),
        ({"training_on_customer_data": "prohibited", "source": "https://[broken"}, "HTTPS"),
        ({"training_on_customer_data": ["unknown"]}, "training_on_customer_data"),
    ],
)
def test_direct_construction_rejects_blank_or_mistyped_values(kwargs, message):
    with pytest.raises(ModelRegistryError, match=message):
        TrainingPolicy(**kwargs)


@pytest.mark.parametrize(
    "key", ["defaults_note", "default_region", "training_set", "policy"]
)
def test_unrelated_legacy_fields_are_still_ignored(key):
    value = document()
    value[key] = "x"
    value["models"][0][key] = "x"
    value["models"][0]["routes"][0][key] = "x"
    parse_model_registry(value)


@pytest.mark.parametrize(
    ("level", "key"),
    [
        ("root", "training_policy"),
        ("model", "training_policy"),
        ("model", "provider_defaults"),
        ("route", "training_policy_overrides"),
        ("route", "provider_defaults"),
    ],
)
def test_policy_fields_at_the_wrong_level_fail_closed(level, key):
    value = document()
    target = {
        "root": value,
        "model": value["models"][0],
        "route": value["models"][0]["routes"][0],
    }[level]
    target[key] = {}
    with pytest.raises(ModelRegistryError, match=f"unknown field '{key}'"):
        parse_model_registry(value)


def test_training_policy_api_is_public():
    from metergraph_core import EffectiveTrainingPolicy as Exported
    from metergraph_core import TrainingPolicy as ExportedPolicy

    assert Exported is EffectiveTrainingPolicy
    assert ExportedPolicy is TrainingPolicy


def test_policy_documents_round_trip_through_copy():
    value = with_defaults(openai=prohibited())
    assert parse_model_registry(copy.deepcopy(value)) == parse_model_registry(value)


@pytest.mark.parametrize("key", ["trainng_policy", "trianing_policy"])
def test_misspelled_route_override_cannot_restore_inherited_commitment(key):
    value = with_defaults(openai=prohibited())
    value["models"][0]["routes"][1][key] = {"training_on_customer_data": "unknown"}
    with pytest.raises(ModelRegistryError, match=f"unknown field '{key}'"):
        parse_model_registry(value)
