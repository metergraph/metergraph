"""Canonical model identity and managed execution-route registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import yaml

from .catalog import _iso_text
from .loader import LoadedCatalog


EXECUTION_PROFILES = frozenset({"default", "bedrock"})
# Route providers use the catalog's normalized provider names: Moonshot is
# "moonshotai", the name its catalog aliases carry ("moonshot" is only an
# input alias the catalog normalizes).
ROUTE_PROVIDERS = frozenset(
    {"anthropic", "bedrock", "fireworks", "mistral", "moonshotai", "openai", "vercel"}
)
ROUTE_PROVIDER_CHANNELS = MappingProxyType(
    {
        "anthropic": "anthropic-api",
        "bedrock": "aws-bedrock",
        "fireworks": "fireworks-api",
        "mistral": "mistral-api",
        "moonshotai": "moonshot-api",
        "openai": "openai-api",
        "vercel": "vercel-ai-gateway",
    }
)
OFFER_GROUP_PROFILES = MappingProxyType(
    {
        "gateway": "default",
        "fireworks": "default",
        "openai-direct": "default",
        "anthropic-direct": "default",
        "moonshot-direct": "default",
        "mistral-direct": "default",
        "bedrock": "bedrock",
    }
)
OFFER_GROUP_CREDENTIALS = MappingProxyType(
    {
        "gateway": "AI_GATEWAY_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "openai-direct": "OPENAI_API_KEY",
        "anthropic-direct": "ANTHROPIC_API_KEY",
        "moonshot-direct": "MOONSHOT_API_KEY",
        "mistral-direct": "MISTRAL_API_KEY",
        "bedrock": None,
    }
)
OFFER_GROUP_PROVIDERS = MappingProxyType(
    {
        "gateway": "vercel",
        "fireworks": "fireworks",
        "openai-direct": "openai",
        "anthropic-direct": "anthropic",
        "moonshot-direct": "moonshotai",
        "mistral-direct": "mistral",
        "bedrock": "bedrock",
    }
)
DEFAULT_MODEL_REGISTRY_PATH = Path(__file__).parent / "data" / "models.yaml"


class ModelRegistryError(ValueError):
    """Raised when a model-registry document violates its contract."""


@dataclass(frozen=True, slots=True)
class ModelRoute:
    key: str
    id: str
    canonical_id: str
    display_name: str
    provider: str
    model_id: str
    pricing_channel: str
    execution_profiles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    canonical_id: str
    display_name: str
    publisher: str
    routes: tuple[ModelRoute, ...]
    # The publisher's general-availability date, when recorded.
    released: date | None = None


@dataclass(frozen=True, slots=True)
class ReleaseDate:
    """When a model became available, and what that date is based on:
    ``released`` is the registry's recorded date; ``first_price`` is the
    earliest price window the catalog has for the model, used when no date
    is recorded."""

    date: date
    basis: str


@dataclass(frozen=True, slots=True)
class OfferGroup:
    id: str
    credential: str | None
    route_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    version: str
    models: Mapping[str, ModelDefinition]
    routes: Mapping[str, ModelRoute]
    execution_profiles: Mapping[str, tuple[str, ...]]
    offer_groups: Mapping[str, OfferGroup]

    def model(self, canonical_id: str) -> ModelDefinition:
        return self.models[canonical_id]

    def release_date(
        self, canonical_id: str, catalog: LoadedCatalog
    ) -> ReleaseDate | None:
        """The model's release date: recorded when the registry has one,
        otherwise the earliest ``effective_from`` of its catalog prices, or
        None when neither exists. Callers never compute the fallback
        themselves, so every product reads the same answer."""
        model = self.models.get(canonical_id)
        if model is not None and model.released is not None:
            return ReleaseDate(model.released, "released")
        entry = next(
            (
                entry
                for entry in catalog.document.get("models", [])
                if entry.get("canonical_id") == canonical_id
            ),
            None,
        )
        # A price window starts on a date or a timestamp; the catalog accepts
        # both, so both are read here.
        first_price = min(
            (
                _window_day(price["effective_from"])
                for price in (entry or {}).get("prices", [])
                if price.get("effective_from") is not None
            ),
            default=None,
        )
        return None if first_price is None else ReleaseDate(first_price, "first_price")

    def route(self, route_key: str) -> ModelRoute:
        return self.routes[route_key]

    def routes_for_execution_profile(self, profile: str) -> tuple[ModelRoute, ...]:
        return tuple(
            self.routes[route_key]
            for route_key in self.execution_profiles[profile]
        )

    def candidates(self, offer_group: str) -> tuple[ModelRoute, ...]:
        group = self.offer_groups[offer_group]
        return tuple(self.routes[route_id] for route_id in group.route_ids)

    def reachable_candidates(
        self, credentials: Iterable[str]
    ) -> tuple[ModelRoute, ...]:
        present = frozenset(credentials)
        seen: set[str] = set()
        result: list[ModelRoute] = []
        for group in self.offer_groups.values():
            if group.credential is None or group.credential not in present:
                continue
            for route_id in group.route_ids:
                if route_id not in seen:
                    result.append(self.routes[route_id])
                    seen.add(route_id)
        return tuple(result)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ModelRegistryError(f"{field} must be a mapping")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ModelRegistryError(f"{field} must be a list")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelRegistryError(f"{field} must be a nonblank string")
    return value.strip()


def _unique_text_list(value: Any, field: str) -> tuple[str, ...]:
    items = tuple(_text(item, field) for item in _list(value, field))
    if not items:
        raise ModelRegistryError(f"{field} must not be empty")
    if len(set(items)) != len(items):
        raise ModelRegistryError(f"{field} contains a duplicate value")
    return items


def _window_day(value: Any) -> date:
    """The calendar day a catalog price window starts, from a date or a
    timestamp as the catalog loader accepts them."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(_iso_text(value)).date()


def _optional_date(value: Any, field: str) -> date | None:
    """An ISO calendar date, or None when absent. A datetime is not accepted:
    a release is a day, and the registry version is already a date."""
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ModelRegistryError(f"{field} must be an ISO date") from exc


def _registry_version_date(version: str) -> date:
    date_value, separator, revision = version.partition(".")
    try:
        parsed = date.fromisoformat(date_value)
    except ValueError as exc:
        raise ModelRegistryError(
            "models document version must be an ISO date, optionally followed by .N"
        ) from exc
    if separator and (
        not revision.isascii()
        or not revision.isdecimal()
        or revision.startswith("0")
    ):
        raise ModelRegistryError(
            "models document version must be an ISO date, optionally followed by .N"
        )
    return parsed


def parse_model_registry(document: Any) -> ModelRegistry:
    """Parse and freeze one versioned model-registry document."""
    root = _mapping(document, "models document")
    version = _text(root.get("version"), "models document version")
    _registry_version_date(version)

    models: dict[str, ModelDefinition] = {}
    routes: dict[str, ModelRoute] = {}
    candidate_profiles: set[tuple[str, str]] = set()
    physical_routes: set[tuple[str, str]] = set()
    for model_value in _list(root.get("models"), "models document models"):
        model = _mapping(model_value, "model entry")
        canonical_id = _text(model.get("canonical_id"), "model entry canonical_id")
        if canonical_id in models:
            raise ModelRegistryError(f"duplicate canonical model {canonical_id!r}")
        display_name = _text(
            model.get("display_name"), f"{canonical_id}: display_name"
        )
        publisher = _text(model.get("publisher"), f"{canonical_id}: publisher")
        if canonical_id.split("/", 1)[0] != publisher:
            raise ModelRegistryError(
                f"{canonical_id}: publisher {publisher!r} does not match canonical id"
            )
        released = _optional_date(model.get("released"), f"{canonical_id}: released")
        model_routes: list[ModelRoute] = []
        route_values = _list(model.get("routes"), f"{canonical_id}: routes")
        if not route_values:
            raise ModelRegistryError(f"{canonical_id}: routes must not be empty")
        for route_value in route_values:
            route = _mapping(route_value, f"{canonical_id}: route")
            route_key = _text(route.get("key"), f"{canonical_id}: route key")
            route_id = _text(route.get("id"), f"{canonical_id}: route id")
            if route_key in routes:
                raise ModelRegistryError(f"duplicate route id/key {route_key!r}")
            profiles = _unique_text_list(
                route.get("execution_profiles"),
                f"{route_id}: execution_profiles",
            )
            unknown_profiles = set(profiles) - EXECUTION_PROFILES
            if unknown_profiles:
                unknown = sorted(unknown_profiles)[0]
                raise ModelRegistryError(
                    f"{route_id}: unknown execution profile {unknown!r}"
                )
            for profile in profiles:
                candidate_profile = (profile, route_id)
                if candidate_profile in candidate_profiles:
                    raise ModelRegistryError(
                        f"duplicate candidate id {route_id!r} in execution "
                        f"profile {profile!r}"
                    )
                candidate_profiles.add(candidate_profile)
            provider = _text(route.get("provider"), f"{route_id}: needs provider")
            if provider not in ROUTE_PROVIDERS:
                raise ModelRegistryError(
                    f"{route_id}: unknown route provider {provider!r}"
                )
            model_id = _text(route.get("model_id"), f"{route_id}: needs model_id")
            pricing_channel = _text(
                route.get("pricing_channel"),
                f"{route_id}: needs pricing_channel",
            )
            if pricing_channel != ROUTE_PROVIDER_CHANNELS[provider]:
                raise ModelRegistryError(
                    f"{route_id}: provider {provider!r} cannot use pricing channel "
                    f"{pricing_channel!r}"
                )
            physical_route = (provider, model_id)
            if physical_route in physical_routes:
                raise ModelRegistryError(
                    f"{route_id}: duplicate physical route {physical_route!r}"
                )
            physical_routes.add(physical_route)
            parsed = ModelRoute(
                key=route_key,
                id=route_id,
                canonical_id=canonical_id,
                display_name=_text(
                    route.get("display_name", display_name),
                    f"{route_id}: display_name",
                ),
                provider=provider,
                model_id=model_id,
                pricing_channel=pricing_channel,
                execution_profiles=profiles,
            )
            routes[route_key] = parsed
            model_routes.append(parsed)
        models[canonical_id] = ModelDefinition(
            canonical_id=canonical_id,
            display_name=display_name,
            publisher=publisher,
            routes=tuple(model_routes),
            released=released,
        )

    execution_profiles: dict[str, tuple[str, ...]] = {}
    declared_candidate_profiles: set[tuple[str, str]] = set()
    for profile_value in _list(
        root.get("execution_profiles"), "models document execution_profiles"
    ):
        profile = _mapping(profile_value, "execution profile")
        profile_id = _text(profile.get("id"), "execution profile id")
        if profile_id not in EXECUTION_PROFILES:
            raise ModelRegistryError(f"unknown execution profile {profile_id!r}")
        if profile_id in execution_profiles:
            raise ModelRegistryError(f"duplicate execution profile {profile_id!r}")
        route_keys = _unique_text_list(
            profile.get("routes"), f"{profile_id}: execution routes"
        )
        for route_key in route_keys:
            if route_key not in routes:
                raise ModelRegistryError(
                    f"{profile_id}: execution profile references unknown route "
                    f"{route_key!r}"
                )
            if profile_id not in routes[route_key].execution_profiles:
                raise ModelRegistryError(
                    f"{profile_id}: route {route_key!r} is not available to "
                    f"execution profile {profile_id!r}"
                )
            declared_candidate_profiles.add((profile_id, route_key))
        execution_profiles[profile_id] = route_keys
    route_candidate_profiles = {
        (profile, route.key)
        for route in routes.values()
        for profile in route.execution_profiles
    }
    if declared_candidate_profiles != route_candidate_profiles:
        missing = sorted(route_candidate_profiles - declared_candidate_profiles)
        raise ModelRegistryError(
            f"execution profiles omit declared routes {missing!r}"
        )

    offer_groups: dict[str, OfferGroup] = {}
    for group_value in _list(
        root.get("offer_groups"), "models document offer_groups"
    ):
        group = _mapping(group_value, "offer group")
        group_id = _text(group.get("id"), "offer group id")
        if group_id not in OFFER_GROUP_PROFILES:
            raise ModelRegistryError(f"unknown offer group {group_id!r}")
        if group_id in offer_groups:
            raise ModelRegistryError(f"duplicate offer group {group_id!r}")
        credential_value = group.get("credential")
        credential = (
            None
            if credential_value is None
            else _text(credential_value, f"{group_id}: credential")
        )
        expected_credential = OFFER_GROUP_CREDENTIALS[group_id]
        if credential != expected_credential:
            raise ModelRegistryError(
                f"{group_id}: requires credential {expected_credential!r}"
            )
        route_ids = _unique_text_list(
            group.get("routes"), f"{group_id}: routes"
        )
        expected_profile = OFFER_GROUP_PROFILES[group_id]
        for route_id in route_ids:
            if route_id not in routes:
                raise ModelRegistryError(
                    f"{group_id}: offer group references unknown route {route_id!r}"
                )
            if expected_profile not in routes[route_id].execution_profiles:
                raise ModelRegistryError(
                    f"{group_id}: route {route_id!r} is not available to "
                    f"execution profile {expected_profile!r}"
                )
            expected_provider = OFFER_GROUP_PROVIDERS[group_id]
            if routes[route_id].provider != expected_provider:
                raise ModelRegistryError(
                    f"{group_id}: route {route_id!r} requires provider "
                    f"{expected_provider!r}"
                )
        offer_groups[group_id] = OfferGroup(
            id=group_id,
            credential=credential,
            route_ids=route_ids,
        )

    return ModelRegistry(
        version=version,
        models=MappingProxyType(models),
        routes=MappingProxyType(routes),
        execution_profiles=MappingProxyType(execution_profiles),
        offer_groups=MappingProxyType(offer_groups),
    )


def load_model_registry(path: str | Path | None = None) -> ModelRegistry:
    """Load the bundled model registry or an explicit replacement path."""
    resolved = Path(path) if path is not None else DEFAULT_MODEL_REGISTRY_PATH
    return parse_model_registry(yaml.safe_load(resolved.read_bytes()))


def validate_model_registry(
    registry: ModelRegistry, catalog: LoadedCatalog
) -> None:
    """Require every execution route to resolve in the pricing catalog."""
    validation_time = datetime.combine(
        _registry_version_date(registry.version), time.min, tzinfo=timezone.utc
    )
    catalog_models = {
        entry.get("canonical_id"): entry
        for entry in catalog.document.get("models", [])
    }
    for route in registry.routes.values():
        catalog_model = catalog_models.get(route.canonical_id)
        if catalog_model is None:
            raise ModelRegistryError(
                f"{route.id}: unknown canonical model {route.canonical_id!r}"
            )
        deployment_alias = next(
            (
                alias
                for alias in (catalog_model or {}).get("aliases", [])
                if str(alias.get("alias", "")).strip().lower()
                == route.model_id.lower()
                and alias.get("channel") == route.pricing_channel
            ),
            None,
        )
        if deployment_alias is None:
            raise ModelRegistryError(
                f"{route.id}: model {route.model_id!r} has no pricing channel "
                f"{route.pricing_channel!r}"
            )
        canonical = catalog.canonical_model_id(route.provider, route.model_id, at=validation_time)
        if canonical == route.model_id:
            resolved = catalog.snapshot.resolve_price(
                model=route.model_id,
                channel=route.pricing_channel,
                at=validation_time,
            )
            if resolved is not None:
                canonical = resolved.canonical_model
        if canonical != route.canonical_id:
            raise ModelRegistryError(
                f"{route.id}: pricing resolves canonical model {canonical!r}, "
                f"expected {route.canonical_id!r}"
            )
        active_channel = any(
            price.get("channel") == route.pricing_channel
            and date.fromisoformat(str(price.get("effective_from")))
            <= validation_time.date()
            and (
                price.get("effective_to") is None
                or validation_time.date()
                < date.fromisoformat(str(price["effective_to"]))
            )
            for price in (catalog_model or {}).get("prices", [])
        )
        if not active_channel:
            raise ModelRegistryError(
                f"{route.id}: no price for pricing channel "
                f"{route.pricing_channel!r}"
            )
