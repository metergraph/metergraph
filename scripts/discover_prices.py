"""Gather pricing evidence and proposals for human review.

The script scans registry sources that are not low-risk (or those named with
``--sources``), the sources behind escalations from a refresh report
(``--escalations``), and new-source content supplied from a file or stdin
(``--new-source``). For each it emits deterministic evidence -- candidate
values, prior catalog values, the reason review is needed, and normalized
facts from the page -- plus a proposed delta for supported catalog changes. Raw
page content is never emitted or stored.

Registered sources are fetched only from their registered URLs, or read as
``<source-id>.*`` from ``--content-dir``. New-source content never comes
from a URL.

An optional adapter (``--llm-adapter module:callable``) may read each
source's content. It is called as ``propose(source, content, current_rows)``
and returns a list of delta operations. Its output is schema-validated and
recorded as review evidence only. Everything this script emits requires
review: it holds no credentials, cannot mark a source low-risk, and never
applies a delta, writes the catalog, publishes, or touches GitHub.

Usage:

    python scripts/discover_prices.py [--catalog prices.yaml] [--registry sources.yaml]
        [--sources id,...] [--escalations refresh.json] [--new-source PATH|-]
        [--content-dir DIR] [--llm-adapter module:callable]
        [--observed-on YYYY-MM-DD] [--json report.json]

Exit status: 0 the report was written, 1 an error.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Run against the in-repository core, not whatever version is installed.
for _path in (str(REPOSITORY_ROOT / "core" / "src"), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import yaml  # noqa: E402

from metergraph_core.catalog_delta import (  # noqa: E402
    CATALOG_DELTA_SCHEMA_VERSION,
    PRICING_DELTA_KIND,
    CatalogDeltaError,
    DeltaOperation,
    parse_catalog_delta,
)
from metergraph_core.extractors import page_tables, redact_evidence  # noqa: E402
from metergraph_core.loader import DEFAULT_CATALOG_PATH, _canonical_value, load_catalog_document  # noqa: E402
from metergraph_core.refresh import current_rows, discovery_evidence  # noqa: E402
from metergraph_core.sources import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    PriceSource,
    SourceRegistryError,
    load_source_registry,
)
from refresh_prices import content_file, fetch_text, refuse_overwrites, write_error  # noqa: E402

REPORT_SCHEMA_VERSION = 1
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_OPERATION_FIELDS = {
    "op", "key", "expect", "price", "effective_to", "model", "alias"
}


class DiscoveryError(Exception):
    pass


def load_adapter(spec: str) -> Callable[..., Any]:
    module_name, _, attribute = spec.partition(":")
    try:
        adapter = getattr(importlib.import_module(module_name), attribute or "propose")
    except Exception as exc:  # importing runs adapter code; report its type only
        raise DiscoveryError(f"cannot load adapter {spec!r}: {type(exc).__name__}") from None
    if not callable(adapter):
        raise DiscoveryError(f"cannot load adapter {spec!r}: not callable")
    return adapter


def _delta(operations: list, observations: list[str], evidence: list[dict]) -> dict[str, Any]:
    return {
        "schema_version": CATALOG_DELTA_SCHEMA_VERSION,
        "kind": PRICING_DELTA_KIND,
        "provenance": {
            "generator": "metergraph discovery",
            "requires_review": True,
            "observations": observations,
            "evidence": evidence,
        },
        "operations": operations,
    }


def _operation_json(operation: DeltaOperation) -> dict[str, Any]:
    """A validated operation in its parsed, JSON-native form: canonical dates
    and no fields beyond the schema's."""
    if operation.op == "add_model":
        key = {"canonical_id": operation.key.canonical_id}
    elif operation.op in ("add_alias", "close_alias"):
        key = {
            "canonical_id": operation.key.canonical_id,
            "provider": operation.key.provider,
            "alias": operation.key.alias,
            "channel": operation.key.channel,
        }
    else:
        key = {
            "canonical_id": operation.key.canonical_id,
            "channel": operation.key.channel,
            "region": operation.key.region,
            "effective_from": operation.key.effective_from,
        }
    data: dict[str, Any] = {
        "op": operation.op,
        "key": key,
        "expect": _canonical_value(operation.expect),
    }
    if operation.price is not None:
        data["price"] = _canonical_value(operation.price)
    if operation.effective_to is not None:
        data["effective_to"] = operation.effective_to
    if operation.model is not None:
        data["model"] = _canonical_value(operation.model)
    if operation.alias is not None:
        data["alias"] = _canonical_value(operation.alias)
    if operation.source_url is not None:
        data["source_url"] = operation.source_url
    return data


def run_adapter(adapter: Callable[..., Any], name: str, source: dict[str, Any], content: str, rows: list[dict]) -> dict[str, Any]:
    """Call the adapter and keep only schema-valid operations on the
    source's channel. Failures and rejections are reported with stable codes
    only: an adapter's messages and rejected items may quote page content, a
    prompt or a local path, so none of their text is emitted."""
    result: dict[str, Any] = {"name": name, "operations": [], "rejected": [], "error": None, "error_type": None}
    try:
        proposed = adapter(copy.deepcopy(source), content, copy.deepcopy(rows))
    except Exception as exc:  # an adapter failure is evidence, not a crash
        result["error"], result["error_type"] = "adapter_error", type(exc).__name__
        return result
    if not isinstance(proposed, list):
        result["error"], result["error_type"] = "adapter_error", "invalid_result"
        return result
    for index, item in enumerate(proposed):
        if isinstance(item, Mapping) and set(item) - _OPERATION_FIELDS:
            result["rejected"].append({"index": index, "error": "unknown_field"})
            continue
        try:
            [operation] = parse_catalog_delta(_delta([item], [], [])).operations
        except CatalogDeltaError:
            result["rejected"].append({"index": index, "error": "invalid_operation"})
            continue
        channel = source.get("channel")
        operation_channel = getattr(operation.key, "channel", None)
        if operation.op == "add_model":
            channels = {
                row.get("channel")
                for field in ("aliases", "prices")
                for row in operation.model.get(field) or []
                if isinstance(row, Mapping)
            }
            operation_channel = channel if channels == {channel} else None
        if channel is not None and operation_channel != channel:
            result["rejected"].append({"index": index, "error": "wrong_channel"})
            continue
        # The schema allows any mapping in expect and rules; only an
        # operation that encodes as strict JSON can enter the report.
        data = _operation_json(operation)
        try:
            json.dumps(data, allow_nan=False)
        except (TypeError, ValueError):
            result["rejected"].append({"index": index, "error": "invalid_operation"})
            continue
        result["operations"].append(data)
    return result


def _source_json(source: PriceSource) -> dict[str, Any]:
    """What an adapter learns about a source: no policy and no credentials."""
    return {
        "id": source.id,
        "provider": source.provider,
        "channel": source.channel,
        "url": source.url,
        "parser": source.parser,
        "dimensions": list(source.dimensions),
    }


def read_new_source(value: str) -> tuple[str, str]:
    if _URL.match(value):
        raise DiscoveryError("new-source content comes from a file or stdin, never a URL")
    if value == "-":
        return "stdin", sys.stdin.read()
    path = Path(value)
    return path.name, path.read_text(errors="ignore")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--sources", help="comma-separated registry source ids (default: every source that is not low-risk)")
    parser.add_argument("--escalations", type=Path, help="a refresh_prices.py JSON report whose escalated sources to scan")
    parser.add_argument("--new-source", help="new-source content: a file path, or - for stdin")
    parser.add_argument("--content-dir", type=Path, help="read <source-id>.* files here instead of fetching")
    parser.add_argument("--llm-adapter", help="optional module:callable implementing propose(source, content, current_rows)")
    parser.add_argument("--observed-on", type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    parser.add_argument("--json", type=Path, help="also write the report here")
    args = parser.parse_args(argv)

    try:
        inputs: list[Path] = []
        report = discover(args, inputs)
        # Nothing is written, not even the report on stdout, if the report
        # file would replace an input the run read.
        if args.json:
            refuse_overwrites([args.json], inputs)
    except (DiscoveryError, OSError, ValueError, SourceRegistryError, yaml.YAMLError) as exc:
        print(f"discover_prices: {exc}", file=sys.stderr)
        return 1
    # The file is written before the report reaches stdout, so a failed write
    # exits 1 with an error naming the path and the reason, and no report.
    try:
        text = json.dumps(report, indent=2) + "\n"
        if args.json:
            args.json.write_text(text)
    except OSError as exc:
        print(f"discover_prices: {write_error(exc)}", file=sys.stderr)
        return 1
    print(text, end="")
    return 0


def discover(args: argparse.Namespace, inputs: list[Path]) -> dict[str, Any]:
    """Build the report, recording every file read in ``inputs``."""
    new_source = read_new_source(args.new_source) if args.new_source else None
    if args.new_source and args.new_source != "-":
        inputs.append(Path(args.new_source))
    adapter = load_adapter(args.llm_adapter) if args.llm_adapter else None
    inputs += [args.catalog, args.registry]
    document = yaml.safe_load(args.catalog.read_text())
    load_catalog_document(document)
    registry = load_source_registry(args.registry)
    allowed = registry.allowed_urls()
    day = args.observed_on

    if args.sources is None:
        requested = {s.id for s in registry.sources if not s.low_risk.eligible}
    else:
        requested = {registry.get(name.strip()).id for name in args.sources.split(",") if name.strip()}
    escalations: dict[str, list[dict]] = {}
    if args.escalations:
        inputs.append(args.escalations)
        for escalation in json.loads(args.escalations.read_text()).get("escalations") or []:
            source_id = registry.get(str(escalation.get("source_id"))).id
            escalations.setdefault(source_id, []).append(escalation)

    fetched: dict[str, str] = {}
    proposals = []
    for source in registry.sources:
        if source.id not in requested and source.id not in escalations:
            continue
        reasons = [] if source.low_risk.eligible else ["not_low_risk"]
        reasons += sorted({f"escalation:{e.get('reason')}" for e in escalations.get(source.id, [])})
        proposal: dict[str, Any] = {
            "source_id": source.id,
            "requires_review": True,
            "reasons": reasons or ["requested"],
            "escalations": escalations.get(source.id, []),
            "evidence": None,
            "proposed_delta": None,
            "adapter": None,
            "error": None,
        }
        proposals.append(proposal)
        if args.content_dir is not None:
            path = content_file(args.content_dir, source.id)
            if path is None:
                proposal["error"] = f"no {source.id}.* file in {args.content_dir}"
                continue
            content = path.read_text(errors="ignore")
            inputs.append(path)
        else:
            if source.url not in fetched:
                try:
                    fetched[source.url] = fetch_text(source.url, allowed)
                except (OSError, ValueError) as exc:
                    proposal["error"] = f"fetch failed: {exc}"
                    continue
            content = fetched[source.url]
        evidence = discovery_evidence(source, content, document, observed_on=day)
        proposal["evidence"] = evidence
        if evidence["operations"]:
            delta = _delta(
                evidence["operations"],
                [f"{source.id}@{day.isoformat()}"],
                [{"source_id": source.id, "url": source.url, "fingerprint": evidence["fingerprint"]}],
            )
            parse_catalog_delta(delta)
            proposal["proposed_delta"] = delta
        if adapter is not None:
            proposal["adapter"] = run_adapter(
                adapter, args.llm_adapter, _source_json(source), content, current_rows(document, source.channel, day)
            )

    new_sources = []
    if new_source is not None:
        label, content = new_source
        entry: dict[str, Any] = {
            "requires_review": True,
            "label": label,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            # Raw page content is never emitted: report only normalized facts.
            "evidence": redact_evidence(
                {
                    "kind": "tables",
                    "tables": [{"headers": rows[0], "rows": rows[1:]} for rows in page_tables(content) if rows],
                }
            ),
            "adapter": None,
        }
        if adapter is not None:
            entry["adapter"] = run_adapter(adapter, args.llm_adapter, {"id": None, "label": label, "channel": None}, content, [])
        new_sources.append(entry)

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "observed_on": day.isoformat(),
        "proposals": proposals,
        "new_sources": new_sources,
    }


if __name__ == "__main__":
    sys.exit(main())
