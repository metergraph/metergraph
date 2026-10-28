"""The daily discovery script, run against fixture content with optional
stub adapters. No test touches the network or a real LLM."""

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
import yaml

from metergraph_core.extractors import structural_fingerprint

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "discover_prices.py"
_spec = importlib.util.spec_from_file_location("discover_prices", _SCRIPT)
discover_prices = importlib.util.module_from_spec(_spec)
sys.modules["discover_prices"] = discover_prices
_spec.loader.exec_module(discover_prices)

FIXTURES = Path(__file__).parent / "fixtures" / "sources"
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
OPENAI_URL = "https://example.com/openai-pricing"
TABLES_URL = "https://example.com/tables-pricing"
LICENSE_URL = "https://example.com/license-pricing"
SHARED_URL = "https://example.com/shared-pricing"


def _fixture(source_id):
    return (FIXTURES / f"{source_id}.html").read_text()


def _generic(name):
    return {"token-table": TOKEN_TABLE_PAGE, "gpu-license-prose": GPU_LICENSE_PAGE}[name]


def _openai():
    dimensions = ["input_per_mtok", "cache_read_per_mtok", "cache_write_5m_per_mtok", "output_per_mtok"]
    return {
        "id": "openai-api-pricing", "provider": "openai", "channel": "openai-api", "catalog_channel": True,
        "url": OPENAI_URL, "parser": "openai",
        "dimensions": dimensions, "fingerprint": structural_fingerprint("openai", _fixture("openai-api-pricing")),
        "low_risk": {"eligible": True, "dimensions": dimensions},
    }


def _discovery_only(source_id, provider, channel, parser, url, **overrides):
    entry = {
        "id": source_id, "provider": provider, "channel": channel, "catalog_channel": True,
        "url": url, "parser": parser,
        "dimensions": ["input_per_mtok", "output_per_mtok"], "fingerprint": None,
        "low_risk": {"eligible": False, "dimensions": []},
    }
    entry.update(overrides)
    return entry


def _sources():
    return [
        _openai(),
        _discovery_only("table-source-pricing", "example-tables", "tables-api", "html-tables", TABLES_URL),
        _discovery_only("shared-page-pricing", "example-shared", "shared-api", "html-tables", SHARED_URL),
        _discovery_only("shared-page-geo-pricing", "example-shared", "shared-geo-api", "html-tables", SHARED_URL),
        _discovery_only(
            "license-source-pricing", "example-license", "license-only", "gpu-license-prose", LICENSE_URL,
            catalog_channel=False,
            dimensions=["gpu_hour_usd", "license_per_gpu_year_usd"],
        ),
    ]


def _row(**rates):
    return dict({"channel": "openai-api", "region": "global", "effective_from": "2026-01-01", "source_url": "https://example.com/model"}, **rates)


def _catalog(gpt_x_input=2.0):
    return {
        "version": "2026-10-19", "currency": "USD", "pricing_verified_at": "2026-10-19",
        "models": [
            {"canonical_id": "example/gpt-x", "aliases": [{"provider": "openai", "alias": "gpt-x", "channel": "openai-api"}],
             "prices": [_row(input_per_mtok=gpt_x_input, cache_read_per_mtok=0.2, cache_write_5m_per_mtok=2.5, output_per_mtok=8.0)]},
            {"canonical_id": "example/gpt-x-mini", "aliases": [{"provider": "openai", "alias": "gpt-x-mini", "channel": "openai-api"}],
             "prices": [_row(input_per_mtok=0.4, cache_read_per_mtok=0.04, output_per_mtok=1.6)]},
        ],
    }


@pytest.fixture
def paths(tmp_path):
    catalog = tmp_path / "prices.yaml"
    registry = tmp_path / "sources.yaml"
    content = tmp_path / "content"
    content.mkdir()
    (content / "openai-api-pricing.html").write_text(_fixture("openai-api-pricing"))
    for source_id in ("table-source-pricing", "shared-page-pricing", "shared-page-geo-pricing"):
        (content / f"{source_id}.html").write_text(_generic("token-table"))
    (content / "license-source-pricing.html").write_text(_generic("gpu-license-prose"))
    catalog.write_text(yaml.safe_dump(_catalog()))
    registry.write_text(yaml.safe_dump({"schema_version": 1, "sources": _sources()}))
    base = ["--catalog", str(catalog), "--registry", str(registry), "--observed-on", "2026-10-20"]
    return {"base": base, "catalog": catalog, "content": content, "tmp": tmp_path}


def _run(argv, capsys):
    status = discover_prices.main(argv)
    captured = capsys.readouterr()
    assert status == 0, captured.err
    return json.loads(captured.out)


def _proposal(output, source_id):
    [proposal] = [p for p in output["proposals"] if p["source_id"] == source_id]
    return proposal


def test_catalog_additions_select_delta_schema_version_two():
    operation = {
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
                }
            ],
            "prices": [
                {
                    "channel": "openrouter-api",
                    "region": "global",
                    "effective_from": "2026-10-20",
                    "input_per_mtok": "1",
                    "output_per_mtok": "4",
                    "source_url": "https://example.com/models",
                }
            ],
        },
    }
    delta = discover_prices._delta([operation], ["observation"], [])

    assert delta["schema_version"] == 2
    assert discover_prices._operation_json(
        discover_prices.parse_catalog_delta(delta).operations[0]
    ) == operation


def test_price_only_discovery_uses_catalog_delta_schema_version():
    operation = {
        "op": "add_price",
        "key": {
            "canonical_id": "example/model-a",
            "channel": "openrouter-api",
            "region": "global",
            "effective_from": "2026-10-20",
        },
        "expect": None,
        "price": {
            "channel": "openrouter-api",
            "region": "global",
            "effective_from": "2026-10-20",
            "input_per_mtok": "1",
            "output_per_mtok": "4",
            "source_url": "https://example.com/models",
        },
    }

    delta = discover_prices._delta([operation], ["observation"], [])

    assert delta["schema_version"] == 2


def test_discovery_without_an_adapter_emits_deterministic_evidence(paths, capsys):
    argv = paths["base"] + ["--content-dir", str(paths["content"])]
    first = _run(argv, capsys)
    second = _run(argv, capsys)
    assert first == second
    assert [p["source_id"] for p in first["proposals"]] == [
        "table-source-pricing", "shared-page-pricing", "shared-page-geo-pricing", "license-source-pricing",
    ]
    tables = _proposal(first, "table-source-pricing")
    assert tables["requires_review"] is True
    assert tables["reasons"] == ["not_low_risk"]
    assert tables["evidence"]["evidence"]["kind"] == "tables"
    assert tables["proposed_delta"] is None
    assert tables["adapter"] is None
    licensing = _proposal(first, "license-source-pricing")
    assert licensing["evidence"]["evidence"]["candidates"] == {"license_per_gpu_year_usd": "1000", "gpu_hour_usd": "0.50"}
    assert licensing["proposed_delta"] is None


def test_discovery_output_carries_no_applicable_delta(paths, capsys):
    output = _run(paths["base"] + ["--content-dir", str(paths["content"])], capsys)
    assert "delta" not in output
    assert all(p["requires_review"] is True for p in output["proposals"])


def test_discovery_fetches_only_registry_urls_once_each(paths, capsys, monkeypatch):
    fetched = []
    pages = {TABLES_URL: "token-table", SHARED_URL: "token-table", LICENSE_URL: "gpu-license-prose"}

    def fake_fetch(url, allowed):
        assert url in allowed
        fetched.append(url)
        return _generic(pages[url])

    monkeypatch.setattr(discover_prices, "fetch_text", fake_fetch)
    _run(paths["base"], capsys)
    assert fetched == [TABLES_URL, SHARED_URL, LICENSE_URL]


def _escalation_report(tmp, reason="dimension_not_allowed"):
    report = tmp / "refresh.json"
    report.write_text(json.dumps({
        "schema_version": 1,
        "observed_on": "2026-10-20",
        "escalations": [{"result": "escalated", "source_id": "openai-api-pricing", "reason": reason, "key": None,
                         "model_key": "gpt-x", "region": "global", "dimension": "input_per_mtok",
                         "current": "1", "observed": "2", "detail": ""}],
    }))
    return report


def test_escalated_low_risk_sources_get_reviewable_proposed_deltas(paths, capsys):
    paths["catalog"].write_text(yaml.safe_dump(_catalog(gpt_x_input=1.0)))
    report = _escalation_report(paths["tmp"])
    output = _run(paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "", "--escalations", str(report)], capsys)
    [proposal] = output["proposals"]
    assert proposal["source_id"] == "openai-api-pricing"
    assert proposal["reasons"] == ["escalation:dimension_not_allowed"]
    assert proposal["escalations"][0]["model_key"] == "gpt-x"
    delta = proposal["proposed_delta"]
    assert delta["provenance"]["requires_review"] is True
    assert delta["provenance"]["observations"] == ["openai-api-pricing@2026-10-20"]
    [operation] = delta["operations"]
    assert operation["key"]["canonical_id"] == "example/gpt-x"
    assert operation["price"]["input_per_mtok"] == "2"


def test_an_unparsable_catalog_rate_is_review_evidence_not_a_crash(paths, capsys):
    paths["catalog"].write_text(yaml.safe_dump(_catalog(gpt_x_input="not-a-rate")))
    report = _escalation_report(paths["tmp"])
    output = _run(paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "", "--escalations", str(report)], capsys)
    proposal = _proposal(output, "openai-api-pricing")
    [candidate] = [c for c in proposal["evidence"]["candidates"] if c["model_key"] == "gpt-x"]
    assert (candidate["reason"], candidate["invalid_dimensions"]) == ("invalid_current_rate", ["input_per_mtok"])
    assert candidate["prior"]["input_per_mtok"] == "not-a-rate"
    assert proposal["evidence"]["operations"] == []
    assert proposal["proposed_delta"] is None


def test_a_new_model_from_an_unstructured_source_stays_evidence(paths, capsys):
    report = _escalation_report(paths["tmp"], reason="unknown_model")
    output = _run(paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "", "--escalations", str(report)], capsys)
    proposal = _proposal(output, "openai-api-pricing")
    assert [c["model_key"] for c in proposal["evidence"]["candidates"] if c["reason"] == "unknown_model"] == ["gpt-legacy"]
    assert proposal["proposed_delta"] is None


def test_a_new_source_is_read_from_a_file(paths, capsys):
    page = paths["tmp"] / "new-provider.html"
    page.write_text("<table><tr><th>Model</th><th>Input</th></tr><tr><td>New X</td><td>$1</td></tr></table>")
    output = _run(paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "", "--new-source", str(page)], capsys)
    [new_source] = output["new_sources"]
    assert new_source["requires_review"] is True
    assert new_source["label"] == "new-provider.html"
    # Raw page content is never emitted: only normalized facts, never cells.
    assert new_source["evidence"] == {
        "kind": "tables",
        "redacted": True,
        "tables": [{"columns": 2, "rows": [{"label": "new-x", "amounts": ["1"]}]}],
    }
    assert "New X" not in json.dumps(output)
    assert len(new_source["sha256"]) == 64


def test_a_new_source_is_read_from_stdin(paths, capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("<table><tr><th>Plan</th></tr></table>"))
    output = _run(paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "", "--new-source", "-"], capsys)
    [new_source] = output["new_sources"]
    assert new_source["label"] == "stdin"
    assert new_source["evidence"]["tables"] == [{"columns": 1, "rows": []}]


def test_the_report_repeats_no_page_cells(paths, capsys):
    output = _run(paths["base"] + ["--content-dir", str(paths["content"])], capsys)
    text = json.dumps(output)
    for cell in ("Example Model Large", "Input (per 1M tokens)", "Example Accelerator", "Licenses start at"):
        assert cell not in text
    assert _proposal(output, "table-source-pricing")["evidence"]["evidence"]["tables"][0]["rows"][0] == {
        "label": "example-model-large",
        "amounts": ["2.00", "0.20", "8.00"],
    }


@pytest.mark.parametrize("value", ["https://example.com/new-pricing", "http://example.com/x", "file:///tmp/page.html"])
def test_a_new_source_url_is_refused(paths, capsys, monkeypatch, value):
    monkeypatch.setattr(discover_prices, "fetch_text", lambda url, allowed: pytest.fail("fetched"))
    assert discover_prices.main(paths["base"] + ["--sources", "", "--new-source", value]) == 1
    assert "never a URL" in capsys.readouterr().err


def _adapter(tmp, monkeypatch, body):
    module = tmp / "stub_pricing_adapter.py"
    module.write_text(body)
    monkeypatch.syspath_prepend(str(tmp))
    sys.modules.pop("stub_pricing_adapter", None)
    return "stub_pricing_adapter:propose"


RECORDING_ADAPTER = '''
CALLS = []

def propose(*args, **kwargs):
    CALLS.append((args, kwargs))
    return []
'''


def test_the_adapter_receives_only_the_source_its_content_and_current_rows(paths, capsys, monkeypatch):
    spec = _adapter(paths["tmp"], monkeypatch, RECORDING_ADAPTER)
    _run(paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "table-source-pricing", "--llm-adapter", spec], capsys)
    import stub_pricing_adapter

    [(args, kwargs)] = stub_pricing_adapter.CALLS
    assert kwargs == {}
    source, content, current_rows = args
    assert source == {
        "id": "table-source-pricing", "provider": "example-tables", "channel": "tables-api", "url": TABLES_URL,
        "parser": "html-tables",
        "dimensions": ["input_per_mtok", "output_per_mtok"],
    }
    assert content == _generic("token-table")
    assert current_rows == []


def test_the_adapter_sees_current_rows_on_its_channel(paths, capsys, monkeypatch):
    spec = _adapter(paths["tmp"], monkeypatch, RECORDING_ADAPTER)
    report = _escalation_report(paths["tmp"])
    _run(paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "", "--escalations", str(report), "--llm-adapter", spec], capsys)
    import stub_pricing_adapter

    [(args, _)] = stub_pricing_adapter.CALLS
    assert [(r["canonical_id"], r["input_per_mtok"]) for r in args[2]] == [("example/gpt-x", 2.0), ("example/gpt-x-mini", 0.4)]


PROPOSING_ADAPTER = '''
ROW = {"channel": "tables-api", "region": "global", "effective_from": "2026-10-20",
       "input_per_mtok": "0.9", "output_per_mtok": "0.9", "source_url": "https://example.com/tables-pricing"}
KEY = {"canonical_id": "example/gpt-x", "channel": "tables-api", "region": "global", "effective_from": "2026-10-20"}

def propose(source, content, current_rows):
    return [
        {"op": "add_price", "key": KEY, "expect": None, "price": ROW},
        {"op": "add_price", "key": KEY, "expect": None, "price": dict(ROW, input_per_mtok=0.9)},
        {"op": "add_price", "key": dict(KEY, channel="openai-api"), "expect": None, "price": dict(ROW, channel="openai-api")},
        {"op": "add_price", "key": KEY, "expect": None, "price": ROW, "low_risk": True},
        "not an operation",
    ]
'''


def test_adapter_output_is_schema_validated_and_invalid_items_are_rejected_with_reasons(paths, capsys, monkeypatch):
    spec = _adapter(paths["tmp"], monkeypatch, PROPOSING_ADAPTER)
    output = _run(paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "table-source-pricing", "--llm-adapter", spec], capsys)
    proposal = _proposal(output, "table-source-pricing")
    adapter = proposal["adapter"]
    assert adapter["name"] == "stub_pricing_adapter:propose"
    assert [op["key"]["channel"] for op in adapter["operations"]] == ["tables-api"]
    # Rejections carry stable codes only, never text derived from the item.
    assert adapter["rejected"] == [
        {"index": 1, "error": "invalid_operation"},
        {"index": 2, "error": "wrong_channel"},
        {"index": 3, "error": "unknown_field"},
        {"index": 4, "error": "invalid_operation"},
    ]
    assert proposal["requires_review"] is True
    assert proposal["proposed_delta"] is None


def test_an_adapter_cannot_mark_anything_low_risk_or_applicable(paths, capsys, monkeypatch):
    spec = _adapter(paths["tmp"], monkeypatch, PROPOSING_ADAPTER)
    output = _run(paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "table-source-pricing", "--llm-adapter", spec], capsys)
    text = json.dumps(output)
    assert '"low_risk"' not in text
    assert "delta" not in output
    for proposal in output["proposals"]:
        assert proposal["requires_review"] is True


def test_a_failing_adapter_is_reported_without_its_message(paths, capsys, monkeypatch):
    # An adapter's message may quote the page, a prompt or a local path.
    body = "def propose(source, content, rows):\n    raise RuntimeError('quoted page text at /private/prompt.txt')\n"
    spec = _adapter(paths["tmp"], monkeypatch, body)
    output = _run(paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "table-source-pricing", "--llm-adapter", spec], capsys)
    adapter = _proposal(output, "table-source-pricing")["adapter"]
    assert adapter["operations"] == []
    assert (adapter["error"], adapter["error_type"]) == ("adapter_error", "RuntimeError")
    assert "quoted page text" not in json.dumps(output) and "prompt.txt" not in json.dumps(output)


def test_an_adapter_that_returns_a_non_list_is_reported(paths, capsys, monkeypatch):
    spec = _adapter(paths["tmp"], monkeypatch, "def propose(source, content, rows):\n    return {'op': 'add_price'}\n")
    output = _run(paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "table-source-pricing", "--llm-adapter", spec], capsys)
    adapter = _proposal(output, "table-source-pricing")["adapter"]
    assert (adapter["error"], adapter["error_type"]) == ("adapter_error", "invalid_result")


LEAKING_ADAPTER = """
def propose(source, content, current_rows):
    return [
        {"op": "copied page text", "key": {}, "expect": None},
        {"op": "add_price", "key": {}, "expect": None, "quoted prompt": 1},
    ]
"""


def test_rejected_adapter_items_never_echo_their_contents(paths, capsys, monkeypatch):
    spec = _adapter(paths["tmp"], monkeypatch, LEAKING_ADAPTER)
    output = _run(paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "table-source-pricing", "--llm-adapter", spec], capsys)
    adapter = _proposal(output, "table-source-pricing")["adapter"]
    assert adapter["rejected"] == [
        {"index": 0, "error": "invalid_operation"},
        {"index": 1, "error": "unknown_field"},
    ]
    text = json.dumps(output)
    assert "copied page text" not in text and "quoted prompt" not in text


DATED_ADAPTER = """
import datetime

def propose(source, content, current_rows):
    day = datetime.date(2026, 10, 20)
    key = {"canonical_id": "example/gpt-x", "channel": "tables-api", "region": "global", "effective_from": day}
    price = {"channel": "tables-api", "region": "global", "effective_from": day,
             "input_per_mtok": "0.9", "output_per_mtok": "0.9", "source_url": "https://example.com/tables-pricing"}
    return [{"op": "add_price", "key": key, "expect": None, "price": price}]
"""


NON_JSON_ADAPTER = """
ROW = {"channel": "tables-api", "region": "global", "effective_from": "2026-10-20",
       "input_per_mtok": "0.9", "output_per_mtok": "0.9", "source_url": "https://example.com/tables-pricing"}
KEY = {"canonical_id": "example/gpt-x", "channel": "tables-api", "region": "global", "effective_from": "2026-10-20"}

def propose(source, content, current_rows):
    old = dict(KEY, effective_from="2026-01-01")
    return [
        {"op": "add_price", "key": KEY, "expect": None, "price": dict(ROW, rules={"tiers": {1, 2}})},
        {"op": "close_price", "key": old, "expect": {"rules": {"tags": {"a"}}}, "effective_to": "2026-10-20"},
        {"op": "close_price", "key": old, "expect": {"input_per_mtok": float("nan")}, "effective_to": "2026-10-20"},
        {"op": "add_price", "key": KEY, "expect": None, "price": ROW},
    ]
"""


def test_adapter_operations_that_are_not_json_are_rejected(paths, capsys, monkeypatch):
    # The schema allows any mapping in expect and rules; the report must not.
    spec = _adapter(paths["tmp"], monkeypatch, NON_JSON_ADAPTER)
    report = paths["tmp"] / "discovery.json"
    output = _run(
        paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "table-source-pricing", "--llm-adapter", spec, "--json", str(report)],
        capsys,
    )
    adapter = _proposal(output, "table-source-pricing")["adapter"]
    assert adapter["rejected"] == [
        {"index": 0, "error": "invalid_operation"},
        {"index": 1, "error": "invalid_operation"},
        {"index": 2, "error": "invalid_operation"},
    ]
    assert [op["op"] for op in adapter["operations"]] == ["add_price"]
    assert json.loads(report.read_text()) == output


def test_adapter_operations_with_date_values_are_reported_as_json(paths, capsys, monkeypatch):
    spec = _adapter(paths["tmp"], monkeypatch, DATED_ADAPTER)
    report = paths["tmp"] / "discovery.json"
    output = _run(
        paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "table-source-pricing", "--llm-adapter", spec, "--json", str(report)],
        capsys,
    )
    [operation] = _proposal(output, "table-source-pricing")["adapter"]["operations"]
    assert operation["key"]["effective_from"] == "2026-10-20"
    assert operation["price"]["effective_from"] == "2026-10-20"
    assert json.loads(report.read_text()) == output


# --- The report never overwrites an input -----------------------------------------


def _refused(argv, capsys, *inputs):
    before = {path: path.read_bytes() for path in inputs}
    status = discover_prices.main(argv)
    captured = capsys.readouterr()
    assert status == 1
    assert "refusing to write" in captured.err
    assert captured.out == ""
    assert {path: path.read_bytes() for path in inputs} == before


def test_the_report_may_not_overwrite_the_catalog(paths, capsys):
    catalog = paths["catalog"]
    _refused(paths["base"] + ["--content-dir", str(paths["content"]), "--json", str(catalog)], capsys, catalog)


def test_the_report_may_not_overwrite_the_catalog_through_a_symlink(paths, capsys):
    link = paths["tmp"] / "report.json"
    link.symlink_to(paths["catalog"])
    _refused(paths["base"] + ["--content-dir", str(paths["content"]), "--json", str(link)], capsys, paths["catalog"])


def test_the_report_may_not_overwrite_the_registry(paths, capsys):
    registry = paths["tmp"] / "sources.yaml"
    _refused(paths["base"] + ["--content-dir", str(paths["content"]), "--json", str(registry)], capsys, registry)


def test_the_report_may_not_overwrite_the_escalations_it_read(paths, capsys):
    report = _escalation_report(paths["tmp"])
    _refused(
        paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "", "--escalations", str(report), "--json", str(report)],
        capsys,
        report,
    )


def test_the_report_may_not_overwrite_the_new_source_file_through_a_symlink(paths, capsys):
    page = paths["tmp"] / "new-provider.html"
    page.write_text("<table><tr><th>Model</th></tr></table>")
    link = paths["tmp"] / "out.json"
    link.symlink_to(page)
    _refused(
        paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "", "--new-source", str(page), "--json", str(link)],
        capsys,
        page,
    )


def test_the_report_may_not_overwrite_a_content_file_it_read(paths, capsys):
    page = paths["content"] / "table-source-pricing.html"
    _refused(
        paths["base"] + ["--content-dir", str(paths["content"]), "--sources", "table-source-pricing", "--json", str(page)],
        capsys,
        page,
    )


def test_a_report_beside_the_inputs_is_written(paths, capsys):
    report = paths["tmp"] / "discovery.json"
    output = _run(paths["base"] + ["--content-dir", str(paths["content"]), "--json", str(report)], capsys)
    assert json.loads(report.read_text()) == output


def test_an_adapter_failing_at_import_is_an_error_without_its_message(paths, capsys, monkeypatch):
    spec = _adapter(paths["tmp"], monkeypatch, "raise ValueError('quoted prompt text')\n")
    assert discover_prices.main(paths["base"] + ["--content-dir", str(paths["content"]), "--llm-adapter", spec]) == 1
    err = capsys.readouterr().err
    assert "cannot load adapter 'stub_pricing_adapter:propose': ValueError" in err
    assert "quoted prompt text" not in err


def test_an_unwritable_report_path_exits_one_without_a_report(paths, capsys):
    report = paths["tmp"] / "missing-directory" / "discovery.json"
    status = discover_prices.main(paths["base"] + ["--content-dir", str(paths["content"]), "--json", str(report)])
    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert f"discover_prices: cannot write {report}" in captured.err
    assert "Example Model" not in captured.err


def test_an_unloadable_adapter_is_an_error(paths, capsys):
    assert discover_prices.main(paths["base"] + ["--content-dir", str(paths["content"]), "--llm-adapter", "no_such_module:propose"]) == 1
    assert "cannot load adapter" in capsys.readouterr().err


def test_a_close_alias_operation_round_trips_through_the_report_form():
    alias = {
        "provider": "openrouter",
        "alias": "example-host/model-a-instruct",
        "channel": "openrouter-api",
    }
    operation = {
        "op": "close_alias",
        "key": dict(alias, canonical_id="example-host/model-a-instruct"),
        "expect": alias,
        "effective_to": "2026-10-20",
        "source_url": "https://example.com/review",
    }
    delta = discover_prices._delta([operation], ["observation"], [])

    assert discover_prices._operation_json(
        discover_prices.parse_catalog_delta(delta).operations[0]
    ) == operation
