#!/usr/bin/env python3
"""Gate that the built server wheel declares and prices through core.

Acceptance criterion: the server distribution must declare its ``metergraph-core``
dependency and work against the *built* core wheel. This script:

1. builds (or, with ``--no-build``, reuses) the core wheel and the server wheel;
2. asserts the server wheel's ``Requires-Dist`` names ``metergraph-core`` with
   the intended ``>=0.6,<0.9`` range;
3. installs the built core and server wheels into a throwaway virtualenv with
   their normal third-party dependencies resolved and no source tree visible;
   and
4. imports the server pricing adapter, loads the catalog, and reproduces the
   golden price and catalog hash.

Archive members are read in memory only; no untrusted archive path is extracted.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import textwrap
import venv
import zipfile
from email.parser import BytesParser
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[3]
CORE_DIR = REPO_DIR / "core"
SERVER_DIR = REPO_DIR / "server"
CORE_DIST = CORE_DIR / "dist"
SERVER_DIST = SERVER_DIR / "dist"

CORE_DISTRIBUTION = "metergraph-core"
EXPECTED_CORE_SPECIFIERS = {">=0.6", "<0.9"}

GOLDEN_COST = "0.52500000"
GOLDEN_PRICE_ID = "openai/gpt-5.4-mini:openai-api:global:2026-03-17"
CATALOG_VERSION = "2026-10-27"


class VerifyError(AssertionError):
    """Raised when the server artifact violates the release contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def _clean(dist_dir: Path) -> None:
    if not dist_dir.exists():
        return
    for artifact in list(dist_dir.glob("*.whl")) + list(dist_dir.glob("*.tar.gz")):
        artifact.unlink()


def _build_wheel(project: Path, dist_dir: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir), str(project)],
        check=True,
    )


def _single_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("*.whl"))
    _require(
        len(wheels) == 1,
        f"expected exactly one wheel in {dist_dir}, found {[w.name for w in wheels]}",
    )
    return wheels[0]


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _requires_dist(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        _require(
            len(metadata_names) == 1,
            f"{wheel.name}: expected one METADATA, found {metadata_names}",
        )
        message = BytesParser().parsebytes(archive.read(metadata_names[0]))
    return message.get_all("Requires-Dist") or []


def _parse_requirement(requirement: str) -> tuple[str, set[str]]:
    """Split a ``Requires-Dist`` value into (normalized name, specifier set).

    Drops any environment marker and extras, then separates the project name
    from its version specifiers without depending on the ``packaging`` library.
    """

    requirement = requirement.split(";", 1)[0].strip()
    index = len(requirement)
    for char in "<>=!~ ([":
        position = requirement.find(char)
        if position != -1:
            index = min(index, position)
    name = _normalize(requirement[:index])
    specifier = requirement[index:].strip()
    specifiers = {token.strip() for token in specifier.split(",") if token.strip()}
    return name, specifiers


def _verify_server_requires_core(wheel: Path) -> None:
    matches = [
        _parse_requirement(requirement)
        for requirement in _requires_dist(wheel)
        if _parse_requirement(requirement)[0] == CORE_DISTRIBUTION
    ]
    _require(
        len(matches) == 1,
        f"{wheel.name}: expected exactly one metergraph-core Requires-Dist, "
        f"found {[m for m in matches]}",
    )
    specifiers = matches[0][1]
    _require(
        specifiers == EXPECTED_CORE_SPECIFIERS,
        f"{wheel.name}: metergraph-core range {sorted(specifiers)} != "
        f"{sorted(EXPECTED_CORE_SPECIFIERS)}",
    )


def _venv_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _verify_isolated_install(core_wheel: Path, server_wheel: Path) -> None:
    check = textwrap.dedent(
        f"""
        from datetime import datetime, timezone

        import metergraph_server
        from metergraph_server import prices

        source = getattr(metergraph_server, "__file__", "")
        assert "site-packages" in source, source

        version, document, snapshot = prices.load()
        assert version == {CATALOG_VERSION!r}, version
        assert document["models"], "catalog document has no models"

        loaded = prices.load_identity()
        assert loaded.version == {CATALOG_VERSION!r}, loaded.version
        assert len(loaded.content_hash) == 64, loaded.content_hash

        result = snapshot.cost(
            provider="openai",
            model="gpt-5.4-mini",
            at=datetime(2026, 8, 17, tzinfo=timezone.utc),
            input_tokens=100_000,
            output_tokens=100_000,
        )
        assert result.price_id == {GOLDEN_PRICE_ID!r}, result.price_id
        assert str(result.cost_usd) == {GOLDEN_COST!r}, result.cost_usd
        assert result.status == "priced", result.status
        print("SERVER-GOLDEN", result.cost_usd, loaded.content_hash[:12])
        """
    )
    # Strip PYTHONPATH so an ambient source-tree entry cannot satisfy the import
    # instead of the installed wheels.
    child_env = os.environ.copy()
    child_env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory() as tmp:
        env_dir = Path(tmp) / "venv"
        venv.create(env_dir, with_pip=True)
        python = _venv_python(env_dir)
        # Install the built core and server wheels together. The local core
        # wheel satisfies the server's metergraph-core requirement; fastapi,
        # uvicorn, psycopg, and pyyaml resolve normally from the index.
        subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", str(core_wheel), str(server_wheel)],
            check=True,
            env=child_env,
        )
        completed = subprocess.run(
            [str(python), "-c", check],
            cwd=tmp,
            check=True,
            capture_output=True,
            text=True,
            env=child_env,
        )
        _require(
            f"SERVER-GOLDEN {GOLDEN_COST}" in completed.stdout,
            f"isolated server install did not reproduce golden price: "
            f"{completed.stdout!r}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="reuse prebuilt wheels in core/dist and server/dist without rebuilding",
    )
    args = parser.parse_args()

    if args.no_build:
        _require(CORE_DIST.is_dir(), f"--no-build requires prebuilt artifacts in {CORE_DIST}")
        _require(
            SERVER_DIST.is_dir(), f"--no-build requires prebuilt artifacts in {SERVER_DIST}"
        )
    else:
        _clean(CORE_DIST)
        _clean(SERVER_DIST)
        _build_wheel(CORE_DIR, CORE_DIST)
        _build_wheel(SERVER_DIR, SERVER_DIST)

    core_wheel = _single_wheel(CORE_DIST)
    server_wheel = _single_wheel(SERVER_DIST)

    _verify_server_requires_core(server_wheel)
    _verify_isolated_install(core_wheel, server_wheel)

    print(f"OK: {server_wheel.name} declares core and prices through {core_wheel.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
