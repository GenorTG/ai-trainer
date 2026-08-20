#!/usr/bin/env python3
"""Bump version in trainer/pyproject.toml and server/pyproject.toml.

Usage: python tools/bump.py [patch|minor|major]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VERSION_RE = re.compile(r'version\s*=\s*"(\d+)\.(\d+)\.(\d+)"')


def bump(version: str, level: str) -> str:
    """Bump a semver version string at the given level."""
    major, minor, patch = (int(x) for x in version.split("."))
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    if level == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Unknown level: {level}")


def update_file(path: Path, level: str) -> tuple[str, str]:
    """Update version in a pyproject.toml file. Returns (old, new)."""
    content = path.read_text()
    match = VERSION_RE.search(content)
    if not match:
        raise ValueError(f"No version found in {path}")
    old = match.group(0).split('"')[1]
    new = bump(old, level)
    new_content = content.replace(match.group(0), match.group(0).replace(old, new))
    path.write_text(new_content)
    return old, new


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("patch", "minor", "major"):
        print(f"Usage: {sys.argv[0]} [patch|minor|major]")
        return 1
    level = sys.argv[1]

    files = [
        ROOT / "trainer" / "pyproject.toml",
        ROOT / "server" / "pyproject.toml",
    ]
    for path in files:
        if path.exists():
            old, new = update_file(path, level)
            print(f"  {path.relative_to(ROOT)}: {old} → {new}")

    # Also update CHANGELOG.md
    changelog = ROOT / "CHANGELOG.md"
    if changelog.exists():
        print(f"\nDon't forget to update CHANGELOG.md with v{new} entry!")

    return 0


if __name__ == "__main__":
    sys.exit(main())