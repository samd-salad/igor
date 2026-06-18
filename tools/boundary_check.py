"""Static check that bounded contexts don't import each other's internals
and that only `external/` adapters import third-party libraries.

Run from repo root: python -m tools.boundary_check
"""
from __future__ import annotations
import ast
import sys
from pathlib import Path

COGNITION_FORBIDDEN = {
    "server.external",
    "server.ha_io",
    "anthropic",
    "requests",
    "sqlite3",
    "fastapi",
}
HA_IO_FORBIDDEN = {
    "server.external",
    "server.cognition.ports",
    "server.cognition.aggregates",
    "server.cognition.services",
    "server.cognition._internal",
    "anthropic",
    "requests",
    "sqlite3",
}
THIRD_PARTY_LOCKED_TO_EXTERNAL = {
    "anthropic": ("server/external/claude_adapter.py",),
    "sqlite3":   ("server/external/sqlite_persistence.py",
                  "server/external/sqlite_retrieval.py",
                  "server/external/_internal/db.py",
                  "server/external/_internal/brain_json_migration.py"),
}


def _iter_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.append(node.module)
    return out


def _violations_under(root: Path, package_prefix: str, forbidden: set[str]) -> list[str]:
    bad = []
    for py in root.rglob("*.py"):
        rel_name = ".".join(py.with_suffix("").relative_to(root.parent).parts)
        if not rel_name.startswith(package_prefix):
            continue
        for imp in _iter_imports(py):
            for f in forbidden:
                if imp == f or imp.startswith(f + "."):
                    bad.append(f"{py}: imports forbidden {imp!r}")
    return bad


def _third_party_leaks(root: Path) -> list[str]:
    bad = []
    for lib, allowed_paths in THIRD_PARTY_LOCKED_TO_EXTERNAL.items():
        allowed_set = set(allowed_paths)
        for py in root.rglob("*.py"):
            rel = py.relative_to(root.parent).as_posix()
            if rel in allowed_set:
                continue
            # Skip files outside our managed contexts (legacy api.py during migration)
            if not (rel.startswith("server/cognition/")
                    or rel.startswith("server/ha_io/")
                    or rel.startswith("server/external/")):
                continue
            for imp in _iter_imports(py):
                if imp == lib or imp.startswith(lib + "."):
                    bad.append(f"{py}: imports {imp!r} but only allowed in {sorted(allowed_set)}")
    return bad


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    server_root = repo_root / "server"
    all_violations: list[str] = []
    all_violations += _violations_under(server_root, "server.cognition", COGNITION_FORBIDDEN)
    all_violations += _violations_under(server_root, "server.ha_io", HA_IO_FORBIDDEN)
    all_violations += _third_party_leaks(server_root)
    if all_violations:
        for v in all_violations:
            print(f"BOUNDARY VIOLATION: {v}", file=sys.stderr)
        return 1
    print("Boundary check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
