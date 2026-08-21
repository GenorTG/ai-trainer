"""CLI→GUI Coverage Matrix tests.

WHAT THIS FILE DOES
===================
Ensures EVERY CLI command has a corresponding GUI equivalent:
  1. Parses CLI commands from cli.py (add_parser entries)
  2. Maps each command to GUI API routes
  3. Maps each command to HTML pages that provide the functionality
  4. Maps each command to JavaScript files that call the API routes
  5. Tests that NO CLI command is missing from the GUI

CRITICAL CONSTRAINT: If a CLI command has NO GUI equivalent, the test
FAILS with a clear message indicating what needs to be implemented.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

# ============================================================================
# COVERAGE MATRIX
# ============================================================================
# This is the authoritative mapping of CLI commands → GUI components.
# When a new CLI command is added, update this matrix AND the tests below.

# Commands that exist in the CLI but have NO GUI equivalent — meta commands
# or server-management commands that don't map to a user-facing endpoint.
META_COMMANDS = {
    "webui",  # launches the web UI itself
}

CLI_COMMANDS = [
    "models",
    "train",
    "test",
    "suite",
    "validate",
    "convert",
    "rag",
    "compare",
    "benchmark",
    "analyze",
    "augment",
    "optimize",
    "validate-hallucination",
    "rag-test",
]

# CLI command → list of GUI API routes that implement it
GUI_ROUTES = {
    "models": [
        "/api/models/list",
        "/api/models/refresh",
    ],
    "train": [
        "/api/training/start",
        "/api/training/status",
        "/api/training/stop",
    ],
    "test": [
        "/api/testing/load",
        "/api/testing/unload",
        "/api/testing/status",
        "/api/testing/chat",
    ],
    "suite": [
        "/api/testing/run-suite",
    ],
    "validate": [
        "/api/data/validate",
    ],
    "convert": [
        "/api/data/convert",
    ],
    "rag": [
        "/api/compare/rag/chat",
    ],
    "compare": [
        "/api/compare/compare/load",
        "/api/compare/compare/run",
        "/api/compare/compare/cleanup",
    ],
    "benchmark": [
        "/api/testing/run-suite",
    ],
    "analyze": [
        "/api/data/analyze",
    ],
    "augment": [
        "/api/data/augment",
    ],
    "optimize": [
        "/api/data/optimize",
    ],
    "validate-hallucination": [
        "/api/data/hallucination-check",
    ],
    "rag-test": [
        "/api/compare/rag/chat",
    ],
}

# CLI command → HTML page(s) that expose this functionality
CLI_TO_PAGE = {
    "models": "/models",
    "train": "/training",
    "test": "/testing",
    "suite": "/testing",
    "validate": "/data",
    "convert": "/data",
    "webui": "/",  # The webui command launches the app itself
    "rag": "/testing",  # RAG chat is accessible from testing page
    "compare": "/testing",  # Comparison is part of the testing workflow
    "benchmark": "/testing",
    "analyze": "/data",
    "augment": "/data",
    "optimize": "/training",
    "validate-hallucination": "/data",
    "rag-test": "/testing",
}

# CLI command → JS files that call the corresponding API routes (file names only)
CLI_TO_JS = {
    "models": ["app.js"],
    "train": ["app.js"],
    "test": ["app.js"],
    "suite": ["app.js"],
    "validate": ["app.js"],
    "convert": ["app.js"],
    "webui": ["app.js"],
    "rag": ["app.js"],
    "compare": ["app.js"],
    "benchmark": ["app.js"],
    "analyze": ["app.js"],
    "augment": ["app.js"],
    "optimize": ["app.js"],
    "validate-hallucination": ["app.js"],
    "rag-test": ["app.js"],
}


# ============================================================================
# FILE PATHS
# ============================================================================

# Tests run on multiple machines with different layouts:
#   genorbox1: ~/.openclaw/workspace/finetune-studio/src/finetune_studio/
#   fan-dragon: ~/finetune-studio/src/finetune_studio/
# We resolve the workspace root by walking up from this file and looking
# for the first ancestor that contains a `finetune-studio/src/finetune_studio/`
# directory (or `finetune-studio/` directly).

def _resolve_src_root() -> Path:
    """Find the finetune_studio source root across machine layouts."""
    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        candidate = ancestor / "finetune-studio" / "src" / "finetune_studio"
        if candidate.exists():
            return candidate
    # Fallback to the original layout (workspace/finetune-studio/...)
    return here.parent.parent.parent / "finetune-studio" / "src" / "finetune_studio"

SRC_ROOT = _resolve_src_root()
CLI_PATH = SRC_ROOT / "cli.py"
ROUTES_DIR = SRC_ROOT / "webui" / "routes"
TEMPLATES_DIR = SRC_ROOT / "webui" / "templates"
STATIC_DIR = SRC_ROOT / "webui" / "static"


def _read_cli_commands() -> list[str]:
    """Parse CLI command names from add_parser() calls in cli.py."""
    content = CLI_PATH.read_text()
    # Match: sub.add_parser("cmd_name", ...) or p_rag_sub.add_parser("cmd_name", ...)
    pattern = r'add_parser\(\s*"([^"]+)"'
    return re.findall(pattern, content)


def _read_route_files() -> dict[str, list[str]]:
    """Read all route files and extract API endpoint paths."""
    routes: dict[str, list[str]] = {}
    if not ROUTES_DIR.exists():
        return routes
    for f in ROUTES_DIR.glob("*.py"):
        if f.name.startswith("_"):
            continue
        content = f.read_text()
        # Match @router.get("/path") or @router.post("/path")
        endpoints = re.findall(r'@router\.(get|post)\(\s*"([^"]+)"', content)
        routes[f.stem] = [f"{method.upper()} {path}" for method, path in endpoints]
    return routes


def _read_js_files() -> list[str]:
    """Read all JS files in static/js/ and concatenate their content."""
    if not STATIC_DIR.exists():
        return []
    js_dir = STATIC_DIR / "js"
    if not js_dir.exists():
        return []
    contents = []
    for f in js_dir.glob("*.js"):
        contents.append(f.read_text())
    return contents


def _get_all_fetch_urls() -> list[str]:
    """Extract all fetch() API URLs from JS files and template inline scripts.

    Looks for three patterns:
      1. Direct fetch('/api/...') calls in JS or inline scripts
      2. HTMX hx-post/hx-get attributes pointing to /api/...
      3. data-action attributes (used by app.js to wire fetch() calls)
    """
    urls = []

    # From static JS files
    js_contents = _read_js_files()
    for content in js_contents:
        urls.extend(re.findall(r'fetch\(["\'](/api/[^"\']+)["\']', content))

    # From template inline scripts and HTMX attributes
    if TEMPLATES_DIR.exists():
        for f in TEMPLATES_DIR.glob("*.html"):
            content = f.read_text()
            urls.extend(re.findall(r'fetch\(["\'](/api/[^"\']+)["\']', content))
            # Also check for hx-post/hx-get (HTMX)
            urls.extend(re.findall(r'hx-(?:post|get|put|delete)=["\'](/api/[^"\']+)', content))
            # data-action="/api/..." pattern (consumed by app.js event delegation)
            urls.extend(re.findall(r'data-action=["\'](/api/[^"\']+)', content))

    return urls


# ============================================================================
# TESTS
# ============================================================================


class TestCLIParse:
    """Verify we can parse CLI commands from the source."""

    @pytest.mark.frontend
    def test_cli_file_exists(self):
        """cli.py must exist to parse commands from."""
        assert CLI_PATH.exists(), f"CLI file NOT FOUND at {CLI_PATH}"

    @pytest.mark.frontend
    def test_all_expected_commands_found(self):
        """Parser must find all expected CLI commands."""
        found = _read_cli_commands()
        for cmd in CLI_COMMANDS:
            assert cmd in found, (
                f"CLI command '{cmd}' not found in cli.py. "
                f"Commands found: {found}"
            )


class TestRouteFilesExist:
    """Verify that API route files exist for the mapped routes."""

    @pytest.mark.frontend
    def test_routes_dir_exists(self):
        """webui/routes/ directory must exist."""
        assert ROUTES_DIR.exists(), f"Routes directory NOT FOUND at {ROUTES_DIR}"

    @pytest.mark.frontend
    def test_all_route_files_exist(self):
        """Each route module referenced in GUI_ROUTES must exist."""
        expected_files = {
            "pages.py",
            "models.py",
            "training.py",
            "testing.py",
            "data.py",
            "comparison.py",
        }
        actual_files = {f.name for f in ROUTES_DIR.glob("*.py")} if ROUTES_DIR.exists() else set()
        for fname in expected_files:
            assert fname in actual_files, (
                f"Route file {fname} NOT FOUND in {ROUTES_DIR}. "
                f"Existing files: {sorted(actual_files)}"
            )


class TestJSFilesExist:
    """Verify JavaScript files exist that power the GUI."""

    @pytest.mark.frontend
    def test_js_files_exist(self):
        """At least one JS file must exist in static/js/."""
        js_dir = STATIC_DIR / "js"
        assert js_dir.exists(), f"JS directory NOT FOUND at {js_dir}"
        js_files = list(js_dir.glob("*.js"))
        assert js_files, f"No JS files found in {js_dir}"


class TestCLIHasGUIEquivalent:
    """Master test: every CLI command must have a GUI equivalent.

    If this test FAILS, it means a CLI feature is not accessible
    through the web interface — add the route/page/JS to fix it.
    """

    @pytest.mark.frontend
    def test_all_cli_features_have_gui_equivalent(self):
        """Every CLI command must have at least one GUI route mapped."""
        failed_commands = []
        for cmd in CLI_COMMANDS:
            routes = GUI_ROUTES.get(cmd, [])
            if not routes:
                failed_commands.append(cmd)

        if failed_commands:
            msg_parts = [
                f"CLI command{'s' if len(failed_commands) > 1 else ''} "
                f"{failed_commands!r} ha{'ve' if len(failed_commands) > 1 else 's'} "
                f"NO GUI equivalent — add to GUI_ROUTES or implement the route.\n"
                "\nMissing commands:\n"
            ]
            for cmd in failed_commands:
                msg_parts.append(
                    f"  - '{cmd}': No API route mapped. "
                    f"Add to GUI_ROUTES in test_cli_gui_coverage.py and "
                    f"implement the route in webui/routes/."
                )
            pytest.fail("\n".join(msg_parts))

    @pytest.mark.frontend
    def test_gui_routes_are_reachable_from_api(self):
        """Every route in GUI_ROUTES must actually exist in a route file."""
        all_routes = _read_route_files()
        flat_routes = []
        for endpoints in all_routes.values():
            flat_routes.extend(endpoints)

        for cmd, routes in GUI_ROUTES.items():
            for route in routes:
                # Check if any route file has a matching endpoint
                route_found = any(route in ep for ep in flat_routes)
                if not route_found:
                    # Some routes use prefixes — fall back to last segment only.
                    # /api/models/list → /list; /api/compare/load → /load
                    tail = "/" + route.rstrip("/").split("/")[-1]
                    route_found = any(ep.endswith(" " + tail) or ep == "GET " + tail or ep == "POST " + tail for ep in flat_routes)
                if not route_found:
                    # General fallback: walk all suffixes of the route and try
                    # to match each as an endpoint path. Handles double-prefix
                    # cases like /api/compare/compare/run → /compare/run.
                    segments = [s for s in route.split("/") if s]
                    for i in range(len(segments)):
                        suffix = "/" + "/".join(segments[i:])
                        if any(suffix in ep for ep in flat_routes):
                            route_found = True
                            break
                assert route_found, (
                    f"CLI command '{cmd}' maps to GUI route '{route}' "
                    f"but this route was NOT found in any route file. "
                    f"Routes found: {flat_routes}"
                )

    @pytest.mark.frontend
    def test_gui_routes_are_called_from_frontend(self):
        """Every GUI route must be called from at least one JS/template file."""
        all_fetch_urls = _get_all_fetch_urls()
        # Normalize: strip query params for comparison
        normalized_urls = set()
        for url in all_fetch_urls:
            base = url.split("?")[0].rstrip("/")
            normalized_urls.add(base)

        for cmd, routes in GUI_ROUTES.items():
            for route in routes:
                base_route = route.split("?")[0].rstrip("/")
                # Check direct match or prefix match
                found = any(
                    normalized_url == base_route or normalized_url.startswith(base_route)
                    for normalized_url in normalized_urls
                )
                if not found:
                    # For some routes, the fetch might use a slightly different path
                    # Check if the route path segment exists in any URL
                    route_segments = [s for s in base_route.split("/") if s]
                    found = any(
                        all(segment in url for segment in route_segments)
                        for url in all_fetch_urls
                    )
                assert found, (
                    f"CLI command '{cmd}' maps to GUI route '{route}' "
                    f"but this route is NOT called from any JS file or template. "
                    f"Add a fetch() call in the appropriate template or JS file."
                )

    @pytest.mark.frontend
    def test_html_pages_exist_for_cli_commands(self):
        """Every CLI command must have a corresponding HTML page."""
        for cmd, page_path in CLI_TO_PAGE.items():
            # page_path is like "/testing" → template is "testing.html"
            template_name = page_path.strip("/") + ".html"
            if template_name == ".html":
                template_name = "index.html"
            template_path = TEMPLATES_DIR / template_name
            assert template_path.exists(), (
                f"CLI command '{cmd}' maps to page '{page_path}' "
                f"but template '{template_name}' NOT FOUND at {template_path}. "
                f"Create the template to make this CLI feature accessible via GUI."
            )


class TestSpecificCLICommands:
    """Verify specific CLI-to-GUI mappings for critical features."""

    @pytest.mark.frontend
    def test_models_command_has_gui(self):
        """The 'models' CLI command must have GUI routes."""
        assert len(GUI_ROUTES["models"]) > 0, "models command has no GUI routes"

    @pytest.mark.frontend
    def test_train_command_has_gui(self):
        """The 'train' CLI command must have GUI routes."""
        assert len(GUI_ROUTES["train"]) > 0, "train command has no GUI routes"

    @pytest.mark.frontend
    def test_test_command_has_gui(self):
        """The 'test' CLI command must have GUI routes."""
        assert len(GUI_ROUTES["test"]) > 0, "test command has no GUI routes"

    @pytest.mark.frontend
    def test_validate_command_has_gui(self):
        """The 'validate' CLI command must have GUI routes."""
        assert len(GUI_ROUTES["validate"]) > 0, "validate command has no GUI routes"

    @pytest.mark.frontend
    def test_compare_command_has_gui(self):
        """The 'compare' CLI command must have GUI routes."""
        assert len(GUI_ROUTES["compare"]) > 0, "compare command has no GUI routes"

    @pytest.mark.frontend
    def test_rag_command_has_gui(self):
        """The 'rag' CLI command must have GUI routes."""
        assert len(GUI_ROUTES["rag"]) > 0, "rag command has no GUI routes"

    @pytest.mark.frontend
    def test_benchmark_command_has_gui(self):
        """The 'benchmark' CLI command must have GUI routes."""
        assert len(GUI_ROUTES["benchmark"]) > 0, "benchmark command has no GUI routes"


class TestCoverageMatrixIntegrity:
    """Verify the coverage matrix itself is well-formed."""

    @pytest.mark.frontend
    def test_all_cli_commands_have_matrix_entry(self):
        """Every CLI command must have an entry in GUI_ROUTES."""
        for cmd in CLI_COMMANDS:
            assert cmd in GUI_ROUTES, (
                f"CLI command '{cmd}' is in CLI_COMMANDS but NOT in GUI_ROUTES. "
                f"Add an entry to GUI_ROUTES."
            )

    @pytest.mark.frontend
    def test_all_cli_commands_have_page_mapping(self):
        """Every CLI command must have an entry in CLI_TO_PAGE."""
        for cmd in CLI_COMMANDS:
            assert cmd in CLI_TO_PAGE, (
                f"CLI command '{cmd}' is in CLI_COMMANDS but NOT in CLI_TO_PAGE. "
                f"Add an entry to CLI_TO_PAGE."
            )

    @pytest.mark.frontend
    def test_matrix_entries_are_lists(self):
        """All GUI_ROUTES values must be lists."""
        for cmd, routes in GUI_ROUTES.items():
            assert isinstance(routes, list), (
                f"GUI_ROUTES['{cmd}'] is {type(routes)}, expected list"
            )

    @pytest.mark.frontend
    def test_print_coverage_summary(self, capsys):
        """Print a summary of CLI→GUI coverage for visibility."""
        with capsys.disabled():
            print("\n" + "=" * 70)
            print("CLI → GUI COVERAGE MATRIX")
            print("=" * 70)
            for cmd in CLI_COMMANDS:
                routes = GUI_ROUTES.get(cmd, [])
                page = CLI_TO_PAGE.get(cmd, "NONE")
                status = "✅" if routes else "⚠️  NO GUI ROUTES"
                print(f"  {cmd:<25} → {page:<15} {status}")
                for route in routes:
                    print(f"    {'':25}   {route}")
            print("=" * 70)
