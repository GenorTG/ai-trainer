"""Static asset and JavaScript interactivity tests.

WHAT THIS FILE DOES
===================
Verifies the frontend static asset layer:
  1. static/ directory exists under webui
  2. CSS file(s) exist
  3. JS file(s) exist
  4. Each JS file has at least one event listener (page is interactive)
  5. Each HTML template includes <script> tags (local or remote)

Uses filesystem checks (no mock needed).
"""

from __future__ import annotations

from pathlib import Path

import re

import pytest

# Paths relative to this test file
# Resolves layout differences between machines (genorbox1 vs fan-dragon).
def _resolve_layout(*sub: str) -> Path:
    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        candidate = ancestor / "finetune-studio" / "src" / "finetune_studio" / Path(*sub)
        if candidate.exists():
            return candidate
    return here.parent.parent.parent / "finetune-studio" / "src" / "finetune_studio" / Path(*sub)

STATIC_DIR = _resolve_layout("webui", "static")
TEMPLATES_DIR = _resolve_layout("webui", "templates")

# Event-related patterns that indicate interactivity in JS
INTERACTIVITY_PATTERNS = [
    r"addEventListener",
    r"onclick",
    r"onsubmit",
    r"onchange",
    r"onkeydown",
    r"onkeyup",
    r"\.on\(",
    r"click\s*\(",
    r"fetch\(",
    r"XMLHttpRequest",
    r"\.submit\(\)",
    r"\.addEventListener\(",
    r"document\.getElementById",
    r"document\.querySelector",
]


class TestStaticDirectory:
    """Verify static asset directory structure exists."""

    @pytest.mark.frontend
    def test_static_dir_exists(self):
        """The static/ directory must exist under webui."""
        assert STATIC_DIR.exists(), (
            f"Static directory NOT FOUND at {STATIC_DIR}. "
            f"Create it in finetune-studio/src/finetune_studio/webui/static/"
        )

    @pytest.mark.frontend
    def test_static_css_dir_exists(self):
        """static/css/ subdirectory must exist."""
        css_dir = STATIC_DIR / "css"
        assert css_dir.exists(), f"CSS directory NOT FOUND at {css_dir}"

    @pytest.mark.frontend
    def test_static_js_dir_exists(self):
        """static/js/ subdirectory must exist."""
        js_dir = STATIC_DIR / "js"
        assert js_dir.exists(), f"JS directory NOT FOUND at {js_dir}"


class TestCSSFiles:
    """Verify CSS files exist in the static directory."""

    @pytest.mark.frontend
    def test_css_file_exists(self):
        """At least one .css file must exist in static/css/."""
        css_dir = STATIC_DIR / "css"
        css_files = list(css_dir.glob("*.css")) if css_dir.exists() else []
        assert css_files, (
            f"No .css files found in {css_dir}. "
            f"Create at least one CSS file for styling."
        )

    @pytest.mark.frontend
    def test_css_file_not_empty(self):
        """Every CSS file must be non-empty."""
        css_dir = STATIC_DIR / "css"
        if not css_dir.exists():
            pytest.skip("CSS directory does not exist")
        css_files = list(css_dir.glob("*.css"))
        for f in css_files:
            content = f.read_text()
            assert len(content.strip()) > 0, f"CSS file {f.name} is empty"


class TestJSFiles:
    """Verify JS files exist and are interactive."""

    @pytest.mark.frontend
    def test_js_file_exists(self):
        """At least one .js file must exist in static/js/."""
        js_dir = STATIC_DIR / "js"
        js_files = list(js_dir.glob("*.js")) if js_dir.exists() else []
        assert js_files, (
            f"No .js files found in {js_dir}. "
            f"Create at least one JS file for interactivity."
        )

    @pytest.mark.frontend
    def test_js_file_not_empty(self):
        """Every JS file must be non-empty."""
        js_dir = STATIC_DIR / "js"
        if not js_dir.exists():
            pytest.skip("JS directory does not exist")
        js_files = list(js_dir.glob("*.js"))
        for f in js_files:
            content = f.read_text()
            assert len(content.strip()) > 0, f"JS file {f.name} is empty"

    @pytest.mark.frontend
    def test_js_files_have_event_listeners(self):
        """Each .js file must contain at least one event listener or DOM interaction."""
        js_dir = STATIC_DIR / "js"
        if not js_dir.exists():
            pytest.skip("JS directory does not exist")
        js_files = list(js_dir.glob("*.js"))
        for f in js_files:
            content = f.read_text()
            has_interaction = any(
                re.search(pattern, content) for pattern in INTERACTIVITY_PATTERNS
            )
            assert has_interaction, (
                f"JS file {f.name} has NO event listeners or DOM interactions. "
                f"It only contains: {content[:200]}"
            )


class TestHTMLTemplatesIncludeScripts:
    """Verify that each HTML template includes script references."""

    @pytest.mark.frontend
    def test_all_templates_have_scripts(self):
        """Every HTML template must include <script> tags or have inline scripts."""
        if not TEMPLATES_DIR.exists():
            pytest.skip("Templates directory does not exist")
        html_files = list(TEMPLATES_DIR.glob("*.html"))
        assert html_files, "No HTML templates found"
        for f in html_files:
            content = f.read_text().lower()
            has_script_tag = "<script" in content
            has_inline_event = any(
                kw in content
                for kw in ["onclick", "onsubmit", "onchange", "onkeydown", "hx-post", "hx-get"]
            )
            assert has_script_tag or has_inline_event, (
                f"Template {f.name} has NO <script> tags and NO inline event handlers. "
                f"The page will not be interactive."
            )

    @pytest.mark.frontend
    def test_base_template_loads_app_js(self):
        """base.html must include the app.js script file."""
        base_path = TEMPLATES_DIR / "base.html"
        if not base_path.exists():
            pytest.skip("base.html not found")
        content = base_path.read_text()
        assert "app.js" in content, (
            "base.html does not reference app.js. "
            "Add <script src=\"/static/js/app.js\"></script> to base.html."
        )

    @pytest.mark.frontend
    def test_base_template_loads_htmx(self):
        """base.html must include HTMX for dynamic content loading."""
        base_path = TEMPLATES_DIR / "base.html"
        if not base_path.exists():
            pytest.skip("base.html not found")
        content = base_path.read_text()
        assert "htmx" in content.lower(), (
            "base.html does not reference HTMX. "
            "HTMX is needed for dynamic content loading in the WebUI."
        )

    @pytest.mark.frontend
    def test_testing_template_has_inline_scripts(self):
        """testing.html must have inline <script> with fetch() calls to API."""
        template_path = TEMPLATES_DIR / "testing.html"
        if not template_path.exists():
            pytest.skip("testing.html not found")
        content = template_path.read_text()
        assert "fetch(" in content, (
            "testing.html has no fetch() calls. "
            "The testing page needs JavaScript to call API endpoints."
        )

    @pytest.mark.frontend
    def test_training_template_has_inline_scripts(self):
        """training.html must have inline <script> with fetch() calls to API."""
        template_path = TEMPLATES_DIR / "training.html"
        if not template_path.exists():
            pytest.skip("training.html not found")
        content = template_path.read_text()
        assert "fetch(" in content, (
            "training.html has no fetch() calls. "
            "The training page needs JavaScript to call API endpoints."
        )

    @pytest.mark.frontend
    def test_data_template_has_inline_scripts(self):
        """data.html must have inline <script> with fetch() calls to API."""
        template_path = TEMPLATES_DIR / "data.html"
        if not template_path.exists():
            pytest.skip("data.html not found")
        content = template_path.read_text()
        assert "fetch(" in content, (
            "data.html has no fetch() calls. "
            "The data page needs JavaScript to call API endpoints."
        )
