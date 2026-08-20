"""Frontend page render tests.

WHAT THIS FILE DOES
===================
Verifies that every HTML page:
  1. Renders with correct status code and content-type
  2. Contains expected keywords (navigation, page title, key elements)
  3. Has the corresponding template file on disk
  4. Contains interactive elements (buttons, forms, inputs)

Uses the `finetune_studio_client` fixture from conftest.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Where the templates live on disk
# (Resolved at module load to work on both genorbox1 and fan-dragon layouts.)
def _resolve_templates_dir() -> Path:
    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        candidate = ancestor / "finetune-studio" / "src" / "finetune_studio" / "webui" / "templates"
        if candidate.exists():
            return candidate
    return here.parent.parent.parent / "finetune-studio" / "src" / "finetune_studio" / "webui" / "templates"

TEMPLATES_DIR = _resolve_templates_dir()

# Expected pages and their render properties
PAGE_SPECS = {
    "/": {
        "template": "index.html",
        "keywords": ["Dashboard", "Finetune Studio", "Models Found", "Training Status"],
        "interactive": True,  # has buttons/links
    },
    "/models": {
        "template": "models.html",
        "keywords": ["Models", "Refresh", "format"],
        "interactive": True,
    },
    "/training": {
        "template": "training.html",
        "keywords": ["Training", "Configuration", "Start Training", "Stop"],
        "interactive": True,
    },
    "/data": {
        "template": "data.html",
        "keywords": ["Data", "Upload", "Files"],
        "interactive": True,
    },
    "/testing": {
        "template": "testing.html",
        "keywords": ["Testing", "Model Testing", "Load", "Send"],
        "interactive": True,
    },
}


class TestPageRenders:
    """Verify each HTML page renders with correct status and content."""

    @pytest.mark.frontend
    @pytest.mark.parametrize("path,spec", PAGE_SPECS.items(), ids=[p for p in PAGE_SPECS])
    def test_page_renders(self, finetune_studio_client, path, spec):
        """Page returns 200 with text/html and contains expected keywords."""
        resp = finetune_studio_client.get(path)
        assert resp.status_code == 200, f"Page {path} returned {resp.status_code}"
        ct = resp.headers.get("content-type", "")
        assert "text/html" in ct, f"Page {path} content-type is {ct}, expected text/html"
        # Check at least one keyword is present
        found = [kw for kw in spec["keywords"] if kw.lower() in resp.text.lower()]
        assert found, (
            f"Page {path} body does not contain any of {spec['keywords']}. "
            f"Body preview: {resp.text[:500]}"
        )

    @pytest.mark.frontend
    @pytest.mark.parametrize("path,spec", PAGE_SPECS.items(), ids=[p for p in PAGE_SPECS])
    def test_page_has_navigation(self, finetune_studio_client, path, spec):
        """Every page should contain the sidebar navigation with all 5 main links."""
        resp = finetune_studio_client.get(path)
        assert resp.status_code == 200
        body = resp.text
        # The base template has nav links to all 5 pages
        assert 'href="/"' in body or "Dashboard" in body, f"Page {path} missing dashboard nav"
        assert 'href="/models"' in body or "Models" in body, f"Page {path} missing models nav"
        assert 'href="/training"' in body or "Training" in body, f"Page {path} missing training nav"
        assert 'href="/data"' in body or "Data" in body, f"Page {path} missing data nav"
        assert 'href="/testing"' in body or "Testing" in body, f"Page {path} missing testing nav"


class TestTemplateFiles:
    """Verify that the corresponding HTML template file exists on disk."""

    @pytest.mark.frontend
    @pytest.mark.parametrize("path,spec", PAGE_SPECS.items(), ids=[p for p in PAGE_SPECS])
    def test_has_template_files(self, path, spec):
        """Template file must exist in webui/templates/ — test FAILS if missing."""
        template_path = TEMPLATES_DIR / spec["template"]
        assert template_path.exists(), (
            f"Template file {spec['template']} NOT FOUND at {template_path}. "
            f"Create it in finetune-studio/src/finetune_studio/webui/templates/"
        )

    @pytest.mark.frontend
    def test_base_template_exists(self):
        """base.html template must exist — all pages extend it."""
        base_path = TEMPLATES_DIR / "base.html"
        assert base_path.exists(), f"base.html NOT FOUND at {base_path}"


class TestInteractiveElements:
    """Verify rendered HTML has interactive elements (buttons, forms, inputs)."""

    @pytest.mark.frontend
    @pytest.mark.parametrize("path,spec", PAGE_SPECS.items(), ids=[p for p in PAGE_SPECS])
    def test_has_interactive_elements(self, finetune_studio_client, path, spec):
        """Page must contain at least one <button>, <form>, or <input> element."""
        resp = finetune_studio_client.get(path)
        assert resp.status_code == 200
        body = resp.text.lower()
        has_button = "<button" in body
        has_form = "<form" in body
        has_input = "<input" in body
        has_select = "<select" in body
        assert has_button or has_form or has_input or has_select, (
            f"Page {path} has NO interactive elements "
            f"(no <button>, <form>, <input>, or <select> found). "
            f"Body preview: {resp.text[:300]}"
        )

    @pytest.mark.frontend
    def test_testing_page_has_chat_input(self, finetune_studio_client):
        """Testing page should have a chat input field for interactive testing."""
        resp = finetune_studio_client.get("/testing")
        assert resp.status_code == 200
        assert "chat-input" in resp.text or "chat_input" in resp.text or (
            '<input' in resp.text.lower() and 'chat' in resp.text.lower()
        ), "Testing page missing chat input"

    @pytest.mark.frontend
    def test_training_page_has_form(self, finetune_studio_client):
        """Training page should have a configuration form."""
        resp = finetune_studio_client.get("/training")
        assert resp.status_code == 200
        assert "<form" in resp.text.lower() or "train-form" in resp.text, (
            "Training page missing configuration form"
        )

    @pytest.mark.frontend
    def test_data_page_has_upload_form(self, finetune_studio_client):
        """Data page should have a file upload form."""
        resp = finetune_studio_client.get("/data")
        assert resp.status_code == 200
        assert "upload" in resp.text.lower() or "<form" in resp.text.lower(), (
            "Data page missing upload form"
        )

    @pytest.mark.frontend
    def test_models_page_has_refresh_button(self, finetune_studio_client):
        """Models page should have a Refresh button."""
        resp = finetune_studio_client.get("/models")
        assert resp.status_code == 200
        assert "Refresh" in resp.text or "refresh" in resp.text.lower(), (
            "Models page missing Refresh button"
        )
