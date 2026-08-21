"""Local conftest for tests/frontend/.

WHY THIS EXISTS
===============
The root conftest's `cleanup_caches` autouse fixture unconditionally
reloads every `finetune_studio` / `inference_server` module at teardown.
Frontend tests don't import those modules, so when the reload tries to
re-import a submodule whose parent package isn't in sys.modules, it
raises ImportError and the test is marked ERROR instead of PASSED.

The root `finetune_studio_client` fixture also mocks `discovered_models`
as a list of strings, which crashes the `/api/models/list` route when
attribute access is attempted. We override that fixture too.

This conftest overrides both for the frontend test tree.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest


@pytest.fixture(autouse=True)
def cleanup_caches():
    """Clear module-level caches between tests (tolerates missing parents)."""
    yield
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith(("finetune_studio", "inference_server")):
            try:
                importlib.reload(sys.modules[mod_name])
            except ImportError:
                # Parent package not in sys.modules (frontend tests don't
                # import the runtime, only statics). Skip silently.
                pass


@pytest.fixture
def finetune_studio_client(mocker):
    """TestClient for finetune-studio WebUI with proper ModelInfo mocks.

    Same override as tests/api/conftest.py — root conftest's version
    uses a list of strings, which crashes attribute access on the
    /api/models/list route.
    """
    fake_info = MagicMock()
    fake_info.name = "fake_model"
    fake_info.path = "/fake/model.gguf"
    fake_info.format = "gguf"
    fake_info.size_gb = 4.2
    fake_info.architecture = "fake_arch"

    # IMPORTANT: import the module BEFORE mocker.patch can resolve the path.
    # mocker.patch uses pkgutil.resolve_name which requires the package to
    # already have the attribute. Importing here forces the module to load.
    import finetune_studio.webui.app as _fs_app  # noqa: F401

    mocker.patch("finetune_studio.webui.app.scan_models", return_value=[fake_info])
    mocker.patch("finetune_studio.webui.app.discovered_models", [fake_info], create=True)
    mocker.patch("finetune_studio.webui.app.inference_engine", MagicMock(), create=True)
    mocker.patch("finetune_studio.webui.app.training_engine", MagicMock(), create=True)

    from finetune_studio.webui.app import app
    return TestClient(app)
