"""Local conftest for tests/api/.

WHY THIS EXISTS
===============
The root conftest.py's `finetune_studio_client` fixture mocks
`discovered_models` as `["fake_model"]` (a list of strings). The
`/api/models/list` route iterates over that list and tries to read
`.name`, `.path`, `.format` attributes, which raises
`AttributeError: 'str' object has no attribute 'name'`.

This conftest OVERRIDES the `finetune_studio_client` fixture for the
api/ test directory only — keeping the root conftest untouched (DRY
principle preserved for everyone else) while giving the API tests
a fixture that returns proper ModelInfo-like objects.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import warnings
_warnings_before = warnings.filters[:]
warnings.simplefilter("ignore")  # suppress ALL warnings during import
from fastapi.testclient import TestClient
warnings.filters[:] = _warnings_before  # restore original filters
import pytest

# ── Session-wide heavy-resource mocks ─────────────────────────────────────
# Some tests construct a fresh TestClient(app) without going through the
# `inference_server_v2_client` fixture, which means lifespan runs and
# tries to instantiate the REAL RAGStore / DocumentIngestor. Those try to
# import chromadb and read the disk. We block that path for every test
# in this directory by stubbing the chromadb module out of sys.modules.


@pytest.fixture(autouse=True)
def _stub_heavy_modules():
    """Auto-applied: stub chromadb + llama_cpp out of sys.modules for tests here."""
    sys.modules.setdefault("chromadb", MagicMock(PersistentClient=MagicMock(return_value=MagicMock())))
    sys.modules.setdefault("llama_cpp", MagicMock(Llama=MagicMock(return_value=MagicMock())))
    yield
    # Don't remove on teardown — let other tests in the directory use the stub.


@pytest.fixture
def finetune_studio_client(mocker, mock_engine, mock_rag_store):
    """TestClient for finetune-studio WebUI with proper ModelInfo mocks."""
    # Build a list of fake ModelInfo objects so attribute access works.
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
    mocker.patch("finetune_studio.webui.app.inference_engine", mock_engine, create=True)
    mocker.patch("finetune_studio.webui.app.training_engine", MagicMock(), create=True)

    from finetune_studio.webui.app import app

    return TestClient(app)


# ── Inference-server fixture overrides ────────────────────────────────────
# The root conftest patches InferenceEngine/RAGStore but the server_v2
# module-level `engine = None` stays None until lifespan runs, and
# `TestClient(app)` (no context manager) doesn't trigger lifespan.
# We bypass lifespan entirely by setting globals directly after import.


@pytest.fixture
def inference_server_client(mocker, mock_engine, mock_rag_store):
    """TestClient for inference-server /v1 API (server.py).

    Forces engine + rag_store into module globals so endpoints see them
    without having to run the lifespan context manager.
    """
    mocker.patch("inference_server.inference.InferenceEngine", return_value=mock_engine)
    mocker.patch("inference_server.rag.RAGStore", return_value=mock_rag_store)

    from fastapi.testclient import TestClient

    import inference_server.config as cfg_mod
    import inference_server.server as srv

    # Same setup as v2: extra methods + ingestor + config.
    mock_rag_store.add_document = MagicMock(return_value=1)
    mock_rag_store.add_chunks = MagicMock(return_value=1)
    mock_rag_store.count = MagicMock(return_value=0)
    mock_rag_store.remove_document = MagicMock(return_value=0)
    mock_rag_store.list_documents = MagicMock(return_value=[])

    mock_ingestor = MagicMock()
    mock_ingestor.ingest_directory = MagicMock(return_value={"files_ingested": 0, "chunks_added": 0})
    mocker.patch("inference_server.rag.DocumentIngestor", return_value=mock_ingestor)

    fake_config = MagicMock()
    fake_config.api.key = ""
    fake_config.model.path = ""
    fake_config.rag.enabled = True
    fake_config.rag.store_path = "/tmp/fake_rag"
    fake_config.rag.documents_path = "/tmp/fake_docs"
    fake_config.rag.chunk_size = 512
    fake_config.rag.chunk_overlap = 50
    fake_config.rag.embedding_model = "all-MiniLM-L6-v2"
    fake_config.rag.min_score = 0.3
    cfg_mod.config = fake_config

    srv.engine = mock_engine
    srv.rag_store = mock_rag_store
    srv.ingestor = mock_ingestor
    srv.config = fake_config
    srv.start_time = 0.0

    return TestClient(srv.app)


@pytest.fixture
def inference_server_v2_client(mocker, mock_engine, mock_rag_store):
    """TestClient for inference-server v2 API (server_v2.py)."""
    mocker.patch("inference_server.inference.InferenceEngine", return_value=mock_engine)
    mocker.patch("inference_server.rag.RAGStore", return_value=mock_rag_store)

    from fastapi.testclient import TestClient

    import inference_server.config as cfg_mod
    import inference_server.server_v2 as srv2

    # Add the methods server_v2 calls but conftest's mock_rag_store lacks.
    mock_rag_store.add_chunks = MagicMock(return_value=0)
    mock_rag_store.count = MagicMock(return_value=0)
    mock_rag_store.remove_document = MagicMock(return_value=0)
    mock_rag_store.list_documents = MagicMock(return_value=[])

    # Mock ingestor + mcp_server so handlers don't blow up on missing attrs.
    mock_ingestor = MagicMock()
    mock_ingestor.ingest_directory = MagicMock(return_value={"files_ingested": 0, "chunks_added": 0})
    mock_mcp = MagicMock()
    mock_mcp.tools = []  # MCP tools list endpoint reads this
    mock_mcp.list_tools = MagicMock(return_value=[])  # server_v2 reads this directly
    mock_mcp.to_dict = MagicMock(return_value={"name": "fake-mcp"})
    mock_mcp.execute = MagicMock(return_value={"result": "ok"})
    mocker.patch("inference_server.rag.DocumentIngestor", return_value=mock_ingestor)
    mocker.patch("inference_server.mcp.RAGMCPServer", return_value=mock_mcp)

    # rag_store methods used by ingest_bytes() in parsers.py.
    mock_rag_store.add_document = MagicMock(return_value=1)
    mock_rag_store.add_chunks = MagicMock(return_value=1)

    # Build a fully-formed config object so endpoints can read nested attrs.
    fake_config = MagicMock()
    fake_config.api.key = ""  # disable auth
    fake_config.model.path = ""  # don't try to load a real GGUF
    fake_config.rag.enabled = True
    fake_config.rag.store_path = "/tmp/fake_rag"
    fake_config.rag.documents_path = "/tmp/fake_docs"
    fake_config.rag.chunk_size = 512
    fake_config.rag.chunk_overlap = 50
    fake_config.rag.embedding_model = "all-MiniLM-L6-v2"
    fake_config.rag.min_score = 0.3
    fake_config.server.host = "127.0.0.1"
    fake_config.server.port = 8888
    cfg_mod.config = fake_config

    srv2.engine = mock_engine
    srv2.rag_store = mock_rag_store
    srv2.ingestor = mock_ingestor
    srv2.mcp_server = mock_mcp
    srv2.config = fake_config
    srv2.start_time = 0.0

    return TestClient(srv2.app)
