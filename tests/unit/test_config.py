"""Tests for finetune_studio.config module."""
from __future__ import annotations

import ast
from pathlib import Path


# Resolve finetune-studio source root across machine layouts.
def _resolve_config_path() -> Path:
    """Find finetune_studio/config.py on disk (genorbox1, fan-dragon, or CI)."""
    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        # ai-trainer layout (canonical)
        candidate = ancestor / "trainer" / "src" / "finetune_studio" / "config.py"
        if candidate.exists():
            return candidate
        # finetune-studio layout (legacy)
        candidate = ancestor / "finetune-studio" / "src" / "finetune_studio" / "config.py"
        if candidate.exists():
            return candidate
    return here.parent.parent.parent / "trainer" / "src" / "finetune_studio" / "config.py"


CONFIG_PATH = _resolve_config_path()


class TestSettings:
    """Tests for the Settings dataclass."""

    def test_settings_defaults(self):
        """Settings has sensible defaults."""
        from finetune_studio.config import Settings
        s = Settings()
        assert s.host == "0.0.0.0"
        assert s.port == 7860
        assert s.debug is False
        assert s.default_lora_rank == 64
        assert s.default_lr == 8e-5
        assert s.default_epochs == 4
        assert s.default_batch_size == 2
        assert s.default_max_seq_length == 2048
        assert s.data_dir == "data"
        assert s.db_path == "data/finetune_studio.db"
        assert s.rag_store_path == "data/rag_store"
        assert s.rag_embedding_model == "all-MiniLM-L6-v2"

    def test_settings_custom_values(self):
        """Settings accepts custom values."""
        from finetune_studio.config import Settings
        s = Settings(host="127.0.0.1", port=9090, debug=True, default_lr=1e-4)
        assert s.host == "127.0.0.1"
        assert s.port == 9090
        assert s.debug is True
        assert s.default_lr == 1e-4

    def test_settings_model_dirs_is_list(self):
        """model_dirs is a list of paths."""
        from finetune_studio.config import Settings
        s = Settings()
        assert isinstance(s.model_dirs, list)
        assert len(s.model_dirs) >= 1
        assert all(isinstance(d, str) for d in s.model_dirs)

    def test_settings_rag_is_ragsettings(self):
        """Settings.rag is an RAGSettings instance."""
        from finetune_studio.config import RAGSettings, Settings
        s = Settings()
        assert isinstance(s.rag, RAGSettings)

    def test_settings_singleton_exists(self):
        """Module-level singleton is created."""
        from finetune_studio import config
        assert hasattr(config, "settings")
        assert isinstance(config.settings, config.Settings)


class TestRAGSettings:
    """Tests for the RAGSettings dataclass."""

    def test_rag_settings_defaults(self):
        """RAGSettings has sensible defaults."""
        from finetune_studio.config import RAGSettings
        r = RAGSettings()
        assert r.store_path == "data/rag_store"
        assert r.embedding_model == "all-MiniLM-L6-v2"
        assert r.min_score == 0.3
        assert r.chunk_size == 512
        assert r.chunk_overlap == 50
        assert r.enabled is True

    def test_rag_settings_custom(self):
        """RAGSettings accepts custom values."""
        from finetune_studio.config import RAGSettings
        r = RAGSettings(store_path="/tmp/store", min_score=0.5, chunk_size=256)
        assert r.store_path == "/tmp/store"
        assert r.min_score == 0.5
        assert r.chunk_size == 256


class TestConfigSourceStructure:
    """Test config.py source code structure."""

    def test_has_settings_class(self):
        """config.py defines Settings class."""
        tree = ast.parse(CONFIG_PATH.read_text())
        class_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert "Settings" in class_names
        assert "RAGSettings" in class_names

    def test_has_settings_singleton(self):
        """config.py creates module-level settings."""
        content = CONFIG_PATH.read_text()
        assert "settings = Settings()" in content

    def test_settings_fields(self):
        """Settings has expected field names."""
        tree = ast.parse(CONFIG_PATH.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Settings":
                field_names = [
                    n.target.id
                    for n in node.body
                    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                ]
                assert "host" in field_names
                assert "port" in field_names
                assert "debug" in field_names
                assert "default_lr" in field_names
                assert "rag" in field_names
                break

    def test_rag_settings_defined_before_settings(self):
        """RAGSettings must be defined BEFORE Settings (no forward-reference bug)."""
        tree = ast.parse(CONFIG_PATH.read_text())
        rag_line = None
        settings_line = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name == "RAGSettings":
                    rag_line = node.lineno
                elif node.name == "Settings":
                    settings_line = node.lineno
        assert rag_line is not None, "RAGSettings class not found"
        assert settings_line is not None, "Settings class not found"
        assert rag_line < settings_line, (
            f"RAGSettings (line {rag_line}) must be defined BEFORE "
            f"Settings (line {settings_line}) to avoid forward-reference bug"
        )
