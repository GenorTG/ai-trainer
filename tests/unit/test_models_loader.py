"""Tests for finetune_studio.models.loader module."""
import json
from unittest.mock import MagicMock, patch


class TestLoadModelInfo:
    """Tests for load_model_info function."""

    def test_gguf_file_info(self, tmp_dir):
        """GGUF file returns correct metadata."""
        from finetune_studio.models.loader import load_model_info
        p = tmp_dir / "model.gguf"
        p.write_bytes(b"GGUF" + b"\x03" * 100)
        info = load_model_info(str(p))
        assert info["format"] == "gguf"
        assert info["name"] == "model.gguf"
        assert "size_gb" in info
        assert info["size_gb"] >= 0

    def test_hf_directory_info(self, tmp_dir):
        """HuggingFace directory with config.json returns correct metadata."""
        from finetune_studio.models.loader import load_model_info
        d = tmp_dir / "hf_model"
        d.mkdir()
        config = {
            "architectures": ["Gemma2ForCausalLM"],
            "model_type": "gemma2",
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "vocab_size": 256000,
        }
        (d / "config.json").write_text(json.dumps(config))
        # Create dummy safetensors files
        (d / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 1024)
        (d / "model-00002-of-00002.safetensors").write_bytes(b"\x00" * 2048)
        info = load_model_info(str(d))
        assert info["format"] == "safetensors"
        assert info["architectures"] == ["Gemma2ForCausalLM"]
        assert info["model_type"] == "gemma2"
        assert info["hidden_size"] == 3584
        assert info["num_layers"] == 28
        assert info["vocab_size"] == 256000
        assert info["shards"] == 2
        assert info["size_gb"] >= 0

    def test_empty_directory(self, tmp_dir):
        """Directory without config.json returns basic info."""
        from finetune_studio.models.loader import load_model_info
        d = tmp_dir / "empty_model"
        d.mkdir()
        info = load_model_info(str(d))
        assert info["name"] == "empty_model"
        assert "format" not in info

    def test_nonexistent_path(self):
        """Non-existent path returns basic info."""
        from finetune_studio.models.loader import load_model_info
        info = load_model_info("/nonexistent/path/model.gguf")
        assert info["name"] == "model.gguf"
        assert "format" not in info

    def test_non_gguf_file(self, tmp_dir):
        """Non-GGUF file returns basic info."""
        from finetune_studio.models.loader import load_model_info
        p = tmp_dir / "model.bin"
        p.write_bytes(b"\x00" * 100)
        info = load_model_info(str(p))
        assert info["name"] == "model.bin"
        assert "format" not in info


class TestLoadForInference:
    """Tests for load_for_inference function (mocked)."""

    def test_gguf_calls_load_gguf_inference(self, tmp_dir):
        """GGUF file path triggers load_gguf_inference."""
        from finetune_studio.models.loader import load_for_inference
        p = tmp_dir / "model.gguf"
        p.write_bytes(b"GGUF" + b"\x00" * 100)
        mock_model = MagicMock()
        # Must mock torch and transformers at sys.modules level
        # because load_for_inference imports them unconditionally
        mock_torch = MagicMock()
        mock_transformers = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}), \
             patch("finetune_studio.models.loader.load_gguf_inference", return_value=(mock_model, None)) as mock_load:
            result = load_for_inference(str(p))
            mock_load.assert_called_once_with(str(p))
            assert result == (mock_model, None)

    def test_hf_calls_transformers(self, tmp_dir):
        """Non-GGUF path triggers transformers loading."""
        from finetune_studio.models.loader import load_for_inference
        d = tmp_dir / "hf_model"
        d.mkdir()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_torch = MagicMock()
        mock_torch.float16 = "float16"
        mock_auto = MagicMock()
        mock_auto.from_pretrained.return_value = mock_model
        mock_tok = MagicMock()
        mock_tok.from_pretrained.return_value = mock_tokenizer
        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "transformers": MagicMock(AutoModelForCausalLM=mock_auto, AutoTokenizer=mock_tok),
        }):
            load_for_inference(str(d))
            mock_tok.from_pretrained.assert_called_once()
            mock_auto.from_pretrained.assert_called_once()

    def test_gguf_custom_params(self, tmp_dir):
        """load_gguf_inference receives custom parameters."""
        from finetune_studio.models.loader import load_for_inference
        p = tmp_dir / "model.gguf"
        p.write_bytes(b"GGUF" + b"\x00" * 100)
        mock_torch = MagicMock()
        mock_transformers = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}), \
             patch("finetune_studio.models.loader.load_gguf_inference", return_value=(MagicMock(), None)) as mock_load:
            load_for_inference(str(p), n_ctx=2048, n_gpu_layers=0)
            mock_load.assert_called_once_with(str(p), n_ctx=2048, n_gpu_layers=0)


class TestLoadGgufInference:
    """Tests for load_gguf_inference function (mocked)."""

    def test_loads_with_llama_cpp(self, tmp_dir):
        """Uses llama_cpp.Llama to load GGUF."""
        from finetune_studio.models.loader import load_gguf_inference
        p = tmp_dir / "model.gguf"
        p.write_bytes(b"GGUF" + b"\x00" * 100)
        mock_llama = MagicMock()
        mock_model = MagicMock()
        mock_llama.return_value = mock_model
        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama)}):
            model, tokenizer = load_gguf_inference(str(p))
            mock_llama.assert_called_once_with(
                model_path=str(p), n_ctx=4096, n_gpu_layers=99, verbose=False
            )
            assert model is mock_model
            assert tokenizer is None
