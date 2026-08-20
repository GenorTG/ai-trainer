"""Tests for inference_server/samplers.py."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestSamplerConfig:
    def test_defaults(self):
        from inference_server.samplers import SamplerConfig
        cfg = SamplerConfig()
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9
        assert cfg.top_k == 40
        assert cfg.repeat_penalty == 1.1
        assert cfg.min_p == 0.05
        assert cfg.max_tokens == 512
        assert cfg.seed == -1
        assert cfg.stop == []

    def test_custom_values(self):
        from inference_server.samplers import SamplerConfig
        cfg = SamplerConfig(temperature=1.5, top_p=0.8, top_k=20, max_tokens=1024, seed=42)
        assert cfg.temperature == 1.5
        assert cfg.top_p == 0.8
        assert cfg.top_k == 20
        assert cfg.max_tokens == 1024
        assert cfg.seed == 42

    def test_to_llama_cpp_kwargs(self):
        from inference_server.samplers import SamplerConfig
        cfg = SamplerConfig(temperature=0.5, seed=42, stop=["END"])
        kwargs = cfg.to_llama_cpp_kwargs()
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.9
        assert kwargs["seed"] == 42
        assert kwargs["stop"] == ["END"]
        assert kwargs["max_tokens"] == 512

    def test_to_llama_cpp_kwargs_no_seed(self):
        from inference_server.samplers import SamplerConfig
        cfg = SamplerConfig(seed=-1)
        kwargs = cfg.to_llama_cpp_kwargs()
        assert "seed" not in kwargs

    def test_to_llama_cpp_kwargs_no_stop(self):
        from inference_server.samplers import SamplerConfig
        cfg = SamplerConfig(stop=[])
        kwargs = cfg.to_llama_cpp_kwargs()
        assert "stop" not in kwargs

    def test_to_llama_cpp_temperature_floor(self):
        """Temperature=0 gets clamped to 0.01 for llama.cpp."""
        from inference_server.samplers import SamplerConfig
        cfg = SamplerConfig(temperature=0.0)
        kwargs = cfg.to_llama_cpp_kwargs()
        assert kwargs["temperature"] == 0.01

    def test_to_openai_kwargs(self):
        from inference_server.samplers import SamplerConfig
        cfg = SamplerConfig(temperature=0.8, frequency_penalty=0.5, presence_penalty=0.3)
        kwargs = cfg.to_openai_kwargs()
        assert kwargs["temperature"] == 0.8
        assert kwargs["frequency_penalty"] == 0.5
        assert kwargs["presence_penalty"] == 0.3
        assert kwargs["max_tokens"] == 512

    def test_from_dict_filters_unknown_keys(self):
        from inference_server.samplers import SamplerConfig
        d = {"temperature": 0.3, "unknown_key": 999, "top_p": 0.8}
        cfg = SamplerConfig.from_dict(d)
        assert cfg.temperature == 0.3
        assert cfg.top_p == 0.8

    def test_from_dict_empty(self):
        from inference_server.samplers import SamplerConfig
        cfg = SamplerConfig.from_dict({})
        assert cfg.temperature == 0.7  # default

    def test_stop_list_mutable(self):
        from inference_server.samplers import SamplerConfig
        cfg1 = SamplerConfig()
        cfg1.stop.append("END")
        cfg2 = SamplerConfig()
        assert cfg2.stop == []  # not shared


@pytest.mark.unit
class TestGetSampler:
    def test_get_balanced(self):
        from inference_server.samplers import get_sampler
        cfg = get_sampler("balanced")
        assert cfg.temperature == 0.7

    def test_get_deterministic(self):
        from inference_server.samplers import get_sampler
        cfg = get_sampler("deterministic")
        assert cfg.temperature == 0.0
        assert cfg.top_k == 1

    def test_get_creative(self):
        from inference_server.samplers import get_sampler
        cfg = get_sampler("creative")
        assert cfg.temperature == 1.0

    def test_get_conservative(self):
        from inference_server.samplers import get_sampler
        cfg = get_sampler("conservative")
        assert cfg.temperature == 0.3

    def test_get_testing(self):
        from inference_server.samplers import get_sampler
        cfg = get_sampler("testing")
        assert cfg.max_tokens == 256

    def test_get_unknown_returns_balanced(self):
        from inference_server.samplers import get_sampler, SamplerConfig
        cfg = get_sampler("nonexistent_preset")
        assert cfg == SamplerConfig()  # balanced is the default

    def test_get_default(self):
        from inference_server.samplers import get_sampler
        cfg = get_sampler()
        assert cfg.temperature == 0.7

    def test_chris_ai_presets(self):
        from inference_server.samplers import get_sampler
        v20 = get_sampler("chris_ai_v20")
        v21 = get_sampler("chris_ai_v21")
        assert v20.temperature == 0.45
        assert v21.temperature == 0.25


@pytest.mark.unit
class TestListPresets:
    def test_returns_all_presets(self):
        from inference_server.samplers import list_presets, PRESETS
        presets = list_presets()
        assert len(presets) == len(PRESETS)
        for name in PRESETS:
            assert name in presets

    def test_preset_structure(self):
        from inference_server.samplers import list_presets
        presets = list_presets()
        for name, values in presets.items():
            assert "temperature" in values
            assert "top_p" in values
            assert "top_k" in values
            assert "repeat_penalty" in values
            assert "min_p" in values

    def test_preset_values_match(self):
        from inference_server.samplers import list_presets, PRESETS
        presets = list_presets()
        for name, p in PRESETS.items():
            assert presets[name]["temperature"] == p.temperature
            assert presets[name]["top_p"] == p.top_p
