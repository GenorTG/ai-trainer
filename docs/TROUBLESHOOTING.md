# Troubleshooting — Common issues and fixes

## Build Errors

### llama-cpp-python: CUDA build fails
```
RuntimeError: Cannot find CUDA toolkit
```
**Fix**: install CUDA toolkit and re-run with explicit CUDA flags:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

### PyTorch CUDA mismatch
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
**Fix**: install pytorch matching your CUDA version:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Runtime Errors

### "Model not loaded"
```
{"error": "No model loaded"}
```
**Fix**: load model first:
```bash
inference-server serve --model models/v21/chris-ai-gemma4-e4b-v21.Q4_K_M.gguf
```

### Out of memory (CUDA OOM)
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Fix**: reduce GPU offloading or use smaller quantization:
```bash
# Use partial GPU offload
inference-server serve --model model.gguf --n-gpu-layers 30

# Use smaller quant (Q4_K_M instead of Q5_K_M)
# Use Q4_0 instead of Q4_K_M if needed
```

### "RAG not enabled"
```
{"error": "RAG not enabled"}
```
**Fix**: enable RAG in config:
```yaml
# config.yaml
rag:
  enabled: true
  embedding_model: all-MiniLM-L6-v2
  persist_directory: ./rag_data
```

### "MCP server not initialized"
```
{"error": "MCP server not initialized"}
```
**Fix**: ensure MCP server starts in lifespan:
```bash
inference-server start --enable-mcp
```

## Network Errors

### Connection refused on localhost:8888
```
curl: (7) Failed to connect to localhost port 8888
```
**Fix**: server isn't running. Start it:
```bash
inference-server start --model model.gguf
```

### 401 Unauthorized
```
{"error": "Invalid API key"}
```
**Fix**: set `INFERENCE_API_KEY` env var and pass in header:
```bash
export INFERENCE_API_KEY="your-key-here"
curl -H "Authorization: Bearer your-key-here" http://localhost:8888/v1/models
```

### 503 Service Unavailable
```
{"error": "Model loading..."}
```
**Fix**: wait for model to finish loading. Check:
```bash
curl http://localhost:8888/health
```

## Training Errors

### Loss is NaN
```
{"loss": NaN}
```
**Fix**: reduce learning rate:
```bash
python train_v21.py --learning-rate 1e-5  # was 2e-5
```

### "Out of memory" during training
**Fix**: enable QLoRA:
```python
# In train script
use_qlora = True
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

### PYTORCH_CUDA_ALLOC_CONF warnings
```
[W000] TORCH_CUDA_ALLOC_CONF is set but unused
```
**Fix**: set to empty or use the right format:
```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

## Test Errors

### "Module not found" during pytest
```
ModuleNotFoundError: No module named 'inference_server'
```
**Fix**: install package in editable mode:
```bash
pip install -e ./server
```

### Tests fail with import errors after refactor
**Fix**: clear pytest cache and rerun:
```bash
find . -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name .pytest_cache -exec rm -rf {} + 2>/dev/null
python tests/run_all.py
```

### Frontend test fails: "fetch is not defined"
**Fix**: tests need `fetch` shim or to mock it. Already handled in conftest.py.

## Performance Issues

### Inference is slow (>2s per token)
- Check GPU usage: `nvidia-smi`
- Increase `--n-gpu-layers` to fully offload
- Use smaller quant (Q4_0 instead of Q8_0)
- Reduce context length (8K → 4K if not needed)

### Training is slow
- Enable mixed precision (bf16)
- Use gradient checkpointing
- Increase batch size if VRAM allows
- Use flash attention if available

### RAG search is slow
- Reduce number of returned chunks
- Cache embeddings (set `persist_directory`)
- Use smaller embedding model

## Git Issues

### "fatal: refusing to merge unrelated histories"
**Fix**: allow unrelated history:
```bash
git pull origin main --allow-unrelated-histories
```

### Pre-commit hooks failing
**Fix**: run hooks manually:
```bash
pre-commit run --all-files
# or skip hooks for urgent commits
git commit --no-verify
```

### "your branch is ahead of origin/main"
**Fix**: push or pull:
```bash
git push origin main
# or
git pull origin main
```

## ComfyUI Conflicts (fan-dragon)

ComfyUI uses 10-17GB GPU. Pauses benchmarking if GPU >90%.

**Check GPU usage**:
```bash
nvidia-smi
```

**If GPU busy**: pause benchmark work, wait for user.

**See also**: `docs/adr/005-fan-dragon-primary.md`

## Getting More Help

1. Check docs: `README.md`, `docs/`, per-module READMEs
2. Search issues: `git log --oneline | grep "fix"`
3. Run tests: `python tests/run_all.py`
4. Check lint: `make lint type-check security`
5. File an issue (private if security-related)