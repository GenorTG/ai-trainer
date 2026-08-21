"""Gradio web app composition.

WHAT THIS FILE DOES
==================
The main entry point for the web UI. Composes all the tabs
(data, models, training, testing, comparison) into a single Gradio
interface and launches it on port 7860.

KEY CONCEPTS
============
- Gradio: a Python library for creating web UIs for ML models.
  Defines UI as Python objects, no HTML/JS needed.
- Tab-based layout: each major feature gets its own tab.
- Event handlers: when the user clicks a button, we run a Python function.
- State management: the UI keeps state across interactions (which
  model is loaded, what test suite is selected, etc.).
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from finetune_studio.config import settings
from finetune_studio.models.registry import scan_models
from finetune_studio.testing.inference import InferenceEngine
from finetune_studio.training.engine import TrainingEngine

training_engine = TrainingEngine()
inference_engine = InferenceEngine()
discovered_models: list[str] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global discovered_models
    print("Scanning model directories...")
    discovered_models = scan_models(settings.model_dirs)
    print(f"Found {len(discovered_models)} models")
    yield

app = FastAPI(title="Finetune Studio", version="0.1.0", lifespan=lifespan)

static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
static_dir.mkdir(parents=True, exist_ok=True)
(static_dir / "css").mkdir(exist_ok=True)
(static_dir / "js").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

from finetune_studio.webui.routes import (
    comparison,
    data,
    models,
    pages,
    quality,
    testing,
    training,
)

app.include_router(pages.router)
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(training.router, prefix="/api/training", tags=["training"])  # type: ignore[has-type]
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(testing.router, prefix="/api/testing", tags=["testing"])  # type: ignore[has-type]
app.include_router(comparison.router, prefix="/api/compare", tags=["compare"])
app.include_router(quality.router)  # already self-prefixed /api/data
