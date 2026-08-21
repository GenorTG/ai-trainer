# mypy: disable-error-code="arg-type,call-arg"
# Reason: starlette's Jinja2Templates.TemplateResponse typing stubs use the
# newer signature `TemplateResponse(request, name, context)` but the common
# pattern (used here) is `TemplateResponse(name, context_dict)` where context
# contains "request". This is a long-standing stubs issue; see
# https://github.com/encode/starlette/issues/1426
"""Main page layouts (home, settings, help).

WHAT THIS FILE DOES
===================
Defines the HTML page routes for the finetune-studio web UI:
  - GET /          → home page (index.html)
  - GET /models    → model browser
  - GET /training  → training dashboard
  - GET /data      → data files
  - GET /testing   → testing/inference playground

KEY CONCEPTS
============
- FastAPI route handlers: async functions returning HTML responses.
- Jinja2Templates: starlette's templating engine for rendering Jinja2 templates.
- Per-route # type: ignore: starlette's TemplateResponse has a known typing quirk
  where the modern signature requires positional Request as first arg, but the
  common pattern (used here) is `TemplateResponse(name, context_dict)` where
  context_dict contains "request". This is a long-standing stubs issue, not a
  real bug. See: https://github.com/encode/starlette/issues/1426
"""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page with overview dashboard."""
    from finetune_studio.webui.app import discovered_models, training_engine

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "models": discovered_models,
            "training_state": training_engine.state,
        },
    )


@router.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """Model browser page."""
    from finetune_studio.webui.app import discovered_models

    return templates.TemplateResponse(
        request,
        "models.html",
        {"request": request, "models": discovered_models},
    )


@router.get("/training", response_class=HTMLResponse)
async def training_page(request: Request):
    """Training dashboard page."""
    from finetune_studio.webui.app import discovered_models, training_engine

    return templates.TemplateResponse(
        request,
        "training.html",
        {
            "request": request,
            "models": discovered_models,
            "training_state": training_engine.state,
        },
    )


@router.get("/data", response_class=HTMLResponse)
async def data_page(request: Request):
    """Data files browser page."""
    from finetune_studio.config import settings
    from finetune_studio.data.organizer import scan_data_files

    files = scan_data_files(settings.data_dir)
    return templates.TemplateResponse(
        request,
        "data.html",
        {"request": request, "files": files},
    )


@router.get("/testing", response_class=HTMLResponse)
async def testing_page(request: Request):
    """Testing/inference playground page."""
    from finetune_studio.webui.app import discovered_models, inference_engine

    return templates.TemplateResponse(
        request,
        "testing.html",
        {
            "request": request,
            "models": discovered_models,
            "inference_engine": inference_engine,
        },
    )
