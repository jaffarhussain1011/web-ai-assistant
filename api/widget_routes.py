"""
api/widget_routes.py
────────────────────
Widget configuration endpoints.

GET  /widget/config  — public; returns merged defaults + saved config
POST /widget/config  — admin; saves customization to data/widget_config.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["widget"])

_CONFIG_PATH = Path(__file__).parent.parent / "data" / "widget_config.json"

DEFAULTS: dict = {
    "preset":         "default",
    "primaryColor":   "#4F46E5",
    "title":          "Live Support",
    "placeholder":    "Ask a question…",
    "welcomeMessage": "👋 Hi! I'm your support assistant. What can I help you with?",
    "position":       "bottom-right",
}


def _load() -> dict:
    """Return DEFAULTS merged with any saved config (saved keys win)."""
    if not _CONFIG_PATH.exists():
        return dict(DEFAULTS)
    try:
        saved = json.loads(_CONFIG_PATH.read_text())
        return {**DEFAULTS, **saved}
    except Exception as exc:
        logger.warning("Could not read widget config: %s", exc)
        return dict(DEFAULTS)


def _save(cfg: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


class WidgetConfig(BaseModel):
    preset:         str | None = None
    primaryColor:   str | None = None
    title:          str | None = None
    placeholder:    str | None = None
    welcomeMessage: str | None = None
    position:       str | None = None


@router.get("/widget/config")
def get_widget_config():
    """Return the active widget configuration (defaults if not yet customised)."""
    return _load()


@router.post("/widget/config")
def save_widget_config(cfg: WidgetConfig):
    """Persist widget customisation. Omitted fields keep their current values."""
    current = _load()
    updates = {k: v for k, v in cfg.model_dump().items() if v is not None}
    merged  = {**current, **updates}
    _save(merged)
    logger.info("Widget config saved: %s", merged)
    return {"ok": True, "config": merged}
