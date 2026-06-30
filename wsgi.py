"""WSGI entry point for Render / production (gunicorn wsgi:app)."""
from web.app import _preload_model, app

_preload_model()
