"""Load environment variables from the project .env file."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_env() -> None:
    """Load .env from the Surfer project root regardless of cwd."""
    load_dotenv(ENV_FILE, override=False)


def env_status() -> str:
    """Human-readable hint when NVIDIA_API_KEY is missing."""
    if not ENV_FILE.exists():
        return f"No .env file at {ENV_FILE}. Copy .env.example and add your key."
    if ENV_FILE.stat().st_size == 0:
        return f".env exists at {ENV_FILE} but is empty. Save your NVIDIA_API_KEY to the file."
    if not os.getenv("NVIDIA_API_KEY"):
        return f".env at {ENV_FILE} was read but NVIDIA_API_KEY is missing or blank."
    return ""
