"""Load the DataBento API key from a .gitignored .env or the environment.

Keeps the secret out of source. No python-dotenv dependency: this reads the
repo-root .env directly if present, otherwise falls back to an existing env var.
"""
import os
from pathlib import Path


def get_databento_key() -> str:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit(
            "DATABENTO_API_KEY not set — add it to a .gitignored .env "
            "(DATABENTO_API_KEY=...) or export it in your environment."
        )
    return key
