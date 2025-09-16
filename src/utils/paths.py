# src/utils/paths.py
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import os

# Carrega o .env automaticamente
dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path=dotenv_path)

PROJECT_ROOT = Path(dotenv_path).resolve().parent if dotenv_path else Path.cwd()

def resolve_env_path(key: str, default: str = ".") -> Path:
    """Converte o valor de uma variável do .env em Path absoluto."""
    raw = os.getenv(key, default)
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


