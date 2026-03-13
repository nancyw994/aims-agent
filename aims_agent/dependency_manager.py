"""
If a suggested ML package is missing, install it via pip (subprocess).
Installation failures are handled gracefully and recorded in logs.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from pathlib import Path

def is_installed(import_name):
    """check if a package is installed"""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

PIP_TO_IMPORT: dict[str, str] = {
    "scikit-learn": "sklearn",
    "scikit-learn-intelex": "sklearn",
}

# Log file for installation errors
LOG_DIR = Path(__file__).resolve().parent.parent
INSTALL_LOG = LOG_DIR / "logs" / "dependency_install.log"


def _ensure_log_dir() -> None:
    INSTALL_LOG.parent.mkdir(parents=True, exist_ok=True)


def _get_logger() -> logging.Logger:
    _ensure_log_dir()
    logger = logging.getLogger("aims_agent.dependency_manager")
    if not logger.handlers:
        handler = logging.FileHandler(INSTALL_LOG, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


def _import_name_for_package(package_name: str) -> str:
    """Return the module name to use for import (may differ from pip name)."""
    return PIP_TO_IMPORT.get(package_name.strip().lower(), package_name.strip())


def is_package_available(package_name: str) -> bool:
    """Return True if the package can be imported (already installed)."""
    import_name = _import_name_for_package(package_name)
    try:
        importlib.import_module(import_name)
        return True
    except Exception:
        return False


def install_package(package_name: str) -> bool:
    """
    Run pip install for the given package. Return True if install succeeded.

    On failure, log the error trace and return False.
    """
    logger = _get_logger()
    cmd = [sys.executable, "-m", "pip", "install", package_name]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.error(
                "pip install %s failed (exit %s): %s",
                package_name,
                result.returncode,
                result.stderr or result.stdout,
            )
            return False
        logger.info("pip install %s succeeded", package_name)
        return True
    except subprocess.TimeoutExpired as e:
        logger.exception("pip install %s timed out: %s", package_name, e)
        return False
    except Exception as e:
        logger.exception("pip install %s error: %s", package_name, e)
        return False


def ensure_package_installed(package_name: str) -> bool:
    """
    Ensure the package is available. If not, install it via pip.

    Returns True if the package is importable after this call (either already
    installed or successfully installed). Returns False if installation failed;
    the error is recorded in logs.
    """
    if is_package_available(package_name):
        return True
    if not install_package(package_name):
        return False
    return is_package_available(package_name)


__all__ = [
    "ensure_package_installed",
    "install_package",
    "is_package_available",
    "INSTALL_LOG",
]
