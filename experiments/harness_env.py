"""Shared subprocess environment for the measurement scripts.

Configure with environment variables before running:

- ``NUQTA_TESS_ROOT``: optional.  Root of an extracted tesseract tree (e.g.
  unpacked from RPMs without sudo); ``usr/bin`` is prepended to PATH and
  ``usr/lib64`` to LD_LIBRARY_PATH so OCRmyPDF's binary check passes.  When
  unset, tesseract must already be on PATH.
- ``NUQTA_OCRMYPDF_SRC``: optional.  Extra PYTHONPATH entries (colon
  separated), e.g. a nuqta-ocr checkout with the ``read_lines`` API.
"""
from __future__ import annotations

import os


def ocrmypdf_env() -> dict:
    """Environment for subprocess ``python -m ocrmypdf`` calls."""
    env = dict(os.environ)
    tess_root = os.environ.get('NUQTA_TESS_ROOT')
    if tess_root:
        env['PATH'] = tess_root + '/usr/bin:' + env['PATH']
        env['LD_LIBRARY_PATH'] = tess_root + '/usr/lib64'
    extra = os.environ.get('NUQTA_OCRMYPDF_SRC')
    if extra:
        env['PYTHONPATH'] = (extra + ':' + env.get('PYTHONPATH', '')).rstrip(':')
    return env


def plugin_src() -> str:
    """Path of the repo's ``src`` directory, for PYTHONPATH."""
    from pathlib import Path

    return str(Path(__file__).resolve().parent.parent / 'src')
