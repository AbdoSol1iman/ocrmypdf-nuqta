"""OCRmyPDF plugin: Nuqta Arabic OCR with tashkeel, line-level hOCR.

OCRmyPDF 17.x hands the engine one PNG per page and expects hOCR whose boxes
are in pixels of that image.  Nuqta segments the page into lines, reads each
line, and writes the boxes back in the original image frame; this module turns
that reading into the hOCR shape OCRmyPDF's parser and fpdf2 renderer require.

Install the package and OCRmyPDF picks the engine up automatically for
``--ocr-engine auto``.  The model is loaded once per process, lazily, behind a
lock.  Because one model per worker is not affordable, the plugin forces
``--jobs 1`` (one process, one model).
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Literal

from ocrmypdf import hookimpl
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

__version__ = '0.1.0'
DEFAULT_MODEL = 'nuqta-0.8b'

_MODELS: dict[tuple[str, str, str, str], object] = {}
_MODEL_LOCK = threading.Lock()
_CPU_WARNED = False


class NuqtaOptions(BaseModel):
    """Options exposed as ``--nuqta-*`` on the OCRmyPDF command line."""

    model: str = Field(default=DEFAULT_MODEL, description='Nuqta model name, hub id or local path')
    backend: Literal['auto', 'transformers', 'vllm'] = Field(default='auto')
    device: Literal['auto', 'cpu', 'cuda'] = Field(default='auto')
    dtype: Literal['bfloat16', 'float16', 'float32'] = Field(default='bfloat16')
    max_new_tokens: int = Field(default=1024, ge=64, le=8192)


def _ns(options):
    """The plugin options namespace, with defaults for plain namespaces."""
    namespace = getattr(options, 'nuqta', None)
    return namespace if namespace is not None else NuqtaOptions()


@hookimpl
def register_options():
    return {'nuqta': NuqtaOptions}


@hookimpl
def add_options(parser):
    group = parser.add_argument_group('Nuqta OCR', 'Arabic OCR engine with faithful tashkeel')
    group.add_argument('--nuqta-model', dest='nuqta_model', default=None,
                       help=f'Nuqta model name, hub id or local path (default: {DEFAULT_MODEL})')
    group.add_argument('--nuqta-backend', dest='nuqta_backend', default=None,
                       choices=['auto', 'transformers', 'vllm'])
    group.add_argument('--nuqta-device', dest='nuqta_device', default=None,
                       choices=['auto', 'cpu', 'cuda'])
    group.add_argument('--nuqta-dtype', dest='nuqta_dtype', default=None,
                       choices=['bfloat16', 'float16', 'float32'])
    group.add_argument('--nuqta-max-new-tokens', dest='nuqta_max_new_tokens', type=int, default=None,
                       help='token budget per line (default: 1024)')


@hookimpl
def check_options(options):
    """Keep one model in memory and one worker: Nuqta is not cheap to fork."""
    if getattr(options, 'ocr_engine', 'auto') not in (None, 'auto'):
        return  # the user selected Tesseract or none explicitly
    jobs = getattr(options, 'jobs', None)
    if jobs != 1:
        log.warning(
            'Nuqta loads one model and is not thread-safe; using --jobs 1 instead of %s',
            'auto' if jobs in (None, 0) else jobs,
        )
        options.jobs = 1


def _resolve_device(device: str) -> str:
    if device != 'auto':
        return device
    try:
        import torch
    except ImportError:  # pragma: no cover - the model runtime needs torch anyway
        return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _load_ocr(options):
    """Import Nuqta and load the model.  Kept separate so tests can fake it."""
    from nuqta import OCR

    ns = _ns(options)
    device = _resolve_device(ns.device)
    backend = ns.backend
    if backend == 'auto':
        if device == 'cpu':
            backend = 'transformers'
        else:
            try:
                import vllm  # noqa: F401
                backend = 'vllm'
            except ImportError:
                backend = 'transformers'
    if device == 'cpu':
        global _CPU_WARNED
        if not _CPU_WARNED:
            log.warning(
                'Nuqta is running on CPU. A line takes tens of seconds there, so a page '
                'takes many minutes; use --nuqta-device cuda on a machine with an NVIDIA GPU.'
            )
            _CPU_WARNED = True
    kwargs: dict = {'dtype': ns.dtype}
    if backend != 'vllm':
        kwargs['device_map'] = 'cuda:0' if device == 'cuda' else 'cpu'
    return OCR(ns.model, backend=backend, **kwargs)


def _get_model(options):
    ns = _ns(options)
    key = (ns.model, ns.backend, ns.device, ns.dtype)
    with _MODEL_LOCK:
        model = _MODELS.get(key)
        if model is None:
            model = _load_ocr(options)
            _MODELS[key] = model
        return model


class NuqtaOcrEngine(OcrEngine):
    @staticmethod
    def version() -> str:
        return f'nuqta-ocr {__version__}'

    @staticmethod
    def creator_tag(options) -> str:
        return f'Nuqta {__version__}'

    def __str__(self) -> str:
        return f'Nuqta {__version__}'

    @staticmethod
    def languages(options) -> set[str]:
        # Nuqta is trained on Arabic (with Latin/digits occurring inline).  The
        # set is deliberately permissive: OCRmyPDF's default is -l eng, and a
        # hard failure there would make the plugin unusable out of the box.
        return {'ara', 'ar', 'eng'}

    @staticmethod
    def get_orientation(input_file: Path, options) -> OrientationConfidence:
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options) -> float:
        return 0.0

    @staticmethod
    def generate_hocr(input_file: Path, output_hocr: Path, output_text: Path, options) -> None:
        from nuqta import to_hocr, validate_hocr
        from PIL import Image

        ocr = _get_model(options)
        result = ocr.read(str(input_file), layout=True, max_new_tokens=_ns(options).max_new_tokens)
        page = result.pages[0]
        if any(re.search(r'[0-9A-Za-z\u0660-\u0669\u06F0-\u06F9]', line.text) for line in page.lines):
            log.warning(
                'Nuqta line contains digits or Latin text; OCRmyPDF 17.13 may extract those '
                'runs reversed in the PDF text layer (line-level word boxes).'
            )
        if page.warnings:
            raise RuntimeError(
                'Nuqta output failed validation for ' + str(input_file) + ': ' + '; '.join(page.warnings)
            )
        with Image.open(input_file) as image:
            dpi_pair = image.info.get('dpi')
        dpi = round(float(dpi_pair[0])) if dpi_pair else None
        # lang="ara" matters to OCRmyPDF: only ISO 639-2 Arabic maps to its
        # Arabic font.  dir="rtl" is for other hOCR consumers and the spec.
        document = to_hocr(result, lang='ara', dpi=dpi, title=Path(input_file).name)
        validate_hocr(document)
        Path(output_hocr).write_text(document, encoding='utf-8')
        Path(output_text).write_text(page.text + '\n', encoding='utf-8')

    @staticmethod
    def generate_pdf(input_file: Path, output_pdf: Path, output_text: Path, options) -> None:
        raise NotImplementedError(
            'ocrmypdf-nuqta supports the hOCR path only; use --pdf-renderer fpdf2 (the default)'
        )


@hookimpl
def get_ocr_engine(options):
    if options is not None and getattr(options, 'ocr_engine', 'auto') not in (None, 'auto'):
        return None
    return NuqtaOcrEngine()
