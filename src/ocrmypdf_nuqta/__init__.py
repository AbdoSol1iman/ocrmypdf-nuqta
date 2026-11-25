"""OCRmyPDF plugin: Nuqta Arabic OCR with faithful tashkeel and word-level hOCR.

OCRmyPDF 17.x hands the engine one PNG per page and expects hOCR whose boxes
are in pixels of that image.  Nuqta segments the page into lines, reads each
line, and writes the boxes back in the original image frame; :mod:`.words`
splits each line box into proportional ``ocrx_word`` elements with RTL/mixed
direction handling and emits the hOCR shape OCRmyPDF's parser and fpdf2
renderer require.

Install the package and OCRmyPDF keeps Tesseract as the default engine unless
the user selects Nuqta via ``--ocr-engine nuqta``, ``--nuqta``, or any
``--nuqta-*`` option.  The model is loaded once per process, lazily, behind a
lock.  Because one model per worker is not affordable, the plugin forces
``--jobs 1`` (one process, one model) when Nuqta is selected.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import threading
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

from ocrmypdf import hookimpl
from ocrmypdf.exceptions import MissingDependencyError
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence
from pydantic import BaseModel, Field, field_validator

from .words import build_hocr

log = logging.getLogger(__name__)

__version__ = '0.1.0'
DEFAULT_MODEL = 'nuqta-0.8b'

MODEL_ALIASES: dict[str, str] = {
    '0.8b': 'nuqta-0.8b',
    'nuqta-0.8b': 'nuqta-0.8b',
    '4b': 'nuqta-4b',
    'nuqta-4b': 'nuqta-4b',
}

_MODELS: dict[tuple[str, str, str, str], object] = {}
_MODEL_LOCK = threading.Lock()
_CPU_WARNED = False


class NuqtaOptions(BaseModel):
    """Options exposed as ``--nuqta`` / ``--nuqta-*`` on the OCRmyPDF command line."""

    enabled: bool = Field(
        default=False,
        description='Select Nuqta OCR instead of the default Tesseract engine',
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description='Nuqta model name (0.8B / 4B / nuqta-0.8b / nuqta-4b), hub id or local path',
    )
    backend: Literal['auto', 'transformers', 'vllm'] = Field(default='auto')
    device: Literal['auto', 'cpu', 'cuda'] = Field(default='auto')
    dtype: Literal['bfloat16', 'float16', 'float32'] = Field(default='bfloat16')
    max_new_tokens: int = Field(default=1024, ge=64, le=8192)

    @field_validator('model')
    @classmethod
    def _normalize_model(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError('model must not be empty')
        return MODEL_ALIASES.get(stripped.lower(), stripped)


_NUQTA_TRIGGER_ATTRS = (
    'nuqta_enabled',
    'nuqta_model',
    'nuqta_backend',
    'nuqta_device',
    'nuqta_dtype',
    'nuqta_max_new_tokens',
)


def _is_nuqta_selected(options) -> bool:
    """Return True only when the user opted into Nuqta (keep Tesseract default)."""
    if options is None:
        return False
    ocr_engine = getattr(options, 'ocr_engine', 'auto')
    if ocr_engine == 'nuqta':
        return True
    if ocr_engine not in (None, 'auto'):
        return False
    extra = getattr(options, 'extra_attrs', None)
    if isinstance(extra, dict):
        if extra.get('nuqta_enabled'):
            return True
        if any(extra.get(attr) is not None for attr in _NUQTA_TRIGGER_ATTRS[1:]):
            return True
    if getattr(options, 'nuqta_enabled', False):
        return True
    namespace = getattr(options, 'nuqta', None)
    return bool(namespace is not None and getattr(namespace, 'enabled', False))


def _ns(options) -> NuqtaOptions:
    """Return the plugin options namespace, with defaults for plain namespaces."""
    namespace = getattr(options, 'nuqta', None)
    return namespace if namespace is not None else NuqtaOptions()


@hookimpl
def register_options() -> dict[str, type[BaseModel]]:
    return {'nuqta': NuqtaOptions}


@hookimpl
def add_options(parser: ArgumentParser) -> None:
    for action in getattr(parser, '_actions', ()):
        if (
            getattr(action, 'dest', None) == 'ocr_engine'
            and isinstance(getattr(action, 'choices', None), list)
            and 'nuqta' not in action.choices
        ):
            action.choices.append('nuqta')
            if action.help:
                action.help += " 'nuqta' uses Nuqta Arabic OCR."

    group = parser.add_argument_group('Nuqta OCR', 'Arabic OCR engine with faithful tashkeel')
    group.add_argument(
        '--nuqta',
        dest='nuqta_enabled',
        action='store_true',
        default=None,
        help='Use Nuqta Arabic OCR instead of the default Tesseract engine (or pass --ocr-engine nuqta)',
    )
    group.add_argument(
        '--nuqta-model',
        dest='nuqta_model',
        default=None,
        help=(
            'Nuqta model choice (0.8B, 4B, nuqta-0.8b, nuqta-4b), '
            f'Hugging Face hub id or local path (default: {DEFAULT_MODEL})'
        ),
    )
    group.add_argument(
        '--nuqta-backend',
        dest='nuqta_backend',
        default=None,
        choices=['auto', 'transformers', 'vllm'],
        help='Nuqta inference backend (default: auto)',
    )
    group.add_argument(
        '--nuqta-device',
        dest='nuqta_device',
        default=None,
        choices=['auto', 'cpu', 'cuda'],
        help='Inference device: auto, cpu or cuda (default: auto)',
    )
    group.add_argument(
        '--nuqta-dtype',
        dest='nuqta_dtype',
        default=None,
        choices=['bfloat16', 'float16', 'float32'],
        help='Model weight dtype (default: bfloat16)',
    )
    group.add_argument(
        '--nuqta-max-new-tokens',
        dest='nuqta_max_new_tokens',
        type=int,
        default=None,
        help='token budget per line (default: 1024)',
    )


@hookimpl
def check_options(options) -> None:
    """Validate Nuqta options and keep one worker so only one model loads."""
    if not _is_nuqta_selected(options):
        return  # Tesseract or none stays active; do not alter jobs or require CUDA/vLLM

    ns = _ns(options)

    if ns.device == 'cuda':
        try:
            import torch
        except ImportError as exc:
            raise MissingDependencyError(
                '--nuqta-device cuda requested, but PyTorch is not installed'
            ) from exc
        if not torch.cuda.is_available():
            raise MissingDependencyError(
                '--nuqta-device cuda requested, but CUDA is not available on this machine; '
                'omit --nuqta-device or pass --nuqta-device cpu'
            )

    if ns.backend == 'vllm' and importlib.util.find_spec('vllm') is None:
        raise MissingDependencyError(
            '--nuqta-backend vllm requested, but vllm is not installed; '
            'install it with `pip install "nuqta-ocr[vllm]"`'
        )

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
            backend = 'vllm' if importlib.util.find_spec('vllm') is not None else 'transformers'
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


def _nuqta_version() -> str:
    try:
        return importlib.metadata.version('nuqta-ocr')
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return 'unknown'


class NuqtaOcrEngine(OcrEngine):
    @staticmethod
    def version() -> str:
        return f'ocrmypdf-nuqta {__version__} (nuqta-ocr {_nuqta_version()})'

    @staticmethod
    def creator_tag(options) -> str:
        return f'Nuqta OCR {__version__} ({_ns(options).model})'

    def __str__(self) -> str:
        return f'Nuqta OCR {__version__}'

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
        from nuqta import read_lines, validate_hocr
        from PIL import Image

        ocr = _get_model(options)
        try:
            result = read_lines(
                ocr,
                [str(input_file)],
                mode='as-written',
                max_new_tokens=_ns(options).max_new_tokens,
            )[0]
        except ImportError as exc:
            raise MissingDependencyError(
                'Nuqta line segmentation requires NumPy and OpenCV; '
                'install them with `pip install "ocrmypdf-nuqta[layout]"`'
            ) from exc

        page = result.pages[0]
        with Image.open(input_file) as image:
            dpi_pair = image.info.get('dpi')
        dpi = round(float(dpi_pair[0])) if dpi_pair else None
        # lang="ara" matters to OCRmyPDF: only ISO 639-2 Arabic maps to its
        # Arabic font.  dir="rtl" is for other hOCR consumers and the spec.
        document = build_hocr(result, lang='ara', dpi=dpi, title=Path(input_file).name)
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
    if not _is_nuqta_selected(options):
        return None
    return NuqtaOcrEngine()
