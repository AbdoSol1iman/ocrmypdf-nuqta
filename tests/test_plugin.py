from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import img2pdf
import ocrmypdf
import pytest
from nuqta import Page, Result
from ocrmypdf._plugin_manager import get_plugin_manager
from ocrmypdf.api import setup_plugin_infrastructure
from ocrmypdf.builtin_plugins.tesseract_ocr import TesseractOcrEngine
from ocrmypdf.cli import get_options_and_plugins, get_parser
from ocrmypdf.exceptions import MissingDependencyError
from ocrmypdf.font import MultiFontManager
from ocrmypdf.fpdf_renderer import Fpdf2PdfRenderer
from ocrmypdf.hocrtransform import HocrParser
from PIL import Image, ImageDraw
from pydantic import ValidationError

import ocrmypdf_nuqta as plugin

XHTML = '{http://www.w3.org/1999/xhtml}'


def text_image(line_ys: tuple[int, ...] = (100, 200, 300), size: tuple[int, int] = (600, 800)) -> Image.Image:
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    for y in line_ys:
        for x in range(80, size[0] - 80, 20):
            draw.rectangle((x, y, x + 8, y + 30), fill='black')
    return image


class FakeNuqta:
    def __init__(self, texts: list[str] | None = None):
        self.calls: list[tuple[int, str, int]] = []
        self._texts = texts

    def read_many(self, srcs, mode='as-written', max_new_tokens=1024):
        self.calls.append((len(srcs), mode, max_new_tokens))
        return [
            Result(
                [
                    Page(
                        text=self._texts[index] if self._texts else f'سَطْرٌ عَرَبِيٌّ {index + 1}',
                    )
                ],
                mode,
            )
            for index in range(len(srcs))
        ]


@pytest.fixture(autouse=True)
def clear_model_cache():
    plugin._MODELS.clear()
    yield
    plugin._MODELS.clear()


def options(**overrides):
    base = {'ocr_engine': 'auto', 'jobs': 0, 'nuqta': plugin.NuqtaOptions(enabled=True)}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_default_engine_stays_tesseract_when_nuqta_not_requested():
    opts_default, pm = get_options_and_plugins(['in.pdf', 'out.pdf'])
    assert isinstance(pm.get_ocr_engine(options=opts_default), TesseractOcrEngine)
    assert plugin.get_ocr_engine(opts_default) is None
    opts_default.jobs = 4
    plugin.check_options(opts_default)
    assert opts_default.jobs == 4


def test_entrypoint_auto_discovery_and_opt_in_selection():
    pm = get_plugin_manager()
    # Explicit --ocr-engine nuqta
    opts_engine, pm2 = get_options_and_plugins(['--ocr-engine', 'nuqta', 'in.pdf', 'out.pdf'])
    assert isinstance(pm2.get_ocr_engine(options=opts_engine), plugin.NuqtaOcrEngine)

    # Convenience --nuqta flag or any --nuqta-* option selects Nuqta
    opts_flag, _ = get_options_and_plugins(['--nuqta', 'in.pdf', 'out.pdf'])
    assert isinstance(pm.get_ocr_engine(options=opts_flag), plugin.NuqtaOcrEngine)
    opts_model, _ = get_options_and_plugins(['--nuqta-model', '4B', 'in.pdf', 'out.pdf'])
    assert isinstance(pm.get_ocr_engine(options=opts_model), plugin.NuqtaOcrEngine)

    # Explicit --ocr-engine tesseract or none always wins even if --nuqta-* is passed
    opts_tess, _ = get_options_and_plugins(
        ['--ocr-engine', 'tesseract', '--nuqta-model', '4B', 'in.pdf', 'out.pdf']
    )
    assert isinstance(pm.get_ocr_engine(options=opts_tess), TesseractOcrEngine)
    assert plugin.get_ocr_engine(options(ocr_engine='tesseract')) is None
    assert plugin.get_ocr_engine(options(ocr_engine='none')) is None
    assert plugin.get_ocr_engine(None) is None


def test_options_model_aliases_and_cli_flags():
    assert plugin.NuqtaOptions(model='0.8B').model == 'nuqta-0.8b'
    assert plugin.NuqtaOptions(model='4B').model == 'nuqta-4b'
    assert plugin.NuqtaOptions(model='SalahALHaismawi/Nuqta-Arabic-OCR-4B').model == (
        'SalahALHaismawi/Nuqta-Arabic-OCR-4B'
    )
    with pytest.raises(ValidationError):
        plugin.NuqtaOptions(model='   ')

    pm = setup_plugin_infrastructure()
    parser = get_parser()
    pm.add_options(parser=parser)
    args = parser.parse_args(
        ['--nuqta-model', '4B', '--nuqta-device', 'cpu', '--nuqta-backend', 'transformers', 'in.pdf', 'out.pdf']
    )
    assert args.nuqta_model == '4B'
    assert args.nuqta_device == 'cpu'
    assert args.nuqta_backend == 'transformers'


def test_check_options_forces_one_worker_and_validates_device(monkeypatch):
    opts = options(jobs=4)
    plugin.check_options(opts)
    assert opts.jobs == 1

    auto = options(jobs=None)
    plugin.check_options(auto)
    assert auto.jobs == 1

    tess = options(jobs=4, ocr_engine='tesseract', nuqta=plugin.NuqtaOptions(enabled=True, device='cuda'))
    plugin.check_options(tess)
    assert tess.jobs == 4

    import torch

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(MissingDependencyError, match='CUDA'):
        plugin.check_options(options(nuqta=plugin.NuqtaOptions(enabled=True, device='cuda')))

    monkeypatch.setattr(plugin.importlib.util, 'find_spec', lambda name: None if name == 'vllm' else object())
    with pytest.raises(MissingDependencyError, match='vllm'):
        plugin.check_options(options(nuqta=plugin.NuqtaOptions(enabled=True, backend='vllm')))


def test_engine_metadata_and_languages():
    opts = options(nuqta=plugin.NuqtaOptions(model='4B'))
    engine = plugin.NuqtaOcrEngine()
    assert 'ocrmypdf-nuqta' in engine.version()
    assert 'nuqta-4b' in engine.creator_tag(opts)
    assert '\n' not in engine.creator_tag(opts)
    assert 'Nuqta' in str(engine)
    assert {'ara', 'eng'} <= engine.languages(opts)
    assert engine.get_orientation(Path('dummy.png'), opts) == plugin.OrientationConfidence(0, 0.0)
    assert engine.get_deskew(Path('dummy.png'), opts) == 0.0


def test_generate_hocr_writes_word_level_boxes(tmp_path, monkeypatch):
    pytest.importorskip('cv2')
    image_path = tmp_path / 'page.png'
    text_image().save(image_path, dpi=(300, 300))
    fake = FakeNuqta(
        [
            'أَمْرَاضُ القُلُوبِ ٢٤',
            'مَرْحَبًا Hello World بِكُمْ',
            'سَنَةُ 2024 مِيلَادِيَّةٌ',
        ]
    )
    monkeypatch.setattr(plugin, '_load_ocr', lambda opts: fake)
    hocr_path, text_path = tmp_path / 'page.hocr', tmp_path / 'page.txt'

    plugin.NuqtaOcrEngine.generate_hocr(
        input_file=image_path, output_hocr=hocr_path, output_text=text_path, options=options()
    )

    document = hocr_path.read_text(encoding='utf-8')
    root = ET.fromstring(document)
    page = root.find(f'.//{XHTML}div[@class="ocr_page"]')
    assert 'bbox 0 0 600 800' in page.get('title')
    assert 'scan_res 300 300' in page.get('title')
    lines = root.findall(f'.//{XHTML}span[@class="ocr_line"]')
    assert len(lines) == 3
    assert all(line.get('lang') == 'ara' and line.get('dir') == 'rtl' for line in lines)

    words_line1 = lines[0].findall(f'.//{XHTML}span[@class="ocrx_word"]')
    words_line2 = lines[1].findall(f'.//{XHTML}span[@class="ocrx_word"]')
    assert [w.text for w in words_line1] == ['أَمْرَاضُ', 'القُلُوبِ', '٢٤']
    assert [w.text for w in words_line2] == ['مَرْحَبًا', 'Hello', 'World', 'بِكُمْ']
    assert text_path.read_text(encoding='utf-8') == (
        'أَمْرَاضُ القُلُوبِ ٢٤\nمَرْحَبًا Hello World بِكُمْ\nسَنَةُ 2024 مِيلَادِيَّةٌ\n'
    )
    assert fake.calls == [(3, 'as-written', 1024)]


def test_generate_hocr_reuses_one_model(tmp_path, monkeypatch):
    pytest.importorskip('cv2')
    image_path = tmp_path / 'page.png'
    text_image().save(image_path, dpi=(300, 300))
    loads = []
    fake = FakeNuqta()

    def load(opts):
        loads.append(opts)
        return fake

    monkeypatch.setattr(plugin, '_load_ocr', load)
    for number in range(2):
        plugin.NuqtaOcrEngine.generate_hocr(
            input_file=image_path,
            output_hocr=tmp_path / f'{number}.hocr',
            output_text=tmp_path / f'{number}.txt',
            options=options(),
        )
    assert len(loads) == 1
    assert len(fake.calls) == 2


def test_generate_hocr_missing_layout_deps_error(tmp_path, monkeypatch):
    image_path = tmp_path / 'page.png'
    text_image().save(image_path, dpi=(200, 200))
    monkeypatch.setattr(plugin, '_load_ocr', lambda opts: FakeNuqta())
    monkeypatch.setitem(sys.modules, 'cv2', None)

    with pytest.raises(MissingDependencyError, match='ocrmypdf-nuqta\\[layout\\]'):
        plugin.NuqtaOcrEngine.generate_hocr(
            input_file=image_path,
            output_hocr=tmp_path / 'page.hocr',
            output_text=tmp_path / 'page.txt',
            options=options(),
        )


def test_short_crop_regression_keeps_all_lines(tmp_path, monkeypatch):
    """Regression test for PR #3 short-crop bug: 1- and 2-line crops must not lose lines."""
    pytest.importorskip('cv2')
    from nuqta import segment_page

    full = text_image(line_ys=(100, 200, 300), size=(600, 800))
    for count in (1, 2):
        plugin._MODELS.clear()
        crop = full.crop((0, 80, 600, 80 + 100 * count))
        segmented = segment_page(crop)
        assert len(segmented.boxes) == count

        crop_path = tmp_path / f'crop_{count}.png'
        crop.save(crop_path, dpi=(200, 200))
        fake = FakeNuqta([f'سَطْرٌ {i + 1}' for i in range(count)])
        monkeypatch.setattr(plugin, '_load_ocr', lambda opts, f=fake: f)
        hocr_path = tmp_path / f'crop_{count}.hocr'
        text_path = tmp_path / f'crop_{count}.txt'

        plugin.NuqtaOcrEngine.generate_hocr(
            input_file=crop_path,
            output_hocr=hocr_path,
            output_text=text_path,
            options=options(),
        )
        root = ET.fromstring(hocr_path.read_text(encoding='utf-8'))
        lines = root.findall(f'.//{XHTML}span[@class="ocr_line"]')
        assert len(lines) == count


def test_sandwich_renderer_is_rejected(tmp_path):
    with pytest.raises(NotImplementedError, match='fpdf2'):
        plugin.NuqtaOcrEngine.generate_pdf(
            input_file=tmp_path / 'page.png',
            output_pdf=tmp_path / 'page.pdf',
            output_text=tmp_path / 'page.txt',
            options=options(),
        )


def _strip_bidi_controls(text: str) -> str:
    return ''.join(ch for ch in text if ch not in '\u202a\u202b\u202c\u200e\u200f')


def _render_e2e_pdfs(tmp_path, monkeypatch, expected_lines: list[str]) -> tuple[str, str]:
    sample_png = tmp_path / 'book_page.png'
    text_image(line_ys=(120, 240, 360, 480), size=(1000, 800)).save(sample_png, dpi=(200, 200))
    sample_pdf = tmp_path / 'book.pdf'
    sample_pdf.write_bytes(img2pdf.convert(str(sample_png)))

    monkeypatch.setattr(plugin, '_load_ocr', lambda opts: FakeNuqta(expected_lines))

    # Render directly via OCRmyPDF's hOCR parser + fpdf2 renderer (always works without tesseract).
    hocr_path = tmp_path / 'direct.hocr'
    text_path = tmp_path / 'direct.txt'
    plugin.NuqtaOcrEngine.generate_hocr(sample_png, hocr_path, text_path, options())
    direct_pdf = tmp_path / 'direct.pdf'
    Fpdf2PdfRenderer(
        HocrParser(hocr_path).parse(),
        dpi=200,
        multi_font_manager=MultiFontManager(),
        invisible_text=True,
    ).render(direct_pdf)
    extracted = _strip_bidi_controls(
        subprocess.check_output(['pdftotext', str(direct_pdf), '-']).decode('utf-8')
    )

    # Full ocrmypdf.ocr pipeline test; bypass tesseract binary check only if tesseract is absent.
    if shutil.which('tesseract') is None:
        from ocrmypdf.builtin_plugins import tesseract_ocr

        monkeypatch.setattr(tesseract_ocr, 'check_external_program', lambda **kw: None)
        monkeypatch.setattr(
            tesseract_ocr.tesseract, 'version', lambda: tesseract_ocr.tesseract.TesseractVersion('5.3.0')
        )
        monkeypatch.setattr(tesseract_ocr.tesseract, 'has_thresholding', lambda: True)

    out_pdf = tmp_path / 'out.pdf'
    exit_code = ocrmypdf.ocr(
        sample_pdf,
        out_pdf,
        ocr_engine='nuqta',
        language=['ara'],
        output_type='pdf',
        optimize=0,
        progress_bar=False,
        jobs=1,
    )
    assert exit_code == ocrmypdf.ExitCode.ok
    full_extracted = _strip_bidi_controls(
        subprocess.check_output(['pdftotext', str(out_pdf), '-']).decode('utf-8')
    )
    return extracted, full_extracted


_E2E_EXPECTED_LINES = [
    'أَمْرَاضُ القُلُوبِ وَشِفَاؤُهَا ٢٤',
    'الْمُتَنافِسُونَ آيَةُ ٢٢ - ٢٦ فَأَمَرَ',
    'مَرْحَبًا Hello World بِكُمْ',
    'صَدَرَ الكِتَابُ سَنَةَ 2024 مِيلَادِيَّةً',
]


def test_end_to_end_pdf_text_layer_extraction(tmp_path, monkeypatch):
    """End-to-end sample PDF test: generate searchable PDF and verify pdftotext output."""
    pytest.importorskip('cv2')
    if shutil.which('pdftotext') is None:
        pytest.skip('pdftotext binary not available')

    extracted, full_extracted = _render_e2e_pdfs(tmp_path, monkeypatch, _E2E_EXPECTED_LINES)
    for text in (extracted, full_extracted):
        assert 'أَمْرَاضُ القُلُوبِ وَشِفَاؤُهَا' in text
        assert 'الْمُتَنافِسُونَ آيَةُ' in text
        assert 'فَأَمَرَ' in text
        assert 'مَرْحَبًا' in text
        assert 'Hello World' in text
        assert 'بِكُمْ' in text
        assert 'صَدَرَ الكِتَابُ سَنَةَ' in text
        assert '2024' in text
        assert 'مِيلَادِيَّةً' in text
        assert 'olleH' not in text


@pytest.mark.xfail(
    reason=(
        'OCRmyPDF >=17.14 fpdf2 renderer reverses Arabic-Indic digit-only words '
        'in the content stream; extractors disagree'
    ),
    strict=False,
)
def test_end_to_end_arabic_indic_digit_order(tmp_path, monkeypatch):
    """Verify Arabic-Indic digit-only word extraction order across OCRmyPDF versions."""
    pytest.importorskip('cv2')
    if shutil.which('pdftotext') is None:
        pytest.skip('pdftotext binary not available')

    extracted, full_extracted = _render_e2e_pdfs(tmp_path, monkeypatch, _E2E_EXPECTED_LINES)
    for text in (extracted, full_extracted):
        assert 'أَمْرَاضُ القُلُوبِ وَشِفَاؤُهَا ٢٤' in text
        assert '٢٢ - ٢٦' in text
        assert '٤٢' not in text


# --- dtype auto-detection (monkeypatched torch; no GPU required) ----------

class _FakeCuda:
    """Stands in for torch.cuda in the dtype/device detection logic."""

    def __init__(self, available=True, bf16=True):
        self._available = available
        self._bf16 = bf16
        self.calls = []

    def is_available(self):
        return self._available

    def is_bf16_supported(self):
        self.calls.append('bf16')
        if self._bf16:
            return True
        raise RuntimeError('no bf16')


class _FakeTorch:
    def __init__(self, cuda):
        self.cuda = cuda


def test_dtype_auto_picks_bfloat16_when_supported(monkeypatch):
    cuda = _FakeCuda(available=True, bf16=True)
    monkeypatch.setitem(sys.modules, 'torch', _FakeTorch(cuda))
    assert plugin._resolve_dtype('auto', 'cuda') == 'bfloat16'


def test_dtype_auto_falls_back_to_float16(monkeypatch):
    cuda = _FakeCuda(available=True, bf16=False)
    monkeypatch.setitem(sys.modules, 'torch', _FakeTorch(cuda))
    assert plugin._resolve_dtype('auto', 'cuda') == 'float16'


def test_dtype_auto_float32_on_cpu(monkeypatch):
    cuda = _FakeCuda(available=False)
    monkeypatch.setitem(sys.modules, 'torch', _FakeTorch(cuda))
    assert plugin._resolve_dtype('auto', 'cpu') == 'float32'


def test_dtype_auto_without_torch_is_float32(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    assert plugin._resolve_dtype('auto', 'cpu') == 'float32'


def test_explicit_dtype_overrides_auto(monkeypatch):
    cuda = _FakeCuda(available=True, bf16=True)
    monkeypatch.setitem(sys.modules, 'torch', _FakeTorch(cuda))
    assert plugin._resolve_dtype('float16', 'cuda') == 'float16'
    assert plugin._resolve_dtype('bfloat16', 'cpu') == 'bfloat16'


def test_dtype_auto_handles_cuda_runtime_error(monkeypatch):
    class Broken:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported():
            raise RuntimeError('driver gone')

    monkeypatch.setitem(sys.modules, 'torch', _FakeTorch(Broken()))
    assert plugin._resolve_dtype('auto', 'cuda') == 'float16'


def test_load_ocr_passes_resolved_dtype(monkeypatch):
    cuda = _FakeCuda(available=True, bf16=False)
    monkeypatch.setitem(sys.modules, 'torch', _FakeTorch(cuda))
    seen = {}

    class FakeOCR:
        def __init__(self, model, backend='transformers', **kwargs):
            seen.update(kwargs, model=model, backend=backend)

    monkeypatch.setattr(plugin, '_CPU_WARNED', True)
    import nuqta
    monkeypatch.setattr(nuqta, 'OCR', FakeOCR)
    opts = SimpleNamespace(nuqta=plugin.NuqtaOptions(device='cuda', backend='transformers'))
    plugin._load_ocr(opts)
    assert seen['dtype'] == 'float16'      # bf16 not supported -> float16
    assert seen['device_map'] == 'cuda:0'
