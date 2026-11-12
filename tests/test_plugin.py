from __future__ import annotations

import sys
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import ocrmypdf_nuqta as plugin
import pytest
from nuqta import Page, Result
from ocrmypdf_nuqta import NuqtaOcrEngine, NuqtaOptions
from PIL import Image, ImageDraw

XHTML = '{http://www.w3.org/1999/xhtml}'


def text_image() -> Image.Image:
    image = Image.new('RGB', (600, 800), 'white')
    draw = ImageDraw.Draw(image)
    for y in (100, 200, 300):
        for x in range(80, 520, 20):
            draw.rectangle((x, y, x + 8, y + 30), fill='black')
    return image


class FakeNuqta:
    def __init__(self):
        self.calls = []

    def read_many(self, srcs, mode='as-written', max_new_tokens=1024):
        self.calls.append((len(srcs), mode, max_new_tokens))
        return [Result([Page(f'سطر {index + 1}')], mode) for index in range(len(srcs))]


@pytest.fixture(autouse=True)
def clear_model_cache():
    plugin._MODELS.clear()
    yield
    plugin._MODELS.clear()


def options(**overrides):
    base = {'ocr_engine': 'auto', 'jobs': 0, 'nuqta': NuqtaOptions()}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_engine_selection():
    assert isinstance(plugin.get_ocr_engine(options()), NuqtaOcrEngine)
    assert plugin.get_ocr_engine(options(ocr_engine='tesseract')) is None
    assert plugin.get_ocr_engine(options(ocr_engine='none')) is None


def test_check_options_forces_one_worker():
    opts = options(jobs=4)
    plugin.check_options(opts)
    assert opts.jobs == 1
    auto = options(jobs=None)
    plugin.check_options(auto)
    assert auto.jobs == 1
    tess = options(jobs=4, ocr_engine='tesseract')
    plugin.check_options(tess)
    assert tess.jobs == 4


def test_languages_are_permissive_for_the_default_eng():
    assert {'ara', 'eng'} <= NuqtaOcrEngine.languages(options())


def test_generate_hocr_writes_a_valid_text_layer(tmp_path, monkeypatch):
    image_path = tmp_path / 'page.png'
    text_image().save(image_path, dpi=(300, 300))
    fake = FakeNuqta()
    monkeypatch.setattr(plugin, '_load_ocr', lambda opts: fake)
    hocr_path, text_path = tmp_path / 'page.hocr', tmp_path / 'page.txt'

    NuqtaOcrEngine.generate_hocr(
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
    assert text_path.read_text(encoding='utf-8') == 'سطر 1\nسطر 2\nسطر 3\n'
    assert fake.calls == [(3, 'as-written', 1024)]


def test_generate_hocr_reuses_one_model(tmp_path, monkeypatch):
    image_path = tmp_path / 'page.png'
    text_image().save(image_path, dpi=(300, 300))
    loads = []
    fake = FakeNuqta()

    def load(opts):
        loads.append(opts)
        return fake

    monkeypatch.setattr(plugin, '_load_ocr', load)
    for number in range(2):
        NuqtaOcrEngine.generate_hocr(
            input_file=image_path,
            output_hocr=tmp_path / f'{number}.hocr',
            output_text=tmp_path / f'{number}.txt',
            options=options(),
        )
    assert len(loads) == 1
    assert len(fake.calls) == 2


def test_sandwich_renderer_is_rejected(tmp_path):
    with pytest.raises(NotImplementedError, match='fpdf2'):
        NuqtaOcrEngine.generate_pdf(
            input_file=tmp_path / 'page.png',
            output_pdf=tmp_path / 'page.pdf',
            output_text=tmp_path / 'page.txt',
            options=options(),
        )


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
    opts = SimpleNamespace(nuqta=NuqtaOptions(device='cuda', backend='transformers'))
    plugin._load_ocr(opts)
    assert seen['dtype'] == 'float16'      # bf16 not supported -> float16
    assert seen['device_map'] == 'cuda:0'
