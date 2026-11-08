from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
from PIL import Image

import ocrmypdf_nuqta as plugin
from nuqta import Line, Page, Result
from ocrmypdf_nuqta import NuqtaOcrEngine, NuqtaOptions

XHTML = '{http://www.w3.org/1999/xhtml}'


class FakeNuqta:
    def __init__(self, warnings=None):
        self.calls = []
        self.warnings = warnings or []

    def read(self, source, layout=False, max_new_tokens=1024):
        self.calls.append((source, layout, max_new_tokens))
        page = Page(
            'سطرٌ واحد',
            lines=[Line('سطرٌ واحد', (10, 20, 190, 60))],
            width=200,
            height=300,
            dpi=300,
            warnings=self.warnings,
        )
        return Result([page], 'as-written')


@pytest.fixture(autouse=True)
def clear_model_cache(monkeypatch):
    plugin._MODELS.clear()
    yield
    plugin._MODELS.clear()


def options(**overrides):
    base = dict(ocr_engine='auto', jobs=0, nuqta=NuqtaOptions())
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
    # explicit Tesseract is left alone
    tess = options(jobs=4, ocr_engine='tesseract')
    plugin.check_options(tess)
    assert tess.jobs == 4


def test_languages_are_permissive_for_the_default_eng():
    langs = NuqtaOcrEngine.languages(options())
    assert {'ara', 'eng'} <= langs


def test_generate_hocr_writes_a_valid_text_layer(tmp_path, monkeypatch):
    image_path = tmp_path / 'page.png'
    Image.new('RGB', (200, 300), 'white').save(image_path, dpi=(300, 300))
    fake = FakeNuqta()
    monkeypatch.setattr(plugin, '_load_ocr', lambda opts: fake)
    hocr_path, text_path = tmp_path / 'page.hocr', tmp_path / 'page.txt'

    NuqtaOcrEngine.generate_hocr(
        input_file=image_path, output_hocr=hocr_path, output_text=text_path, options=options()
    )

    document = hocr_path.read_text(encoding='utf-8')
    root = ET.fromstring(document)
    page = root.find(f'.//{XHTML}div[@class="ocr_page"]')
    assert 'bbox 0 0 200 300' in page.get('title')
    assert 'scan_res 300 300' in page.get('title')
    line = root.find(f'.//{XHTML}span[@class="ocr_line"]')
    assert line.get('lang') == 'ara' and line.get('dir') == 'rtl'
    assert root.find(f'.//{XHTML}span[@class="ocrx_word"]').text == 'سطرٌ واحد'
    assert text_path.read_text(encoding='utf-8') == 'سطرٌ واحد\n'
    assert fake.calls == [(str(image_path), True, 1024)]


def test_generate_hocr_reuses_one_model(tmp_path, monkeypatch):
    image_path = tmp_path / 'page.png'
    Image.new('RGB', (200, 300), 'white').save(image_path, dpi=(300, 300))
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


def test_bad_output_fails_loudly(tmp_path, monkeypatch):
    image_path = tmp_path / 'page.png'
    Image.new('RGB', (200, 300), 'white').save(image_path, dpi=(300, 300))
    monkeypatch.setattr(plugin, '_load_ocr', lambda opts: FakeNuqta(warnings=['line 1: counting sequence']))
    with pytest.raises(RuntimeError, match='counting sequence'):
        NuqtaOcrEngine.generate_hocr(
            input_file=image_path,
            output_hocr=tmp_path / 'page.hocr',
            output_text=tmp_path / 'page.txt',
            options=options(),
        )


def test_sandwich_renderer_is_rejected(tmp_path):
    with pytest.raises(NotImplementedError, match='fpdf2'):
        NuqtaOcrEngine.generate_pdf(
            input_file=tmp_path / 'page.png',
            output_pdf=tmp_path / 'page.pdf',
            output_text=tmp_path / 'page.txt',
            options=options(),
        )
