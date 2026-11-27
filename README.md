# ocrmypdf-nuqta

An **alpha** [OCRmyPDF](https://ocrmypdf.readthedocs.io/) engine plugin for
[Nuqta OCR](https://github.com/SalahAlHaismawi/nuqta-ocr) (Arabic OCR with
faithful tashkeel / diacritics).

Status: **Alpha** (not on PyPI). Supports single-column Arabic pages with
proportional word-level `ocrx_word` hOCR boxes, running in one worker
(`--jobs 1`).

## Supported OCRmyPDF versions

- **Requirement:** `ocrmypdf>=17.13,<18`
- **Verified:** End-to-end run with the real `nuqta-0.8b` model on a Colab T4
  with OCRmyPDF `17.14.2`; unit and e2e tests in the suite use a fake Nuqta.

## Install

Install the plugin together with the optional NumPy/OpenCV layout extra needed
by `nuqta.read_lines`:

```bash
pip install "ocrmypdf-nuqta[layout]"
```

NumPy and OpenCV remain **optional dependencies** (matching upstream
`nuqta-ocr`). Installing without `[layout]`:

```bash
pip install ocrmypdf-nuqta
```

allows importing `ocrmypdf_nuqta` and using its pure-Python word-splitting
module (`ocrmypdf_nuqta.words`) without NumPy/OpenCV, and raises a clear
install hint if OCR line segmentation is invoked without them.

> **Note:** Until `ocrmypdf-nuqta` is published to PyPI, install from a local
> checkout with `pip install -e ".[layout]"`.
>
> **Note on Tesseract:** The `tesseract` binary is still required on `PATH`
> because OCRmyPDF's built-in `tesseract_ocr.check_options` hook checks for it
> unconditionally, even when `--ocr-engine nuqta` is selected.
>
> **First run:** Downloads the selected Nuqta weights from Hugging Face.

## Usage

Once installed, `ocrmypdf-nuqta` is discovered through OCRmyPDF's `ocrmypdf`
entry point (`[project.entry-points."ocrmypdf"]`). **The default engine stays
Tesseract** unless you opt into Nuqta via `--ocr-engine nuqta`, `--nuqta`, or
any `--nuqta-*` option:

```bash
ocrmypdf --ocr-engine nuqta -l ara --nuqta-device cuda in.pdf out.pdf
```

Or with `--nuqta`:

```bash
ocrmypdf --nuqta -l ara --nuqta-device cuda in.pdf out.pdf
```

> Do **not** pass `--plugin ocrmypdf_nuqta` when the package is already
> installed via its entry point: OCRmyPDF loads the entry point first and then
> `pluggy` rejects `--plugin ocrmypdf_nuqta` as already registered.

### Options

```text
  --ocr-engine {auto,tesseract,none,nuqta}
                        OCR engine to use. 'auto' (default) selects the best
                        available engine. 'tesseract' uses Tesseract OCR.
                        'none' skips OCR entirely, useful for PDF/A conversion
                        or image processing without text recognition. 'nuqta'
                        uses Nuqta Arabic OCR.

Nuqta OCR:
  Arabic OCR engine with faithful tashkeel

  --nuqta               Use Nuqta Arabic OCR instead of the default Tesseract
                        engine (or pass --ocr-engine nuqta)
  --nuqta-model NUQTA_MODEL
                        Nuqta model choice (0.8B, 4B, nuqta-0.8b, nuqta-4b),
                        Hugging Face hub id or local path (default:
                        nuqta-0.8b)
  --nuqta-backend {auto,transformers,vllm}
                        Nuqta inference backend (default: auto)
  --nuqta-device {auto,cpu,cuda}
                        Inference device: auto, cpu or cuda (default: auto)
  --nuqta-dtype {auto,bfloat16,float16,float32}
                        Model weight dtype; auto picks what the device
                        supports (default: auto)
  --nuqta-max-new-tokens NUQTA_MAX_NEW_TOKENS
                        token budget per line (default: 1024)
  --nuqta-text-layer {as-written,bare}
                        script written into the invisible text layer (default:
                        as-written)
  --nuqta-line-hocr     emit one ocrx_word per line instead of word-level
                        boxes
```

The plugin forces `--jobs 1` when Nuqta is selected so only one model instance
is loaded per process. When Tesseract or `none` is active, `--jobs` is left
unchanged.

## Colab / GPU check script

Run `scripts/colab_check.sh` on a GPU machine (such as Google Colab) to install
dependencies, run OCRmyPDF with the Nuqta engine, time the run, and extract
text with `pdftotext` and PyMuPDF:

```bash
bash scripts/colab_check.sh book.pdf out.pdf nuqta-0.8b
```

## Architecture

- `ocrmypdf_nuqta/__init__.py`: OCRmyPDF plugin hooks (`register_options`,
  `add_options`, `check_options`, `get_ocr_engine`) and `NuqtaOcrEngine`
  (`version`, `creator_tag`, `languages`, `get_orientation`, `get_deskew`,
  `generate_hocr`, `generate_pdf`).
- `ocrmypdf_nuqta/words.py`: Pure-Python proportional word-box splitter and
  word-level hOCR builder, kept outside core Nuqta inference.

## Speed (CPU vs GPU)

- **CPU:** Very slow. Recorded on an AMD Ryzen 5 2400G (4 cores / 8 threads,
  15 GB RAM, no GPU) with Nuqta 0.8B (`transformers`, `bfloat16`): roughly
  **350 s** for 8 lines (~44 s/line) on a 1024×620 crop — see
  [MEASUREMENTS.md](MEASUREMENTS.md). The plugin logs a warning on CPU.
- **GPU:** Use `--nuqta-device cuda` on a CUDA machine for practical throughput.

## Known limitations

- **Alpha release:** Not on PyPI.
- **Untested configurations:** The `4B` model and the `vllm` backend are
  untested through the plugin.
- **Single-column pages only:** `nuqta.read_lines` uses a horizontal projection
  profile; multi-column pages are read across columns rather than down.
- **Proportional word boxes:** Word boxes (`ocrx_word`) are proportional splits
  of line boxes by character width, not detected word bounding boxes.
- **Arabic-Indic digits and mixed lines:** Arabic-Indic digit runs and mixed
  Arabic/digit/Latin lines may be extracted in a different order depending on
  the PDF tool or viewer.
- **Diacritic-sensitive search:** PDF viewers match vocalized queries against
  vocalized text layers; unvocalized queries do not match vocalized text.
- **Renderer support:** `--pdf-renderer sandwich` is not supported; use the
  default `fpdf2` hOCR renderer.

## License

Apache-2.0, matching Nuqta OCR.
