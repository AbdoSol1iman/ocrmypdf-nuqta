# ocrmypdf-nuqta

An **alpha** [OCRmyPDF](https://ocrmypdf.readthedocs.io/) engine plugin for
[Nuqta OCR](https://github.com/SalahAlHaismawi/nuqta-ocr) (Arabic OCR with
faithful tashkeel / diacritics).

Status: **Alpha** (not yet published on PyPI). Supports single-column Arabic
pages with proportional word-level `ocrx_word` hOCR boxes and logical-order
mixed Arabic/digits/Latin handling, running in one worker (`--jobs 1`).

## Supported OCRmyPDF versions

- **Minimum supported version:** **OCRmyPDF 17.13.0** (`ocrmypdf>=17.13,<18`).
  Tested on `17.13.0` (passes all tests). Earlier 17.0.0 was tested and failed
  end-to-end Arabic `pdftotext` extraction (broken RTL word splitting/shaping
  in `17.0.0`'s `fpdf2` renderer).

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
> **Note on Tesseract:** Even when `--ocr-engine nuqta` is selected (so
> `NuqtaOcrEngine` handles `generate_hocr`, `get_orientation`, and `get_deskew`
> without calling Tesseract), OCRmyPDF `17.13.0`'s built-in
> `tesseract_ocr.check_options` hook still checks for a `tesseract` binary on
> `PATH` unconditionally. Ensure `tesseract` is installed or on `PATH`.
>
> **First run:** Downloads the selected Nuqta weights from Hugging Face
> (~1.7 GB for `nuqta-0.8b`).

## Usage

Once installed, `ocrmypdf-nuqta` is discovered through OCRmyPDF's `ocrmypdf`
entry point (`[project.entry-points."ocrmypdf"]`). **Tesseract remains the
default engine** for normal `ocrmypdf` runs unless you opt into Nuqta via
`--ocr-engine nuqta`, `--nuqta`, or any `--nuqta-*` option:

```bash
ocrmypdf --ocr-engine nuqta -l ara --nuqta-device cuda book.pdf out.pdf
```

Or with the convenience flag:

```bash
ocrmypdf --nuqta -l ara --nuqta-device cuda book.pdf out.pdf
```

> Do **not** pass `--plugin ocrmypdf_nuqta` when the package is already
> installed via its entry point: OCRmyPDF loads the entry point first and then
> `pluggy` rejects `--plugin ocrmypdf_nuqta` as already registered.

### Options

Excerpt from `ocrmypdf --help`:

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
  --nuqta-dtype {bfloat16,float16,float32}
                        Model weight dtype (default: bfloat16)
  --nuqta-max-new-tokens NUQTA_MAX_NEW_TOKENS
                        token budget per line (default: 1024)
```

The plugin forces `--jobs 1` when Nuqta is selected so only one model instance
is loaded per process. When Tesseract or `none` is active, `--jobs` is left
unchanged.

### Examples

```bash
# Use the 4B model on GPU
ocrmypdf --ocr-engine nuqta --nuqta-model 4B --nuqta-device cuda -l ara book.pdf out.pdf

# Force CPU execution (slow; see speed section below)
ocrmypdf --nuqta --nuqta-device cpu -l ara sample.pdf out.pdf
```

## Colab / GPU check script

Run the included check script on a GPU machine (e.g. Google Colab) to install
dependencies, run OCRmyPDF with Nuqta on a PDF, time it, and extract text with
both `pdftotext` and PyMuPDF:

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

- **CPU:** Very slow. Observed on an AMD Ryzen 5 2400G (4 cores / 8 threads,
  15 GB RAM, no GPU) with Nuqta 0.8B (`transformers`, `bfloat16`): roughly
  **350 s** for 8 lines (~44 s/line) on a 1024×620 crop — see
  [MEASUREMENTS.md](MEASUREMENTS.md). The plugin logs a warning on CPU.
- **GPU:** Practical page throughput requires an NVIDIA CUDA GPU (`--nuqta-device cuda`).
  End-to-end GPU throughput for this plugin has not yet been measured in this
  environment; see upstream `nuqta-ocr` for upstream's `read_lines` GPU
  measurements.

## Known limitations (Alpha)

- **Alpha release:** Not yet published to PyPI; full-book real-model GPU runs
  of the plugin are still pending.
- **Single-column pages only:** `nuqta.read_lines` uses a horizontal projection
  profile; multi-column pages are read across columns rather than down.
- **Proportional word boxes:** `ocrx_word` bounding boxes are estimated
  proportionally by non-combining character width within each line box (with
  visual slot ordering for embedded LTR/numeric runs). Text selection
  highlights are therefore approximate rather than ink-tight.
- **Missing space in PDF text layer on RTL transitions:** OCRmyPDF `17.13.0`'s
  `Fpdf2PdfRenderer` only emits a literal `' '` glyph between consecutive words
  when `next_word.x_baseline > word.x_baseline` (or on the line-final word).
  For RTL words advancing right-to-left (`advance < 0`), no literal space is
  written in the PDF stream (confirmed with `pypdf` and `pikepdf`). Poppler
  (`pdftotext`) infers spaces from gaps between Arabic words, except where an
  RTL word sits immediately to the left of an LTR/digit run (e.g.
  `Hello Worldبِكُمْ`), where Poppler places the inferred space inside the LTR
  embedding instead of after it.
- **Diacritic-sensitive search:** PDF viewers match vocalized queries against
  vocalized text layers; unvocalized queries do not match words containing
  tashkeel.
- **Renderer support:** `--pdf-renderer sandwich` is not supported; use the
  default `fpdf2` hOCR renderer.

## License

Apache-2.0, matching Nuqta OCR.
