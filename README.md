# ocrmypdf-nuqta

An [OCRmyPDF](https://ocrmypdf.readthedocs.io/) engine plugin that uses
[Nuqta](https://github.com/SalahAlHaismawi/nuqta-ocr) for Arabic OCR. Nuqta is
a vision-language model that preserves printed tashkeel (diacritics), which
Tesseract does not.

Status: prototype. Single-column pages, line-level hOCR, one worker.

## Install

```bash
# not on PyPI yet
pip install "ocrmypdf-nuqta @ git+https://github.com/AbdoSol1iman/ocrmypdf-nuqta"
```

The plugin is discovered through OCRmyPDF's `ocrmypdf` entry point and is used
automatically with `--ocr-engine auto` (the default). The first run downloads
the Nuqta model (~1.7 GB). OCRmyPDF 17.13 still checks for the `tesseract`
binary even when another engine is selected; install Tesseract or make sure a
`tesseract` executable is on `PATH`.

## Use

```bash
ocrmypdf --nuqta-device cuda -l ara in.pdf out.pdf
```

| option | default | meaning |
|---|---|---|
| `--nuqta-model` | `nuqta-0.8b` | model name, Hugging Face id or local path |
| `--nuqta-backend` | `auto` | `auto`, `transformers` or `vllm` |
| `--nuqta-device` | `auto` | `auto`, `cpu` or `cuda` |
| `--nuqta-dtype` | `bfloat16` | `bfloat16`, `float16` or `float32` |
| `--nuqta-max-new-tokens` | `1024` | token budget per line |

The plugin forces `--jobs 1`: a model per OCRmyPDF worker would exhaust memory.
A persistent inference server is future work.

## Limits

- **GPU strongly recommended.** On a 4-core CPU the 0.8B model took tens of
  seconds per line; a page has ~20-30 lines. The plugin logs a warning when it
  falls back to CPU.
- **Single column only.** Line boxes come from a horizontal projection
  profile; two columns are read across, not down.
- **Line-level hOCR.** Each line is one `ocrx_word` spanning the line box, so
  PDF text selection snaps to whole lines. Word-level boxes are future work.
- **Arabic only.** Latin words and digits inside Arabic lines are read, but
  OCRmyPDF 17.13's fpdf2 renderer reverses Arabic-Indic digits and Latin runs
  when a line is emitted as one word (it pre-reverses RTL text and relies on
  bidi to restore it). Word-level boxes are the fix and the next step.
- `--pdf-renderer sandwich` is not supported; the default `fpdf2` hOCR path is.
- Tested against OCRmyPDF 17.13.0; the hOCR shape follows its parser (one
  `ocrx_word` per `ocr_line`, wrapped in `ocr_par`, `lang="ara"`).

## License

Apache-2.0, matching Nuqta OCR.
