# ocrmypdf-nuqta

An early **prototype** [OCRmyPDF](https://ocrmypdf.readthedocs.io/) engine
plugin that uses [Nuqta](https://github.com/SalahAlHaismawi/nuqta-ocr) for
Arabic OCR. Nuqta is a vision-language model that preserves printed tashkeel
(diacritics). This is not production software and is not on PyPI.

Status: single-column pages, line-level hOCR, one worker. Word-level boxes and
a persistent inference server are not implemented.

## Install

```bash
# from source; not on PyPI
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

## Known limits

- **Early prototype.** No output-quality validation, no word-level boxes, no
  benchmark suite, and no PyPI release.
- **GPU needed in practice.** CPU inference is far too slow for normal use;
  the plugin logs a warning when it falls back to CPU. The one wall-clock
  observation we have is in [MEASUREMENTS.md](MEASUREMENTS.md); it is not a
  controlled benchmark.
- **Single column only.** Line boxes come from a horizontal projection
  profile; two columns are read across, not down.
- **Line-level hOCR.** Each line is one `ocrx_word` spanning the line box.
  Selection snaps to whole lines. In OCRmyPDF 17.13 this also means
  Arabic-Indic digits and mixed Latin/Arabic runs can be extracted reversed;
  isolated digits on their own line may be correct. Search is therefore
  unreliable for numerals.
- **Search is diacritic-sensitive.** Vocalized strings match the text layer;
  unvocalized queries do not match vocalized text. That is a viewer/analyzer
  limitation, not fixed by this plugin.
- **Arabic only.** Latin words inside Arabic lines are read, but the text
  layer may garble them.
- `--pdf-renderer sandwich` is not supported; the default `fpdf2` hOCR path is.
- Tested against OCRmyPDF 17.13.0; the hOCR shape follows its parser (one
  `ocrx_word` per `ocr_line`, wrapped in `ocr_par`, `lang="ara"`).

## License

Apache-2.0, matching Nuqta OCR.
