# DRAFT README sections — for the maintainer to review, edit, and place.

Everything in this file is a DRAFT: text the maintainer may adopt, edit, or
discard before any release. Facts labelled "measured" come from the
experiment runs recorded in `REPORT.md` and `experiments/out_*/results.json`;
anything else is marked "unverified".

---

## 1. AI disclosure (DRAFT)

Suggested wording for the README:

> ### How this plugin was built
>
> This plugin was developed with heavy assistance from an AI coding agent
> (code, tests, benchmark harness, and documentation drafts). The agent
> produced the implementation and the measurement scripts; all results
> reported here were produced by reproducible scripts in `experiments/` on
> the developer's machine, and the developer reviewed and committed the
> code. Issues and pull requests are reviewed by a human.

(Maintainer: adjust to the disclosure policy you are comfortable with;
the substance to keep is "AI wrote code, human reviewed, measurements are
reproducible via `experiments/`".)

## 2. Known limits (DRAFT — measured results only)

Suggested "Known limitations" section. Every bullet below traces to a
measured result (see REPORT.md for the run and numbers):

> ### Known limitations
>
> - **Digit/Latin runs inside Arabic lines.** With word-level hOCR (the
>   default), reversed Arabic-Indic digit runs in the extracted text layer
>   went from 67–100% (line-level boxes) to 0% on the three audited book
>   pages, and the synthetic all-checks score rose from 0.25 to 0.75 with
>   poppler's `pdftotext`. However, **PyMuPDF shows the inverse pattern**:
>   with word-level streams its extraction reverses digit runs that
>   `pdftotext` gets right (67–100% reversed on the same pages). Choose the
>   extractor you audit with accordingly; this appears to be an extractor
>   bidi-handling difference, not an hOCR defect, and is under investigation.
> - **Visual order in mixed lines.** `pdftotext` emits digit-containing
>   mixed lines in visual, not logical, word order (e.g. `٢٢ - ٢٦` extracted
>   as `٢٦] -٢٢`). Copy-paste of such lines reads backwards; unvocalized
>   keyword search still matches. Pure-Arabic lines extract in logical
>   order. Not fixable from the hOCR side.
> - **Search of unvocalized text in vocalized layers.** A layer written
>   `as-written` (with tashkeel, the current default) does not match
>   unvocalized search queries: 0 of 5 test queries found with MuPDF search,
>   versus 2 of 5 on a `bare` (harakat-stripped) layer. Use
>   `--nuqta-text-layer bare` if copy/search matters more than faithful
>   reproduction. (The default is a maintainer decision, DRAFT.)
> - **Dense vocalized print can fuse short words.** Short words printed
>   with no ink gap (e.g. `قط بل`, `هو كلٌّ`, `على من` in dense vocalized
>   pages) can merge in the text layer; the word-refinement step fixed the
>   worst observed case (`يوجههلايأتبخير`) but 3–4 fused pairs remain on
>   page 22. This lowers search recall on such pages.
> - **`--nuqta-line-hocr` (the pre-0.1 shape) reverses digits.** One
>   `ocrx_word` per line makes OCRmyPDF's renderer treat the whole line as
>   one RTL run: 67–100% of digit runs reversed on the audited pages. Kept
>   only as an escape hatch.
> - **CPU-only runs are impractical.** Unverified on this dev machine (no
>   GPU); the model author's GPU measurements (0.8B, vLLM 0.19.1, RTX 5090
>   Laptop) put a full page at a median 3.49 s and line mode at 0.79 s
>   (`experiments/nuqta-gpu-tests.txt`) — CPU inference is orders of
>   magnitude slower. Use `--nuqta-device cuda` on a machine with an NVIDIA
>   GPU; see `scripts/gpu_e2e.py` for the plugin benchmark, whose numbers
>   are unverified until reproduced.
> - **OCRmyPDF still requires tesseract.** OCRmyPDF 17.13 fails at startup
>   (`MissingDependencyError`) without `tesseract` on PATH even when a
>   plugin provides the OCR engine (measured). Install `tesseract-ocr`
>   alongside this plugin; it is not used for recognition.

## 3. Nuqta model license (DRAFT — link only)

Suggested wording for the README (link, do not restate the terms — the
model cards are authoritative and may change):

> ### Model licenses
>
> The plugin's code is Apache-2.0, but the OCR models it runs are licensed
> separately by their authors:
>
> - [`SalahALHaismawi/Nuqta-Arabic-OCR-0.8B`](https://huggingface.co/SalahALHaismawi/Nuqta-Arabic-OCR-0.8B) (the default, `nuqta-0.8b`)
> - [`SalahALHaismawi/Nuqta-Arabic-OCR-4B`](https://huggingface.co/SalahALHaismawi/Nuqta-Arabic-OCR-4B) (`nuqta-4b`)
>
> See each model card for its license and terms of use; the Nuqta source
> code is Apache-2.0.

(Both model-card URLs verified reachable — HTTP 200, checked while preparing
the alpha-prep report.)
