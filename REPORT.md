# alpha-prep report — ocrmypdf-nuqta 0.1.0.dev0

Status at the time of writing: everything below ran on this machine
(Fedora 44, kernel 6.19.10-300, Python 3.14.3) in this session. Nothing has
been pushed; the branch `alpha-prep` is local.

## What is on the branch

Focused commits, oldest first:

1. `b25d125` — pin nuqta-ocr to upstream main merge commit `832f55a1`
   (`832f55a1703eb88a8e501d74709013c255db6b66`), allow direct references in
   hatch metadata, version `0.1.0.dev0`, CI + README updated to the pin.
2. `687d07c` — `src/ocrmypdf_nuqta/words.py` + `tests/test_words.py`:
   word-level boxes (Otsu binarization, vertical-projection gap splitting,
   token/script-run splitting, DP width alignment, proportional fallback).
3. `91fef56` — plugin wiring: word-level hOCR by default
   (`--nuqta-line-hocr` escape hatch), `--nuqta-text-layer {as-written,bare}`
   (DRAFT default `as-written`), dtype auto-detection via
   `torch.cuda.is_bf16_supported()`.
4. `7a0c116` — measurement harness: `experiments/` (stub engine, synthetic
   cases, real-page runs, search test) and `scripts/gpu_e2e.py`.
5. drafts + this report (see below).

## Task 2 — word-level hOCR vs digit/Latin reversal

**Approach.** For each line box: binarize the page once (Otsu), split the
line's ink at vertical-projection gaps into x-ranges, split the line text
into whitespace tokens and each token into script runs (Arabic /
Arabic-Indic digits / Latin / ASCII digits), then align tokens to ink
segments with dynamic programming over expected glyph widths (tashkeel
weight 0; pair/merge/split/skip operations). Lines whose aligned boxes come
out touching are re-segmented at tighter thresholds (0.08/0.05/0.03) and the
variant with the fewest touching pairs wins (DP cost breaks ties). If
nothing aligns, fall back to proportional distribution.

**Rejected designs** (all measured worse on synthetic + real pages, then
reverted):

- count-match adaptive threshold (first threshold whose segment count equals
  the token count) — misassigns words when the right count comes from the
  wrong split;
- DP-cost-based threshold selection across the whole sweep — regressed
  synthetic cases via merges across real word gaps;
- gap-aware merge penalty and skip-cost tuning in the DP — still misassigned
  punctuation segments;
- proportional layout snapped to ink gaps ("approach C") — regressed p27 to
  75% reversal.

**Measured (this session; poppler pdftotext 26.01 + PyMuPDF 2.1):**

- Synthetic shaped-Arabic cases (8 cases, real font + raqm rendering):
  all-checks pass rate with pdftotext: line-boxes **0.25** → token/run
  word-level **0.75** (order checks included). PyMuPDF shows the inverse
  (0.625 line vs 0.375 token/run).
- Real pages p21/p22/p27 (200 dpi, 1024×1867): reversed Arabic-Indic digit
  runs with pdftotext: **67/100/75%** (line) → **0/0/0%** (token/run); with
  PyMuPDF the same word-level boxes reverse **67/100/75%** (extractor-side
  inverse pattern — open risk, below).
- All 27 tests pass (`pytest -q`, fake OCR, no model download).

## Task 3 — text-layer option (`--nuqta-text-layer`)

Measured on the heavily vocalized page 22, word-level hOCR:

- `as-written` layer: **282** harakat characters survive into the extracted
  layer (of 1786 extracted chars); **0 of 5** unvocalized test queries found
  by MuPDF `search_for`, 0 by pdftotext raw match.
- `bare` layer: **0** harakat chars (1504 extracted chars); **2 of 5**
  queries found by MuPDF search, 1 of 5 by pdftotext raw match.

Trade-off table (no default chosen by me — the shipped default is a
maintainer decision; the prototype defaults to `as-written`):

| | `as-written` | `bare` |
|---|---|---|
| copied text faithful to print | yes (keeps tashkeel) | no (harakat dropped) |
| unvocalized search recall | poor (0/5 measured) | better (2/5 measured) |
| copy/search of vocalized text | exact | mismatches |

`--nuqta-line-hocr` keeps the pre-0.1 one-word-per-line shape.

## Task 4 — dtype auto-detection

`--nuqta-dtype auto` resolves via `torch.cuda.is_bf16_supported()`: CUDA+bf16
→ `bfloat16`, CUDA otherwise → `float16`, CPU → `float32`; explicit
`--nuqta-dtype` always wins. Covered by 6 monkeypatched tests (fake torch
module, including bf16-unsupported and RuntimeError paths).

**Unverified on real hardware**: no GPU here, so the T4 behaviour of
`is_bf16_supported()` (which is known to consult driver capability) has not
been observed; only the code paths are tested.

## Task 5 — tesseract requirement

Measured with OCRmyPDF 17.13.0: running `ocrmypdf` with a plugin that
provides the OCR engine, without `tesseract` on PATH, fails at startup:

```
Could not find program 'tesseract' on the PATH
MissingDependencyError
```

The error message suggests `sudo apt/dnf install tesseract-ocr`. Reading
OCRmyPDF's validation (`_validation.py` → `plugin_manager.check_options`)
shows the built-in tesseract plugin's `check_options` hook always runs; no
option or documented bypass exists to skip it. **Minimal solution: install
tesseract-ocr alongside the plugin** (it is not used for recognition). On
this machine tesseract 5.5.4 was extracted from Fedora RPMs without sudo and
only satisfies the PATH check.

## Task 6 — GPU end-to-end script

`scripts/gpu_e2e.py` (committed): runs one full `ocrmypdf` pass with the
Nuqta plugin on a CUDA machine, sampling peak VRAM (torch counters +
`nvidia-smi`) in a background thread, timing total/per-page seconds, and
auditing the output layer per page (reversed/missing digit runs vs the
source PDF text) plus search queries. Writes a JSON report next to the
output.

**Not executed** — no GPU in this environment. `py_compile` and `--help`
verified; every number it would print is unverified until a T4/RTX run.
Model-level GPU performance is now available third-party (next section);
what remains unverified is the *plugin path* (OCRmyPDF + renderer + hOCR
through this package) on real hardware.

## Third-party GPU measurements of Nuqta (model author, 2026-11-11)

Raw data: `experiments/nuqta-gpu-tests.txt` (verbatim copy of the file the
Nuqta author provided; measured with Nuqta directly, **not** through this
plugin). Setup: 0.8B model, vLLM 0.19.1, RTX 5090 Laptop GPU, as-written
mode, model warmed up; 67 pages across four document sets (turath,
finepdfs, vandyck, UN), plus 3 mixed Arabic/Latin/digit pages.

- **Throughput**: full page median **3.49 s/page** (10–90 pct 2.2–6.6 s);
  line mode (what this plugin uses via `read_lines`) **0.79 s/page**
  (0.54–1.3 s). Segmentation is under 0.1 s of that.
- **Word recall** (vs ground truth, tashkeel ignored): full **97.0%**, line
  **95.3%**. Exact tashkeel: full **85.0%**, line **84.8%**. The line-vs-full
  gap has median 0.9 points and is print-dependent: near zero on clean
  modern print (UN +0.1, finepdfs −0.8), −4.5 on worn 19th-century print
  (VanDyck).
- **Tashkeel dropout in line mode**: on fully-vowelled pages, 25 of 554
  lines (4.5%) returned no tashkeel at all (each crop decides for itself);
  full-page reading lost marks on only 1 of 30 such pages. Implication for
  the `as-written` text layer: a small fraction of lines will be bare even
  in as-written mode — worth documenting and, later, an upstream
  `read_many` crop-batching follow-up (the author notes full-page batching
  already beats per-line batching: 70 pages in 40.9 s vs 55.5 s).
- **Mixed-script lines read well**: Latin words 96–100%, digits 100%,
  Arabic 98–100% in both modes — so the digit reversal we fixed was a
  text-layer/bidi problem, not a recognition problem.
- **Validation of word-level boxes**: 34% of 1,620 detected lines mix
  Arabic with digits or Latin (73% of lines on VanDyck, 28% on turath) —
  confirming that line-as-one-word hOCR would garble about a third of
  lines on typical book pages.
- The author offered to run our p21/p22/p27 page images through both modes
  on the GPU; attaching the images would give plugin-comparable numbers
  for our audited pages.

## Task 7 — release preparation

- Version set to `0.1.0.dev0` (`pyproject.toml` + `__version__`).
- Dependency pinned to upstream nuqta-ocr commit `832f55a1` because **no
  nuqta-ocr release or tag contains `read_lines`/`to_hocr`** (verified via
  `gh release list` / tags earlier in this work stream; PyPI nuqta-ocr 0.2.0
  predates PR #3). Hatch `allow-direct-references = true` makes the build
  pass.
- sdist + wheel built fresh this session:
  `ocrmypdf_nuqta-0.1.0.dev0.tar.gz` / `-py3-none-any.whl`; **`twine check`
  PASSED** for both.
- Clean-venv wheel install verified: fresh venv → `pip install
  ocrmypdf==17.13.0`, then the wheel + runtime deps `--no-deps`, then local
  nuqta-ocr `--no-deps`; `ocrmypdf --help` lists all six `--nuqta-*`
  options (the entry point registers them without any `--plugin` flag).
  **Honest caveat:** the wheel's real dependency line pulls nuqta-ocr from
  git, which drags in torch (~2 GB) and friends; the `--no-deps` +
  local-nuqta-ocr procedure above was the workaround for this CPU-only test
  box. A user installing the wheel on a GPU box gets torch anyway (the
  model needs it); on a CPU-only box that is a heavy but expected install.
- `drafts/pypi-publish.yml`: trusted-publishing workflow, **disabled**
  (workflow_dispatch only, publish step commented out) with the four
  preconditions listed in the file header. PyPI **rejects direct git
  references**, so a real PyPI release additionally needs a released
  nuqta-ocr version first — recorded in the workflow draft and the
  pyproject comment.

## Task 8 — README drafts

`drafts/README-sections.md` (all marked DRAFT for maintainer review):

- AI disclosure paragraph.
- Known-limits section, restricted to measured results (extractor inverse
  pattern, visual-order mixed lines, vocalized-layer search, dense-print
  fusion, line-hOCR digit reversal, CPU impracticality, tesseract still
  required).
- Model-license section: **link-only** to the two model cards
  (0.8B and 4B); both URLs verified reachable (HTTP 200) while preparing
  this report. Terms are not restated — model cards are authoritative.

## Verified vs unverified

| Claim | Status |
|---|---|
| Synthetic pdftotext 0.25→0.75 (token/run), all 27 tests pass | **measured this session** |
| Real pages p21/p22/p27: pdftotext reversal 67/100/75%→0% | **measured this session** |
| PyMuPDF inverse reversal with word-level boxes (67/100/75%) | **measured this session** |
| as-written layer keeps 282 harakat; bare keeps 0 | **measured this session** |
| MuPDF search: bare 2/5 vs as-written 0/5 | **measured this session** |
| OCRmyPDF 17.13 fails without tesseract; no bypass | **measured this session** |
| sdist+wheel build + `twine check` PASS | **measured this session** |
| Clean-venv wheel install; `--nuqta-*` in `--help` | **measured this session** |
| Model-card URLs reachable | **measured this session** |
| Nuqta model GPU throughput/accuracy (0.8B, vLLM, RTX 5090) | **third-party measured** (model author, `experiments/nuqta-gpu-tests.txt`; not the plugin path) |
| dtype auto on real T4/CUDA hardware | unverified (no GPU) |
| `scripts/gpu_e2e.py` end-to-end numbers | unverified (compile-checked only) |
| Plugin-path GPU behaviour (OCRmyPDF + renderer + this plugin) | unverified (no GPU run of the full path) |
| CPU inference being minutes-per-page | unverified (upstream docs claim; GPU numbers above are third-party) |
| Dense-print fusion limits on other books | unverified (one book sampled) |

## Environment and commands

- Fedora 44, Python 3.14.3, ocrmypdf 17.13.0, poppler pdftotext 26.01.0,
  PyMuPDF 2.1, OpenCV 5.0.0, NumPy 2.5.4, Pillow 13.0.0, tesseract 5.5.4
  (RPM-extracted, PATH-only use), nuqta-ocr @ `832f55a1` (PYTHONPATH or
  `--no-deps` install).
- Tests: `PYTHONPATH=src:/tmp/opencode/nuqta-ocr/src python -m pytest -q`
- Synthetic: `python experiments/make_cases.py && python
  experiments/run_cases.py && python experiments/evaluate_cases.py`
- Real pages: `python experiments/run_real.py` (line data from
  `experiments/spike_out/page_{21,22,27}.json`, recorded earlier with the
  real model on CPU)
- Search: `python experiments/search_test.py`
- Build: `python -m build && python -m twine check dist/*`
- Raw outputs: `experiments/out_cases/results.json`,
  `experiments/out_real/results.json`, `experiments/out_search/results.json`.

## Open risks

1. **Extractor disagreement (pdftotext vs PyMuPDF).** The two extractors
   invert each other's digit handling on the same word-level PDF. Whatever
   renderer/hOCR side does, one popular extractor reports reversals. Needs
   upstream investigation (OCRmyPDF renderer + poppler + MuPDF bidi).
2. **Visual order in mixed lines.** pdftotext emits digit-containing mixed
   lines visually, not logically; copy-paste of such lines reads backwards.
   Not fixable from hOCR.
3. **PyPI direct-reference rejection.** A real PyPI release is blocked on a
   nuqta-ocr release containing `read_lines` (or a PyPI-reachable nuqta
   wheel). Tracked in the publish draft.
4. **Dense vocalized print.** Short zero-gap words still fuse
   (`قطبل`, `هوكل`, `علىمن` on p22); worst fusion fixed, remainder lowers
   search recall.
5. **torch install weight.** The git dependency line installs the full Nuqta
   stack; acceptable but heavy for CPU-only users.
6. **No GPU verification of the plugin path.** dtype auto and the e2e
   benchmark are unproven on the target hardware. (Model-level GPU numbers
   exist third-party; the OCRmyPDF plugin path specifically has not been
   exercised on CUDA.)
7. **Line-mode tashkeel dropout.** The plugin reads per line (`read_lines`);
   the author's GPU data shows 4.5% of lines on fully-vowelled pages return
   with no tashkeel at all, so the `as-written` layer has occasional bare
   lines by construction. Full-page mode avoids this but is ~4× slower and
   loses ~1–2 points of word recall; a mode option could be a follow-up.

## Go / no-go for a 0.1.0 alpha release

**Recommendation: GO for an alpha** (GitHub release and/or TestPyPI) **after
the maintainer:**

1. decides the shipped `--nuqta-text-layer` default (as-written vs bare);
2. reviews the DRAFT README sections and the AI-disclosure wording;
3. picks the final version (drop `.dev0`) and, for PyPI specifically, waits
   for a nuqta-ocr release that removes the direct-reference blocker
   (a GitHub-release alpha has no such constraint);
4. optionally runs `scripts/gpu_e2e.py` on one CUDA box to replace the
   unverified rows in the table above.

Risks 1, 2 and 4 are honest-documentation issues, not blockers for an alpha
labelled as such.
