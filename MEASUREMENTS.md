# Measurements

This is a raw development note, not a benchmark. It exists so the README does
not claim numbers that are not recorded anywhere.

## 2026-11-08 — OCRmyPDF plugin on CPU

- Host: AMD Ryzen 5 2400G (4 cores / 8 threads), 15 GB RAM, no GPU
- OCRmyPDF 17.13.0, plugin 0.1.0, Nuqta 0.8B, `transformers`, bfloat16
- Input: a 1024×620 crop of page 21, eight detected lines
- Model was already cached; the run produced the PDF at 12:48:40 and started
  after 12:42:47, so wall clock was roughly **350 s** for eight lines
  (about 44 s/line including image work). This was not instrumented per line
  and should be treated as an observation, not a benchmark.
- Output: searchable text layer; `٢٤` was extracted as `٤٢` (the known
  single-word RTL reversal).

No GPU, VRAM, accuracy or throughput measurement is recorded here.
