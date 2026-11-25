#!/usr/bin/env bash
# End-to-end Colab / GPU verification script for ocrmypdf-nuqta.
# Installs system + Python dependencies, runs OCRmyPDF with the Nuqta engine
# on an input PDF, times the run, and extracts text with pdftotext and PyMuPDF.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input.pdf> [output.pdf] [model]" >&2
  exit 2
fi

INPUT_PDF="$1"
OUTPUT_PDF="${2:-out_nuqta.pdf}"
MODEL="${3:-nuqta-0.8b}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$INPUT_PDF" ]]; then
  echo "Input PDF not found: $INPUT_PDF" >&2
  exit 1
fi

run_apt() {
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get "$@"
  else
    sudo apt-get "$@"
  fi
}

if ! command -v tesseract >/dev/null 2>&1 || ! command -v pdftotext >/dev/null 2>&1; then
  echo "[1/5] Installing system packages (tesseract-ocr, poppler-utils)..."
  run_apt update -qq
  run_apt install -y -qq tesseract-ocr poppler-utils
fi

echo "[2/5] Installing ocrmypdf-nuqta[layout] and pymupdf..."
python3 -m pip install -q -e "${REPO_DIR}[layout]" pymupdf

echo "[3/5] Environment check:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
  echo "No NVIDIA GPU detected (CPU fallback will be slow)."
fi
DEVICE="$(python3 -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
python3 -m ocrmypdf --version

echo "[4/5] Running OCRmyPDF with Nuqta (${MODEL}, device=${DEVICE})..."
START_NS="$(date +%s%N)"
python3 -m ocrmypdf \
  --ocr-engine nuqta \
  --nuqta-model "${MODEL}" \
  --nuqta-device "${DEVICE}" \
  -l ara \
  --force-ocr \
  "${INPUT_PDF}" "${OUTPUT_PDF}"
END_NS="$(date +%s%N)"
ELAPSED_S="$(python3 -c "print(f'{(${END_NS} - ${START_NS}) / 1e9:.2f}')")"
echo "Elapsed: ${ELAPSED_S}s -> ${OUTPUT_PDF}"

echo "[5/5] Extracting text layer with pdftotext and PyMuPDF..."
PDFTOTEXT_OUT="${OUTPUT_PDF%.pdf}.pdftotext.txt"
PYMUPDF_OUT="${OUTPUT_PDF%.pdf}.pymupdf.txt"

pdftotext "${OUTPUT_PDF}" "${PDFTOTEXT_OUT}"
python3 - "${OUTPUT_PDF}" "${PYMUPDF_OUT}" <<'PY'
import sys
import pymupdf

pdf_path, out_path = sys.argv[1], sys.argv[2]
doc = pymupdf.open(pdf_path)
text = "\n\f\n".join(page.get_text("text") for page in doc)
doc.close()
with open(out_path, "w", encoding="utf-8") as f:
    f.write(text)
PY

echo "--- pdftotext (${PDFTOTEXT_OUT}) ---"
cat "${PDFTOTEXT_OUT}"
echo "--- PyMuPDF (${PYMUPDF_OUT}) ---"
cat "${PYMUPDF_OUT}"
