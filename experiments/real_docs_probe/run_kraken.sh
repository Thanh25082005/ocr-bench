#!/usr/bin/env bash
# Kraken (OCR truyền thống đọc từng dòng, bám sát hình ảnh) trên mọi ảnh trong <out>/img — chạy CPU.
#   bash experiments/real_docs_probe/run_kraken.sh /kaggle/working/real_docs_out
# Venv RIÊNG, KHÔNG dùng chung gói hệ thống (kraken kéo torch riêng → không đụng torch/CUDA của Kaggle).
# Hai mô hình nhận dạng (sau bước tách dòng baseline "blla"):
#   openiti : Printed Arabic base model, OpenITI (zenodo 10.5281/zenodo.7050296) — chữ IN Ả Rập
#   ppocrv6 : small-models-for-glam/kraken-ppocrv6-medium (Apache-2.0) — 44 ngôn ngữ, có Ả Rập + Hebrew
set -euo pipefail
OUT="$1"
VENV=/kaggle/working/venvs/kraken
M=/kaggle/working/kraken_models
if [ ! -x "$VENV/bin/kraken" ]; then
  python -m pip install -q uv
  python -m uv venv "$VENV"
  python -m uv pip install -q -p "$VENV/bin/python" "kraken>=7.1.0" huggingface_hub
fi
mkdir -p "$M"
if [ ! -f "$M/openiti/arabic_best.mlmodel" ]; then
  mkdir -p "$M/openiti"
  curl -fsSL -o "$M/openiti/arabic_best.mlmodel" "https://zenodo.org/records/7050296/files/arabic_best.mlmodel?download=1"
fi
if [ ! -d "$M/ppocrv6" ]; then
  "$VENV/bin/python" -c "from huggingface_hub import snapshot_download as s; s('small-models-for-glam/kraken-ppocrv6-medium', local_dir='$M/ppocrv6')"
fi
PP=$(ls "$M"/ppocrv6/*.safetensors "$M"/ppocrv6/*.mlmodel 2>/dev/null | head -1)
"$VENV/bin/kraken" --version
for name in openiti ppocrv6; do
  if [ "$name" = openiti ]; then MODEL="$M/openiti/arabic_best.mlmodel"; else MODEL="$PP"; fi
  D="$OUT/kraken_$name"; mkdir -p "$D"
  for IMG in "$OUT"/img/*.png; do
    STEM=$(basename "$IMG" .png)
    [ -s "$D/$STEM.txt" ] && continue
    START=$(date +%s)
    "$VENV/bin/kraken" -i "$IMG" "$D/$STEM.txt" segment -bl ocr -m "$MODEL" > "$D/$STEM.log" 2>&1 \
      || { echo "  ✘ $name $STEM — xem $D/$STEM.log"; continue; }
    echo "  kraken_$name $STEM: $(( $(date +%s) - START ))s, $(wc -m < "$D/$STEM.txt") ký tự"
  done
done
