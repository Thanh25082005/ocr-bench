#!/usr/bin/env bash
# Chạy TRÊN MÁY CỦA ANH: đưa bộ test đã dựng sẵn (thư mục testset/) lên Kaggle thay vì dựng lại (~540 MB).
#   bash scripts/kaggle/push_testset.sh <ssh-host> [thư-mục-testset]
set -euo pipefail
HOST=${1:?cần <ssh-host>}
SRC=${2:-testset}
WORK=${WORK:-/kaggle/working}
[ -f "$SRC/manifest.jsonl" ] || { echo "Không thấy $SRC/manifest.jsonl"; exit 1; }
rsync -az --info=progress2 --exclude '.cache_*' "$SRC/" "$HOST:$WORK/testset/"
echo "Xong. Trên Kaggle: ocrbench validate --config $WORK/config.yaml (so dấu vân tay với máy mình)"
