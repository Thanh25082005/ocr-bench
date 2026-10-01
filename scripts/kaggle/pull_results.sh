#!/usr/bin/env bash
# Chạy TRÊN MÁY CỦA ANH: kéo kết quả từ Kaggle về để không mất khi phiên Kaggle kết thúc.
#   bash scripts/kaggle/pull_results.sh <ssh-host> [thư-mục-đích]
# Nên chạy định kỳ, vd. mỗi 30 phút:  watch -n 1800 bash scripts/kaggle/pull_results.sh <ssh-host>
set -euo pipefail
HOST=${1:?cần <ssh-host> (tên trong ~/.ssh/config hoặc user@host)}
DEST=${2:-kaggle_results}
WORK=${WORK:-/kaggle/working}
mkdir -p "$DEST"
rsync -az --info=stats1 "$HOST:$WORK/runs/" "$DEST/runs/"
for f in config.yaml EXPERIMENTS.md FINAL_REPORT.md; do
  rsync -az "$HOST:$WORK/$f" "$DEST/" 2>/dev/null || true
done
rsync -az "$HOST:$WORK/reports/" "$DEST/reports/" 2>/dev/null || true
echo "Đã kéo về $DEST lúc $(date '+%H:%M:%S')"
