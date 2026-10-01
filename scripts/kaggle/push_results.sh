#!/usr/bin/env bash
# Chạy TRÊN MÁY CỦA ANH, sau khi mở phiên Kaggle MỚI và đã chạy setup.sh:
# đẩy kết quả đã kéo về lên lại, để `ocrbench run` chạy tiếp từ chỗ dừng thay vì chạy lại từ đầu.
#   bash scripts/kaggle/push_results.sh <ssh-host> [thư-mục-nguồn]
# Kết quả chỉ được dùng tiếp nếu bộ test trên Kaggle có CÙNG dấu vân tay (xem `ocrbench validate`);
# nếu khác, báo cáo sẽ in cảnh báo "dữ liệu đã thay đổi".
set -euo pipefail
HOST=${1:?cần <ssh-host>}
SRC=${2:-kaggle_results}
WORK=${WORK:-/kaggle/working}
rsync -az --info=stats1 "$SRC/runs/" "$HOST:$WORK/runs/"
for f in config.yaml EXPERIMENTS.md; do
  [ -f "$SRC/$f" ] && rsync -az "$SRC/$f" "$HOST:$WORK/"
done
[ -d "$SRC/reports" ] && rsync -az "$SRC/reports/" "$HOST:$WORK/reports/"
echo "Đã đẩy lên. Trên Kaggle chạy lại đúng lệnh ocrbench run trước đó để tiếp tục."
