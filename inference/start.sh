#!/usr/bin/env bash
# Web demo dots.mocr: kéo thả PDF/ảnh → xem bố cục từng trang + tải DOCX.
#
#   bash inference/start.sh            # cài, tự kiểm tra model, mở web ở 127.0.0.1:7860
#   SHARE=1 bash inference/start.sh    # thêm link công khai *.gradio.live (KHÔNG dùng với tài liệu thật/mật)
#   GPUS=1 bash inference/start.sh     # chỉ dùng GPU 1 (khi GPU 0 đang chạy benchmark)
#   SKIP_CHECK=1 bash inference/start.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${PORT:-7860}"
GPUS="${GPUS:-auto}"

echo "== 1. Cài thư viện (app + hf + dots)"
python -m pip install -q -e ".[app,hf,dots]"

if [ -z "${SKIP_CHECK:-}" ]; then
  echo "== 2. Tự kiểm tra: dots đọc một trang mẫu (lần đầu tải model ~6 GB)"
  python inference/check.py --gpus "$GPUS"
fi

echo "== 3. Mở web: http://127.0.0.1:${PORT}"
echo "   Máy mình: ssh -L ${PORT}:localhost:${PORT} kaggle-ngrok   rồi vào http://localhost:${PORT}"
ARGS=(--config inference/config.yaml --model dots_mocr --port "$PORT" --gpus "$GPUS" --ocr-all
      --title "dots.mocr — đọc bố cục tài liệu (demo)")
if [ -n "${SHARE:-}" ]; then ARGS+=(--share); fi
exec ocrbench serve "${ARGS[@]}"
