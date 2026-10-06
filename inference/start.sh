#!/usr/bin/env bash
# Web demo dots.mocr: kéo thả PDF/ảnh → xem bố cục từng trang + tải DOCX.
#
#   bash inference/start.sh            # tạo venv dots (transformers 4.56.1), tự kiểm tra model, mở web 127.0.0.1:7860
#   SHARE=1 bash inference/start.sh    # thêm link công khai *.gradio.live (KHÔNG dùng với tài liệu thật/mật)
#   GPUS=1 bash inference/start.sh     # chỉ dùng GPU 1 (khi GPU 0 đang chạy benchmark)
#   SKIP_CHECK=1 bash inference/start.sh
#   AUTH=demo:matkhau bash inference/start.sh   # BẮT BUỘC khi cho người khác vào qua ngrok (nhiều người: a:1,b:2)
#   NO_VENV=1 bash inference/start.sh  # dùng python hệ thống (transformers 5.x: dots trả chữ rỗng — chỉ để thử)
#
# Vì sao venv: code của dots (trust_remote_code) viết cho transformers 4.56.1 (requirements.txt của họ ghim đúng
# bản này). Trên Kaggle (transformers 5.0.0) model nạp được, nhận ảnh đúng, nhưng phần ngôn ngữ trả chuỗi rỗng.
# venv dùng chung torch/CUDA của hệ thống (--system-site-packages), chỉ thay transformers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${PORT:-7860}"
GPUS="${GPUS:-auto}"
VENV="${DOTS_VENV:-/kaggle/working/venvs/dots}"

if [ -z "${NO_VENV:-}" ]; then
  if [ ! -x "$VENV/bin/python" ]; then
    echo "== 0. Tạo venv cho dots: $VENV (transformers==4.56.1, dùng chung torch của hệ thống)"
    python -m pip install -q uv
    python -m uv venv "$VENV" --system-site-packages
  fi
  PY="$VENV/bin/python"
  echo "== 1. Cài thư viện vào venv"
  python -m uv pip install -q -p "$PY" "transformers==4.56.1" -e ".[app,dots]"
else
  PY=python
  echo "== 1. Cài thư viện (python hệ thống)"
  "$PY" -m pip install -q -e ".[app,hf,dots]"
fi
"$PY" -c "import torch, transformers; print('   torch', torch.__version__, '| transformers', transformers.__version__, '| GPU', torch.cuda.device_count())"

# LibreOffice: dựng thử DOCX "giữ nguyên bố cục" để tự hiệu chỉnh cỡ chữ / vị trí từng khối (cần cho tiếng Ả Rập).
# Bản AppImage giải nén (không cần quyền root / FUSE), cài một lần ~300 MB. Bỏ qua: NO_LIBREOFFICE=1
LO_DIR="${LO_DIR:-/kaggle/working/libreoffice}"
if [ -z "${NO_LIBREOFFICE:-}" ] && ! command -v soffice >/dev/null 2>&1 && ! ls "$LO_DIR"/squashfs-root/opt/libreoffice*/program/soffice >/dev/null 2>&1; then
  echo "== 1b. Cài LibreOffice (để hiệu chỉnh DOCX bố cục) vào $LO_DIR"
  mkdir -p "$LO_DIR"
  ( cd "$LO_DIR" && curl -fsSL -o lo.AppImage https://appimages.libreitalia.org/LibreOffice-still.basic-x86_64.AppImage \
      && chmod +x lo.AppImage && ./lo.AppImage --appimage-extract >/dev/null && rm -f lo.AppImage ) \
    || echo "   ⚠ Không cài được LibreOffice — DOCX bố cục vẫn tạo được nhưng KHÔNG hiệu chỉnh (tiếng Ả Rập lệch vài mm)"
fi
SOFFICE_BIN="$(ls "$LO_DIR"/squashfs-root/opt/libreoffice*/program/soffice 2>/dev/null | tail -1 || true)"
if [ -n "$SOFFICE_BIN" ]; then export OCRBENCH_SOFFICE="$SOFFICE_BIN"; fi
"$PY" -c "from ocrbench.docx_exact import find_soffice; s = find_soffice(); print('   LibreOffice (hiệu chỉnh DOCX bố cục):', s or 'KHÔNG CÓ')"

if [ -z "${SKIP_CHECK:-}" ]; then
  echo "== 2. Tự kiểm tra: dots đọc một trang mẫu (lần đầu tải model ~6 GB)"
  "$PY" inference/check.py --gpus "$GPUS"
fi

echo "== 3. Mở web: http://127.0.0.1:${PORT}"
echo "   Máy mình: ssh -L ${PORT}:localhost:${PORT} kaggle-ngrok   rồi vào http://localhost:${PORT}"
ARGS=(--config inference/config.yaml --model dots_mocr --port "$PORT" --gpus "$GPUS" --ocr-all
      --title "dots.mocr — đọc bố cục tài liệu (demo)")
if [ -n "${SHARE:-}" ]; then ARGS+=(--share); fi
if [ -n "${AUTH:-}" ]; then ARGS+=(--auth "$AUTH"); fi
exec "$PY" -m ocrbench.cli serve "${ARGS[@]}"
