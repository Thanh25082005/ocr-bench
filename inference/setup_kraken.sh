#!/usr/bin/env bash
# Cài kraken (tách dòng blla cho bố cục web: adapter kraken_layout) vào venv RIÊNG.
#   bash inference/setup_kraken.sh          # in ra đường dẫn python của venv ở dòng cuối
# Vì sao venv riêng: kraken 7 kéo bản torch riêng (2.9+) — cài chung sẽ nâng torch của hệ thống, làm hỏng transformers
# / dots. Venv KHÔNG dùng gói hệ thống; ocrbench nói chuyện với nó qua tiến trình con (ocrbench/kraken_worker.py).
# Vị trí: $KRAKEN_VENV, mặc định /kaggle/working/venvs/kraken (chung với experiments/real_docs_probe/run_kraken.sh)
# hoặc ~/venvs/kraken khi không chạy trên Kaggle. Adapter kraken_layout tự tìm đúng chỗ này.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -d /kaggle/working ]; then DEFAULT=/kaggle/working/venvs/kraken; else DEFAULT="$HOME/venvs/kraken"; fi
VENV="${KRAKEN_VENV:-$DEFAULT}"
if [ ! -x "$VENV/bin/python" ]; then
  python -m pip install -q uv
  python -m uv venv "$VENV" -q
fi
if ! "$VENV/bin/python" -c "import kraken" 2>/dev/null; then
  echo "== Cài kraken vào $VENV (lần đầu tải torch ~2–3 GB)"
  python -m uv pip install -q -p "$VENV/bin/python" "kraken>=7.1.0"
fi
"$VENV/bin/kraken" --version
# tự kiểm tra: tách dòng một trang mẫu
"$VENV/bin/python" "$ROOT/ocrbench/kraken_worker.py" --check "$ROOT/inference/samples/04_hop_dong_tieng_a_rap.png"
echo "$VENV/bin/python"
