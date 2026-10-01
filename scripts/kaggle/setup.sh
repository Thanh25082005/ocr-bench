#!/usr/bin/env bash
# Chạy TRÊN MÁY KAGGLE, sau khi đã SSH vào. Chạy lại nhiều lần cũng được (bước nào xong rồi thì bỏ qua).
#
#   GITHUB_TOKEN=<token chỉ-đọc> bash setup.sh        # repo private cần token để clone
#
# Biến môi trường tùy chọn:
#   WORK=/kaggle/working     thư mục làm việc
#   BUILD_TESTSET=1          dựng bộ test ngay trên Kaggle (~20–30 phút) nếu chưa có;
#                            đặt 0 nếu sẽ tải bộ test từ máy mình lên bằng push_testset.sh
set -euo pipefail

WORK=${WORK:-/kaggle/working}
REPO=github.com/khanhkhmt/ocr-bench.git
CODE=$WORK/ocr-bench
DATA=$WORK/testset
RUNS=$WORK/runs
CONFIG=$WORK/config.yaml
BUILD_TESTSET=${BUILD_TESTSET:-1}

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "GPU"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_cap --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

step "Ổ đĩa (model Hugging Face lưu ở ~/.cache/huggingface, mỗi model 5–20 GB)"
df -h "$WORK" "$HOME" | sed 's/^/  /'

step "Code"
if [ -d "$CODE/.git" ]; then
  git -C "$CODE" pull --ff-only
elif [ -n "${GITHUB_TOKEN:-}" ]; then
  git clone -q "https://x-access-token:${GITHUB_TOKEN}@${REPO}" "$CODE"
  git -C "$CODE" remote set-url origin "https://${REPO}"   # không để token nằm lại trong .git/config
else
  git clone -q "https://${REPO}" "$CODE" || {
    echo "Clone thất bại. Repo private: chạy lại với GITHUB_TOKEN=<token chỉ-đọc> bash setup.sh"; exit 1; }
fi
git -C "$CODE" log --oneline -1

step "Thư viện"
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)
command -v tesseract >/dev/null || (apt-get update -qq && apt-get install -y -qq tesseract-ocr tesseract-ocr-ara >/dev/null)
pip install -q -e "$CODE[teds,hf,tesseract,easyocr,testset]"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| CUDA', torch.cuda.is_available())"

step "Bộ test"
if [ -f "$DATA/manifest.jsonl" ]; then
  echo "Đã có $DATA/manifest.jsonl"
elif [ "$BUILD_TESTSET" = "1" ]; then
  python -m playwright install --with-deps chromium >/dev/null
  ocrbench build-testset "$DATA" --public-n 120 --synth-n 120
else
  echo "Chưa có bộ test. Từ máy của anh chạy: bash scripts/kaggle/push_testset.sh <ssh-host>"
fi

step "Config"
if [ ! -f "$CONFIG" ]; then
  python - "$CODE/configs/kaggle_example.yaml" "$CONFIG" "$DATA/manifest.jsonl" "$RUNS" <<'PY'
import sys, yaml
src, dst, data, runs = sys.argv[1:]
cfg = yaml.safe_load(open(src))
cfg["dataset"], cfg["output_dir"] = data, runs
yaml.safe_dump(cfg, open(dst, "w"), allow_unicode=True, sort_keys=False)
PY
  echo "Đã tạo $CONFIG"
else
  echo "Giữ nguyên $CONFIG"
fi
[ -f "$DATA/manifest.jsonl" ] && ocrbench validate --config "$CONFIG" | head -3

step "Xong. Bước tiếp theo"
cat <<TXT
  tmux new -s bench                 # mọi lệnh chạy lâu phải nằm trong tmux (mất SSH vẫn chạy tiếp)
  cd $WORK
  ocrbench vram --config $CONFIG
  ocrbench run  --config $CONFIG --per-category 1 --gpus 0,1
  (thoát tmux mà không dừng: Ctrl-b rồi d · vào lại: tmux attach -t bench)
TXT
