#!/usr/bin/env bash
# Chạy TRÊN MÁY KAGGLE, sau khi đã SSH vào. Chạy lại nhiều lần cũng được (bước nào xong rồi thì bỏ qua).
#
#   GITHUB_TOKEN=<token> bash setup.sh
#
# Token GitHub (fine-grained, chỉ repo ocr-bench, quyền Contents: Read and write) dùng để clone code VÀ để
# tự đẩy status.md + kết quả lên nhánh `results` sau mỗi model (phòng server sập). Token được lưu vào
# ~/.config/ocrbench/github_token (chmod 600) để tmux / agent dùng được; không ghi vào .git/config.
#
# Biến môi trường tùy chọn:
#   WORK=/kaggle/working     thư mục làm việc
#   TESTSET=drive            lấy bộ test khi chưa có:
#                              drive = tải từ Google Drive (scripts/get_testset.py, ~20–25 phút)
#                              build = dựng lại ngay trên Kaggle (~20–30 phút)
#                              none  = không làm gì (sẽ tự đưa lên bằng push_testset.sh)
set -euo pipefail

WORK=${WORK:-/kaggle/working}
REPO=github.com/khanhkhmt/ocr-bench.git
CODE=$WORK/ocr-bench
DATA=$WORK/testset
RUNS=$WORK/runs
CONFIG=$WORK/config.yaml
TESTSET=${TESTSET:-drive}

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "GPU"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_cap --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

step "Ổ đĩa (model Hugging Face lưu ở ~/.cache/huggingface, mỗi model 5–20 GB)"
df -h "$WORK" "$HOME" | sed 's/^/  /'

if [ -n "${GITHUB_TOKEN:-}" ]; then
  mkdir -p "$HOME/.config/ocrbench" && chmod 700 "$HOME/.config/ocrbench"
  ( umask 077; printf '%s' "$GITHUB_TOKEN" > "$HOME/.config/ocrbench/github_token" )
fi

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
python -m pip install -q -e "$CODE[teds,hf,tesseract,easyocr,testset,app]"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| CUDA', torch.cuda.is_available())"

step "Bộ test"
if [ -f "$DATA/manifest.jsonl" ]; then
  echo "Đã có $DATA/manifest.jsonl"
elif [ "$TESTSET" = "drive" ]; then
  python -m pip install -q gdown
  python "$CODE/scripts/get_testset.py" --out "$DATA" || {
    echo "Tải chưa xong. Chạy lại: python $CODE/scripts/get_testset.py --out $DATA"; exit 1; }
elif [ "$TESTSET" = "build" ]; then
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
cfg["status"] = {"push": True, "every_min": 30}  # tự đẩy status.md + kết quả lên nhánh results
yaml.safe_dump(cfg, open(dst, "w"), allow_unicode=True, sort_keys=False)
PY
  echo "Đã tạo $CONFIG"
else
  echo "Giữ nguyên $CONFIG"
fi
[ -f "$DATA/manifest.jsonl" ] && ocrbench validate --config "$CONFIG" | head -3

step "Khôi phục kết quả của các phiên trước (nhánh results trên GitHub)"
if [ -f "$HOME/.config/ocrbench/github_token" ]; then
  ocrbench restore --config "$CONFIG" || echo "Chưa khôi phục được (lần chạy đầu tiên thì bỏ qua)."
  ocrbench status --config "$CONFIG" --push --note "setup xong trên $(hostname)" \
    || echo "⚠ Chưa đẩy được status lên GitHub: kiểm tra token có quyền Contents: Read and write."
else
  echo "Không có token: sẽ KHÔNG tự đẩy kết quả lên GitHub. Chạy lại với GITHUB_TOKEN=<token> bash setup.sh"
fi

step "Xong. Bước tiếp theo"
cat <<TXT
  tmux new -s bench                 # mọi lệnh chạy lâu phải nằm trong tmux (mất SSH vẫn chạy tiếp)
  cd $WORK
  cat $RUNS/status.md               # tiến độ hiện tại (đã khôi phục nếu có phiên trước)
  ocrbench vram --config $CONFIG
  ocrbench run  --config $CONFIG --per-category 1 --gpus 0,1
  (thoát tmux mà không dừng: Ctrl-b rồi d · vào lại: tmux attach -t bench)
  Theo dõi từ xa: https://github.com/khanhkhmt/ocr-bench/blob/results/status.md
TXT
