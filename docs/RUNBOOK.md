# Runbook: chạy benchmark trên Kaggle qua SSH, có agent, chịu được server sập

Tài liệu cho **người vận hành**. Phần việc của agent nằm trong `AGENT_PROMPT.md`.

```
Máy của anh ──SSH──▶ Server Kaggle (2× T4)
                       ├─ /kaggle/working/ocr-bench      code (clone từ GitHub, nhánh main)
                       ├─ /kaggle/working/testset        bộ test (tải từ Google Drive)
                       ├─ /kaggle/working/config.yaml
                       ├─ /kaggle/working/runs/          kết quả + status.md
                       └─ /kaggle/working/EXPERIMENTS.md nhật ký của agent
                                │
                                └── tự đẩy sau mỗi model + mỗi 30 phút ──▶ GitHub, nhánh `results`
```

Server sập thì mất toàn bộ `/kaggle/working`, nhưng **nhánh `results` trên GitHub vẫn còn**. Phiên sau chạy
`setup.sh` là kết quả và nhật ký được kéo về, agent làm tiếp đúng chỗ dừng.

---

## Bước 0 — Chuẩn bị một lần

**Token GitHub.** Tạo tại GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens**:
- Repository access: **Only select repositories** → `khanhkhmt/ocr-bench`
- Permissions → Repository → **Contents: Read and write** (cần quyền ghi để đẩy kết quả)
- Expiration: ngắn thôi (vd. 30 ngày)

Token chỉ dùng được cho repo này, nên lộ ra thì thiệt hại cũng giới hạn trong repo này. Đừng dán token vào chat,
issue hay file nào được commit.

**Bộ test trên Google Drive.** Thư mục `testset` để chế độ *Bất kỳ ai có đường liên kết*. ID đã ghi sẵn trong
`scripts/get_testset.py`.

## Bước 1 — Mở server Kaggle và SSH vào

1. Mở notebook, *Settings*: **Accelerator = GPU T4 x2**, **Internet = On**.
2. Chạy cell khởi động sshd + tunnel của anh. Ghi lại **giờ bắt đầu phiên** (phiên tối đa 12 giờ).
3. Từ máy anh: `ssh kaggle-ngrok`.
   Nếu báo `REMOTE HOST IDENTIFICATION HAS CHANGED`: chạy `!ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`
   trong notebook. **Chỉ khi** dấu vân tay khớp với dấu vân tay SSH báo, và địa chỉ ngrok đúng, thì mới chạy
   `ssh-keygen -R '[<host>]:<port>'` trên máy anh rồi kết nối lại.

## Bước 2 — Lấy code từ GitHub và cài đặt (trên server)

```bash
tmux new -s setup
cd /kaggle/working
read -rs GITHUB_TOKEN && export GITHUB_TOKEN          # dán token rồi Enter (không hiện ra màn hình, không lưu vào history)
curl -fsSL -H "Authorization: token $GITHUB_TOKEN" \
  https://raw.githubusercontent.com/khanhkhmt/ocr-bench/main/scripts/kaggle/setup.sh -o setup.sh
bash setup.sh
```

`setup.sh` làm lần lượt (chạy lại bao nhiêu lần cũng được):

| Bước | Việc | Thời gian |
|---|---|---|
| GPU, Ổ đĩa | in thông tin GPU, tiến trình đang chiếm GPU, dung lượng trống | vài giây |
| Code | clone `main` (hoặc `git pull` nếu đã có); token **không** lưu vào `.git/config` | vài giây |
| Thư viện | `tmux`, `tesseract` (+ tiếng Ả Rập), `pip install -e ocr-bench[...]` | 2–5 phút |
| Bộ test | tải từ Drive vào `/kaggle/working/testset`, kiểm tra đủ 4.268 file và dấu vân tay `aa8fdd44e1715845` | 20–25 phút |
| Config | tạo `/kaggle/working/config.yaml`, bật `status.push` | vài giây |
| Khôi phục | `ocrbench restore` kéo kết quả + nhật ký từ nhánh `results` (phiên đầu thì chưa có gì), rồi đẩy thử một status | vài giây |

Token được lưu ở `~/.config/ocrbench/github_token` (chmod 600) để tmux và agent dùng được. File này mất theo server.

**Kiểm tra trước khi giao cho agent:**

```bash
ocrbench validate --config /kaggle/working/config.yaml | head -2   # dấu vân tay phải là aa8fdd44e1715845
ocrbench status --config /kaggle/working/config.yaml --push --note "kiểm tra"   # phải in ✔ STATUS
```

Mở `https://github.com/khanhkhmt/ocr-bench/blob/results/status.md` trên trình duyệt, phải thấy giờ cập nhật mới.

## Bước 3 — Chạy agent (trên server, trong tmux)

```bash
tmux new -s agent
cd /kaggle/working/ocr-bench
```

Cài và đăng nhập agent (ví dụ Claude Code; xem tài liệu chính thức nếu lệnh thay đổi):

```bash
curl -fsSL https://claude.ai/install.sh | bash     # hoặc: npm install -g @anthropic-ai/claude-code
claude                                             # lần đầu: làm theo hướng dẫn đăng nhập (mở URL trên máy anh)
```

Điền mục 2 của `AGENT_PROMPT.md` (giờ bắt đầu phiên, ngân sách GPU, danh sách nhóm), rồi giao việc:

> Đọc và làm đúng theo `/kaggle/working/ocr-bench/AGENT_PROMPT.md`. Config: `/kaggle/working/config.yaml`.
> OUT = `/kaggle/working/runs`, WORK = `/kaggle/working`. Phiên Kaggle bắt đầu lúc `<giờ>`. Ngân sách: `<số>` giờ GPU.

Agent sẽ hỏi quyền trước mỗi lệnh shell. Muốn để agent chạy không cần người bấm duyệt thì phải cho phép trước các
lệnh nó dùng (`ocrbench`, `nvidia-smi`, `python`, `cat`, `tail`...). Hãy cân nhắc kỹ trước khi tắt hẳn việc hỏi quyền.

Thoát tmux mà không dừng agent: `Ctrl-b` rồi `d`. Vào lại: `tmux attach -t agent`.

## Bước 4 — Theo dõi từ xa (không cần SSH)

- **`status.md`**: https://github.com/khanhkhmt/ocr-bench/blob/results/status.md. Có: model nào đang chạy, đã chạy
  bao nhiêu mẫu, số lỗi, CER tạm tính theo nhóm, VRAM, mục nhật ký gần nhất.
- Cập nhật **mỗi khi một model chạy xong** và **mỗi 30 phút** trong lúc chạy, cùng với mỗi lần agent ghi nhật ký.
- Lịch sử: mỗi lần cập nhật là một commit trên nhánh `results`, nên xem lại được tiến độ theo thời gian.

Nếu `status.md` **không đổi quá 45 phút** trong khi đáng lẽ đang chạy, nhiều khả năng server đã sập hoặc agent
đang dừng chờ hỏi: SSH vào xem `tmux attach -t agent`.

## Bước 5 — Khi server sập / hết phiên 12 giờ

1. Mở phiên Kaggle mới (bước 1), SSH vào.
2. Chạy lại **đúng** bước 2. `setup.sh` tự `restore`: kết quả thô, `EXPERIMENTS.md`, `config.yaml` (cả các mục
   model agent đã thêm) và `reports/` được kéo về.
3. Chạy agent (bước 3) với lời giao việc:
   > Phiên mới sau khi server dừng. Làm mục 4.00 của `AGENT_PROMPT.md` rồi tiếp tục.

Mất mát tối đa: phần đang chạy dở trong **30 phút cuối** trước khi sập (thường chỉ vài chục mẫu). Tool tự bỏ qua
những mẫu đã có kết quả, nên không chạy lại gì thừa.

## Bước 6 — Lấy kết quả về máy

```bash
git -C ~/ocr-bench fetch origin results
git -C ~/ocr-bench worktree add ../ocr-bench-results results     # thư mục ~/ocr-bench-results chứa mọi kết quả
```

Hoặc xem thẳng trên GitHub, nhánh `results`: `status.md`, `EXPERIMENTS.md`, `FINAL_REPORT.md`,
`runs/dev/_report/report.md`, `runs/dev/_report/decision_*.md`.

---

## Sự cố thường gặp

| Hiện tượng | Xử lý |
|---|---|
| `setup.sh` báo clone thất bại | Token sai hoặc thiếu quyền; kiểm tra token có repo `ocr-bench` và Contents: Read and write |
| `✘ ĐẨY STATUS THẤT BẠI ... 403` | Token chỉ có quyền đọc. Tạo token có **Read and write**, chạy lại `setup.sh` |
| `get_testset.py` báo lỗi / chưa đủ file | Chạy lại `python /kaggle/working/ocr-bench/scripts/get_testset.py --out /kaggle/working/testset` (tải tiếp). Bị Drive chặn thì thêm `--workers 4` |
| Dấu vân tay khác `aa8fdd44e1715845` | Bộ test trên Drive khác bộ gốc; đừng chạy, kiểm tra lại thư mục Drive |
| Báo cáo có `⚠ CẢNH BÁO: dữ liệu đã thay đổi` | Kết quả cũ được chạy trên bộ test khác; xem `AGENT_PROMPT.md` mục 9 |
| `ssh` báo `REMOTE HOST IDENTIFICATION HAS CHANGED` | Bước 1, mục 3 |

**Cách cũ, không cần nữa:** `scripts/kaggle/pull_results.sh`, `push_results.sh` và `push_testset.sh` (rsync qua SSH)
vẫn dùng được nếu không muốn đẩy kết quả lên GitHub, nhưng phải tự chạy định kỳ.
