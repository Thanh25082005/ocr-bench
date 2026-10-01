# Prompt /goal: chạy lần lượt ~10 model OCR cho tới khi xong

> **Người vận hành:** sau khi `setup.sh` chạy xong (docs/RUNBOOK.md bước 2), mở agent trên server trong tmux rồi đặt goal:
>
> ```
> /goal Làm đúng theo /kaggle/working/ocr-bench/docs/GOAL_PROMPT.md cho tới khi lệnh
> `ocrbench goal-check --config /kaggle/working/config.yaml` in ra dòng `GOAL: ĐẠT`.
> ```
>
> Muốn đổi danh sách model: sửa `goal/queue.yaml` **trước khi bắt đầu**, chạy
> `ocrbench goal-check --config /kaggle/working/config.yaml --write-lock`, rồi commit cả hai file lên GitHub.

---

## 0. Bạn là ai, làm gì, khi nào xong

Bạn chạy **lần lượt từng model** trong `goal/queue.yaml` trên toàn bộ split `dev`. Mỗi model chạy xong thì:
**(1)** ghi vào bảng benchmark, **(2)** đẩy lên GitHub, **(3)** xóa trọng số model khỏi ổ đĩa, rồi mới sang model tiếp theo.

**Xong** = lệnh `ocrbench goal-check --config C` in ra `GOAL: ĐẠT`. Không có định nghĩa "xong" nào khác.
Tự cho là đã xong khi lệnh đó chưa in `GOAL: ĐẠT` là vi phạm.

Ký hiệu dùng trong file này:
- `C` = `/kaggle/working/config.yaml`
- `OUT` = `/kaggle/working/runs`
- `WORK` = `/kaggle/working`
- `<nhóm thường>` = mọi nhóm trừ `syn_longtable,syn_longtext`. Lấy danh sách từ `ocrbench validate --config C`
  và ghi một lần vào đầu nhật ký.

## 1. Luật

**Mọi luật L1–L20 trong `AGENT_PROMPT.md` vẫn bắt buộc.** Hãy đọc file đó trước. Khi phần **quy trình** (mục 4 của
AGENT_PROMPT) khác với file này, làm theo file này. Thêm các luật sau:

| # | Luật |
|---|---|
| G1 | **CẤM** sửa `goal/queue.yaml`, `goal/GOAL_LOCK.sha256`, và **CẤM** chạy `goal-check --write-lock`. Tool phát hiện được: khóa hỏng thì goal không bao giờ đạt. |
| G2 | **Mỗi lúc chỉ làm một model**, đúng thứ tự trong hàng đợi. Model nào thì luôn do dòng `VIỆC TIẾP THEO` của `goal-check` quyết định. |
| G3 | **Model chưa xong thì CẤM sang model sau.** "Xong" nghĩa là `goal-check` báo model đó `HOÀN THÀNH` hoặc `BỎ QUA (...)`. |
| G4 | **Ngay khi một model chạy xong** (`goal-check` báo `CHẠY XONG, CHƯA GHI BENCHMARK`), **PHẢI** làm liền mạch, không chen việc khác: `ocrbench benchmark` → ghi nhật ký → `ocrbench status --push` (thấy `✔ STATUS`) → `ocrbench clean-cache`. |
| G5 | **CẤM** xóa trọng số bằng `rm` hay bất kỳ cách nào khác ngoài `ocrbench clean-cache`. Tool từ chối xóa khi kết quả chưa lên GitHub; **CẤM** dùng `--force` trừ model đã có trong `goal_skips.yaml`. |
| G6 | **CẤM** xóa hay sửa kết quả (`OUT/...`) và `BENCHMARK.md` bằng tay. `BENCHMARK.md` chỉ được sinh bằng `ocrbench benchmark`. |
| G7 | **CẤM** bỏ qua model khi chưa đi hết quy trình sửa lỗi (AGENT_PROMPT mục 6 và 7). Bỏ qua **CHỈ ĐƯỢC** với 4 lý do ở mục 4, và tối đa 3 model. |
| G8 | Sau **mỗi** lệnh `ocrbench run` kết thúc, **PHẢI** chạy `ocrbench goal-check --config C` và làm theo đúng dòng `VIỆC TIẾP THEO`. |

## 2. Bắt đầu (và bắt đầu lại sau khi server sập)

```bash
tmux attach -t bench 2>/dev/null || tmux new -s bench     # mọi lệnh run phải ở trong tmux (L17)
cd /kaggle/working
ocrbench restore --config C
ocrbench goal-check --config C
tail -n 40 WORK/EXPERIMENTS.md 2>/dev/null || echo "chưa có nhật ký"
```

- Dòng `KHÓA KHÔNG HỢP LỆ` xuất hiện: **DỪNG và hỏi** (không tự sửa).
- Còn lại: làm theo `VIỆC TIẾP THEO`. Nếu nhật ký cho thấy một lệnh `ocrbench run` đang chạy dở, chạy lại **đúng
  lệnh đó**; tool tự bỏ qua những mẫu đã có.

## 3. Vòng lặp cho MỘT model (gọi là `M`)

### Bước A — Chuẩn bị

1. Trong config, mục `M` phải có `enabled: true`. Nếu đang là `false` (vd. `hunyuan_ocr`, `surya`), đổi thành `true`.
   Với `surya`: cài venv riêng theo README mục 4 rồi mới bật.
2. Kiểm tra GPU và chọn cấu hình:

```bash
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv
ocrbench vram --config C --models M
```

3. `vram` in `→ Thêm mục sau vào config`: chép **nguyên văn** mục đó vào config. Tên mục mới (vd. `M__2gpu`) là
   **biến thể chạy** `V`. `vram` in `Không cần thay đổi`: `V = M`. Model nhỏ không cần kiểm tra VRAM: `V = M`.

### Bước B — Chạy thử 1 mẫu mỗi nhóm

```bash
ocrbench run --config C --models V --per-category 1 --categories <nhóm thường> --gpus 0,1
```

Kiểm tra 4 điều theo AGENT_PROMPT mục 4.1. Không đạt thì:
- Hết bộ nhớ → AGENT_PROMPT mục 6.2: tạo biến thể bậc kế tiếp; biến thể mới thành `V`.
- Lỗi khác → AGENT_PROMPT mục 7: sửa bằng **biến thể mới** tên `M__fix1`, `M__fix2`, `M__fix3` (L5: cấm sửa mục đã có kết quả).
- Sau 3 lần sửa vẫn không đạt → mục 4 (bỏ qua với `LOAD_FAILED_3X`).

### Bước C — Chạy đủ các nhóm thường

```bash
ocrbench run --config C --models V --categories <nhóm thường> --gpus 0,1
ocrbench goal-check --config C
```

- Báo `ĐANG CHẠY nhóm thường (...)` mà đã hết mẫu (số đã chạy = tổng) nhưng lỗi > 1%: chạy lại với `--retry-errors` **một lần**.
  Vẫn > 1%: xem `OUT/dev/V/run.log`, sửa theo mục 7 bằng biến thể mới, chạy lại bước C với biến thể đó.
- Model chạy lâu (hàng giờ) là bình thường: cứ 30 phút tool tự đẩy tiến độ lên GitHub.

### Bước D — Tài liệu dài (`syn_longtable`, `syn_longtext`)

- **Adapter `hf_vlm`:** tạo mục mới tên `V__long`: chép nguyên mục `V`, đặt `max_new_tokens: 8192`, thêm
  `processor_kwargs: {max_pixels: 1003520}`. Chạy `ocrbench vram --config C --models V__long --long`, xử lý theo mục 6
  nếu không vừa (tên biến thể vẫn phải **kết thúc bằng `__long`**). Rồi:

```bash
ocrbench run --config C --models V__long --categories syn_longtable,syn_longtext --gpus 0,1
```

- **Adapter khác** (`tesseract`, `easyocr`, `paddleocr`, `paddleocr_vl`, `surya`): chạy chính `V`:

```bash
ocrbench run --config C --models V --categories syn_longtable,syn_longtext --gpus 0,1
```

Sau đó `ocrbench goal-check --config C`. Phải thấy `CHẠY XONG, CHƯA GHI BENCHMARK`.

### Bước E — Ghi benchmark và đẩy lên GitHub (G4: làm liền, không chen việc khác)

```bash
ocrbench benchmark --config C
```

Ghi một mục nhật ký vào `WORK/EXPERIMENTS.md` (mẫu ở AGENT_PROMPT mục 8). Dòng "Số liệu" chép **nguyên văn** dòng
của `M` trong bảng tổng hợp mà `benchmark` vừa in. Rồi:

```bash
ocrbench status --config C --push --note "benchmark: M"
ocrbench goal-check --config C
```

Phải thấy `✔ STATUS` và `goal-check` báo `M` là `ĐÃ LÊN GITHUB, CHƯA XÓA TRỌNG SỐ` (hoặc `HOÀN THÀNH` nếu model không
có trọng số Hugging Face). Không thấy `✔ STATUS` → thử lại 1 lần → vẫn lỗi: **DỪNG và hỏi** (L19).

### Bước F — Xóa model để chạy model mới

```bash
ocrbench clean-cache --config C --item M
```

Rồi trong config đặt `enabled: false` cho `M` và **mọi** biến thể `M__...`. Kiểm tra ổ đĩa (`df -h /kaggle/working ~`)
và ghi số GB trống vào nhật ký. Chạy `ocrbench status --config C --push --note "xong M, đã xóa trọng số"`.

### Bước G — Sang model tiếp theo

```bash
ocrbench goal-check --config C
```

`M` phải là `HOÀN THÀNH`. Làm lại từ bước A với model ở dòng `VIỆC TIẾP THEO`.

## 4. Bỏ qua một model (chỉ khi thật sự hết cách)

Chỉ có 4 lý do được chấp nhận. Tool kiểm tra và **từ chối** mọi lý do khác:

| `reason_code` | Khi nào được dùng |
|---|---|
| `OOM_ALL_LEVELS` | Đã thử hết các bậc ở AGENT_PROMPT mục 6.2 (đến bậc H) mà vẫn hết bộ nhớ |
| `LOAD_FAILED_3X` | Đã sửa 3 lần theo AGENT_PROMPT mục 7 (`M__fix1..3`) mà vẫn không chạy được |
| `GATED_OR_LICENSE` | Model đòi quyền truy cập / đồng ý giấy phép trên Hugging Face mà con người chưa cấp |
| `DEPENDENCY_CONFLICT` | Xung đột thư viện, venv riêng (README mục 4) cũng không giải quyết được |

Thêm vào `WORK/goal_skips.yaml` (tạo nếu chưa có), **giữ nguyên các mục cũ**:

```yaml
- model: M                          # tên đúng như trong goal/queue.yaml
  reason_code: LOAD_FAILED_3X
  detail: "fix1: input_mode template -> lỗi X; fix2: trust_remote_code -> lỗi Y; fix3: 8bit -> lỗi Z"
  evidence_variant: M__fix3         # mục model có OUT/dev/<tên>/run.log chứng minh lỗi
```

Rồi chạy:

```bash
ocrbench goal-check --config C            # mục bỏ qua phải hợp lệ (không có dòng ✘ goal_skips)
ocrbench benchmark --config C
ocrbench status --config C --push --note "bỏ qua M: <reason_code>"
ocrbench clean-cache --config C --item M --force
```

Đến model bỏ qua thứ 4: `goal-check` báo vượt giới hạn, nên phải **DỪNG và hỏi**.

## 5. Khi nào DỪNG (và chờ con người)

Mọi trường hợp ở AGENT_PROMPT mục 9, cộng thêm:
1. `goal-check` in `KHÓA KHÔNG HỢP LỆ`.
2. `goal-check` in `vượt giới hạn` số model bỏ qua.
3. `clean-cache` từ chối xóa mà bạn không hiểu vì sao.
4. Ổ đĩa còn dưới 15 GB **sau** khi đã `clean-cache` model vừa xong.
5. Phiên Kaggle còn dưới 1 giờ (L18). Trước khi dừng: đảm bảo bước E của model đang làm đã đẩy lên GitHub, nếu model đã chạy xong.

Khi dừng, gửi đúng mẫu: `DỪNG: <lý do>. goal-check: <dán nguyên văn output>. Cần anh: <việc cụ thể>.`

## 6. Khi `GOAL: ĐẠT`

Chạy lần cuối:

```bash
ocrbench benchmark --config C
ocrbench status --config C --push --note "GOAL: ĐẠT"
```

Báo cáo cho con người: dán nguyên văn **bảng tổng hợp** trong `BENCHMARK.md`, kèm link
`https://github.com/khanhkhmt/ocr-bench/blob/results/BENCHMARK.md`. Không tự rút kết luận chọn model; việc đó do con người làm.
