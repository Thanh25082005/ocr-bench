# Prompt tiếp tục benchmark OCR (đọc lại file này BẤT CỨ KHI NÀO bạn không chắc mình đang làm gì)

> File này dành cho agent chạy trên server Kaggle. Nó tự đủ ngữ cảnh: bạn không cần nhớ gì từ các phiên trước.
> Nếu cuộc hội thoại của bạn bị nén / mất ngữ cảnh: dừng tay, làm lại **mục 5 (Khôi phục ngữ cảnh)** rồi mới làm tiếp.

---

## 1. Bối cảnh dự án (vì sao bạn ở đây)

- Người dùng cần **chọn model OCR tốt nhất** để chuyển tài liệu **tiếng Anh + tiếng Ả Rập** (văn bản in, bảng, hóa đơn,
  biểu mẫu, **chữ viết tay**, tài liệu dài nhiều trang, ảnh chụp xấu) thành văn bản / DOCX. Có người duyệt lại kết quả.
- Bên đối tác **không cấp dữ liệu thật**, nên bộ test là **dữ liệu thay thế**: 2.040 mẫu (dữ liệu công khai + tài liệu
  tự sinh), chia `dev` / `holdout`. Bạn **chỉ dùng `dev`**. `holdout` để con người chấm lần cuối — **cấm đụng vào**.
- Tool đánh giá là `ocrbench` (repo GitHub private `khanhkhmt/ocr-bench`). Nó chạy model, chấm điểm (CER, TEDS, ô đúng
  vị trí...), ghi benchmark và **tự đẩy kết quả lên nhánh `results` trên GitHub** (vì server Kaggle có thể sập / hết
  phiên 12 giờ bất cứ lúc nào).
- Người dùng **không chấp nhận đánh đổi độ chính xác** để chạy nhanh hơn (không giảm độ phân giải ảnh, không nén model
  nếu chưa hỏi).

## 2. Nhiệm vụ của bạn

Chạy lần lượt **12 mục** (11 model + 1 thí nghiệm tiền xử lý ảnh) trong `goal/queue.yaml` trên split `dev`. Mỗi model: chạy → chấm → ghi `BENCHMARK.md` → đẩy
lên GitHub → xóa trọng số khỏi ổ đĩa → sang model sau. **Xong** khi và chỉ khi lệnh

```bash
ocrbench goal-check --config /kaggle/working/config.yaml
```

in ra dòng `GOAL: ĐẠT`. Không có định nghĩa "xong" nào khác.

## 3. Bản đồ hệ thống

| Thứ | Đường dẫn / lệnh |
|---|---|
| Code tool (git, nhánh `main`) | `/kaggle/working/ocr-bench` |
| Config (gọi tắt `C`) | `/kaggle/working/config.yaml` |
| Kết quả (`OUT`) | `/kaggle/working/runs/dev/<tên model>/predictions.jsonl`, `meta.json`, `run.log` |
| Nhật ký của bạn (BẮT BUỘC ghi) | `/kaggle/working/EXPERIMENTS.md` |
| Bảng xếp hạng (do tool sinh) | `/kaggle/working/BENCHMARK.md` |
| Model bỏ qua (nếu có) | `/kaggle/working/goal_skips.yaml` |
| Hàng đợi + khóa (CẤM sửa) | `ocr-bench/goal/queue.yaml`, `ocr-bench/goal/GOAL_LOCK.sha256` |
| Luật chi tiết | `ocr-bench/AGENT_PROMPT.md` (luật L1–L20, mục 6 VRAM, mục 7 sự cố, mục 8 mẫu nhật ký) |
| Quy trình /goal chi tiết | `ocr-bench/docs/GOAL_PROMPT.md` (bước A–G cho mỗi model) |
| Giải thích chỉ số | `ocr-bench/docs/METRICS.md` |
| Tiến độ trên GitHub | `https://github.com/khanhkhmt/ocr-bench/blob/results/status.md` và `.../BENCHMARK.md` |

Lệnh `ocrbench` dùng nhiều nhất:

| Lệnh | Làm gì |
|---|---|
| `ocrbench goal-check --config C` | Trạng thái 12 mục + **VIỆC TIẾP THEO** + `GOAL: ĐẠT/CHƯA ĐẠT` |
| `ocrbench restore --config C` | Kéo kết quả + nhật ký + config từ GitHub về (đầu mỗi phiên) |
| `ocrbench vram --config C --models M [--long]` | VRAM cần, VRAM trống, cấu hình nên dùng |
| `ocrbench run --config C --models M --categories ... [--per-category N] --gpus 0,1` | Chạy model (trong tmux) |
| `ocrbench benchmark --config C` | Sinh lại `BENCHMARK.md` |
| `ocrbench status --config C --push --note "..."` | Ghi `status.md`, đẩy mọi thứ lên GitHub (phải thấy `✔ STATUS`) |
| `ocrbench clean-cache --config C --item M` | Xóa trọng số model (tool từ chối nếu benchmark chưa lên GitHub) |

Hành vi tự động của tool (bạn không cần làm tay):
- `run` **tự bỏ qua mẫu đã có kết quả** → chạy lại đúng lệnh cũ là chạy tiếp từ chỗ dừng.
- `run` **tự đẩy status lên GitHub** khi model bắt đầu, kết thúc, và mỗi 30 phút.
- Model vừa 1 GPU + `--gpus 0,1` → **tự chia mẫu cho 2 GPU** (dòng `⇉ ... chia mẫu cho 2 GPU`).
- Đổi tham số của model đã có kết quả mà giữ tên cũ → tool **từ chối chạy** (mã 3). Muốn cấu hình khác: tên mới.

## 4. Trạng thái tại thời điểm viết file này (2026-10-03 11:30 UTC)

| # | Model | Trạng thái | Ghi chú |
|---:|---|---|---|
| 1 | tesseract | ✔ xong | CER văn bản 30%, không đọc được chữ viết tay |
| 2 | easyocr | ✔ xong | CER văn bản 33% |
| 3 | sherif_handwriting | nhóm thường ✔ 923/923 · tài liệu dài: chạy lại đọc từng trang | Đứng đầu (CER văn bản 15,7%; chữ viết tay 7–11%). Đọc cả tài liệu một lượt: 5/24 hết VRAM, 3/24 bị cắt ở 8192 token |
| 3b | sherif_handwriting_pre | chưa chạy | = sherif + **tiền xử lý ảnh**; chỉ ~224/947 mẫu phải chạy model, còn lại chép kết quả sherif |
| 3c | dots_mocr | chưa chạy | Đọc **bố cục** cả trang (khối + bbox, bảng HTML); ~3 tỉ tham số, vừa 1 T4 |
| 4 | qari_0_4 | chưa chạy | Qwen3-VL-4B + LoRA |
| 5 | amad_vlm6 | chưa chạy | 8,3 tỉ tham số, cần 2 GPU |
| 6–10 | paddleocr_ar, paddleocr_vl, baseer, hunyuan_ocr, surya | chưa chạy | |

Trạng thái THẬT luôn là output của `goal-check` sau khi `restore` — tin nó hơn bảng này.

## 5. Khôi phục ngữ cảnh (đầu mỗi phiên, và mỗi khi bạn không chắc đang làm gì)

```bash
tmux attach -t bench 2>/dev/null || tmux new -s bench      # mọi lệnh run phải ở trong tmux
cd /kaggle/working
ocrbench restore --config C                                 # chỉ cần ở đầu phiên mới
ocrbench goal-check --config C
tail -n 60 /kaggle/working/EXPERIMENTS.md
cat /kaggle/working/runs/status.md | head -30
nvidia-smi --query-gpu=index,memory.used --format=csv
pgrep -af "ocrbench" || echo "không có lệnh ocrbench nào đang chạy"
```

Rồi quyết định theo thứ tự:
1. `goal-check` có `KHÓA KHÔNG HỢP LỆ` → **DỪNG và hỏi**.
2. Có tiến trình `ocrbench run` đang chạy → **đừng chạy lệnh mới**; đợi nó xong (xem `tmux attach -t bench`).
3. Mục nhật ký cuối là một lệnh run chưa ghi "Kết thúc" và không còn tiến trình nào → chạy lại **đúng lệnh đó**.
4. Còn lại → làm đúng dòng `VIỆC TIẾP THEO` của `goal-check`.

Đầu phiên mới: ghi mục nhật ký `## <giờ UTC> — Phiên mới — khôi phục từ GitHub, tiếp tục: <việc>` rồi
`ocrbench status --config C --push --note "phiên mới"`.

## 6. Quy trình cho MỘT model `M` (chi tiết đầy đủ: docs/GOAL_PROMPT.md bước A–G)

`<nhóm thường>` = `pub_handwriting_ar,pub_handwriting_en,pub_printed_ar,pub_tables_ar,pub_tables_en,syn_degraded,syn_form_ar,syn_form_en,syn_invoice_ar,syn_invoice_en,syn_invoice_mixed,syn_text_ar,syn_text_en,syn_text_mixed`

**A. Chuẩn bị:** mục `M` trong config phải `enabled: true`. `nvidia-smi` + `ocrbench vram --config C --models M`. Nếu
vram in "→ Thêm mục sau vào config": chép NGUYÊN VĂN vào config, tên mới đó là biến thể chạy `V`; không thì `V = M`.

**B. Chạy thử:** `ocrbench run --config C --models V --per-category 1 --categories <nhóm thường> --gpus 0,1`.
Đạt khi: mã thoát 0, cột Lỗi/rỗng = 0 (trừ mẫu rỗng của OCR truyền thống), chữ đọc được. Không đạt: sửa theo AGENT_PROMPT
mục 7 bằng biến thể mới `M__fix1`/`M__fix2`/`M__fix3`; hết bộ nhớ: AGENT_PROMPT mục 6.2 (chỉ tự làm bậc A, B).

**C. Nhóm thường đầy đủ:** `ocrbench run --config C --models V --categories <nhóm thường> --gpus 0,1`, rồi `goal-check`.

**D. Tài liệu dài — CHỈ 12 mẫu cố định mỗi nhóm (BẮT BUỘC `--per-category 12`):**
- Adapter `hf_vlm`: tạo mục `V__pp__long` = chép mục `V` (giữ `gpus`), đặt `max_new_tokens: 8192`,
  `stop_on_loop: true`, `multi_page: per_page` (đọc từng trang ở độ phân giải đầy đủ rồi nối — đọc cả tài liệu một
  lượt thì tràn VRAM). **CẤM** `max_pixels`, `max_image_side`. Chạy:
  `ocrbench run --config C --models V__pp__long --categories syn_longtable,syn_longtext --per-category 12 --gpus 0,1`
  Có mẫu `OutOfMemoryError` → biến thể `V__pp2g__long` (như trên + `gpus: 2`), chạy lại cả 24 mẫu. Vẫn hết → DỪNG hỏi.
- Adapter khác: `ocrbench run --config C --models V --categories syn_longtable,syn_longtext --per-category 12 --gpus 0,1`

**E. Ghi benchmark (làm liền, không chen việc khác):** `ocrbench benchmark --config C` → ghi nhật ký (chép nguyên văn
dòng của `M` trong bảng tổng hợp) → `ocrbench status --config C --push --note "benchmark: M"` → thấy `✔ STATUS`.

**F. Xóa model:** `ocrbench clean-cache --config C --item M` → trong config đặt `enabled: false` cho `M` và mọi `M__...`
→ `df -h /kaggle/working ~` (ghi số GB trống vào nhật ký) → `ocrbench status --config C --push --note "xong M"`.

**G.** `ocrbench goal-check --config C` → `M` phải `HOÀN THÀNH` → model tiếp theo.

## 7. Ghi chú riêng từng model còn lại

| Model | Ghi chú |
|---|---|
| **sherif_handwriting** (đang dở) | Nhóm thường đã xong. Biến thể `sherif_handwriting__sl__long` (đọc cả tài liệu một lượt) đã chạy 24/24 nhưng **5 mẫu hết VRAM** ở bước prefill → đặt `enabled: false` cho nó và cho `sherif_handwriting__long`, KHÔNG xóa kết quả. Tạo `sherif_handwriting__pp__long` theo bước D (chép mục `sherif_handwriting`, giữ `gpus: 1` → tool tự chia 2 GPU). Ước ~1 giờ. Rồi E, F. |
| **sherif_handwriting_pre** (ngay sau sherif) | Thí nghiệm **tiền xử lý ảnh** (`ocrbench/preprocess.py`: làm phẳng nền, tăng tương phản, chỉnh nghiêng, phóng ảnh nhỏ — chỉ chạy khi ảnh có lỗi đó). Tạo 2 mục trong config: (1) `sherif_handwriting_pre` = chép **NGUYÊN** mục `sherif_handwriting` trong config đang chạy (mọi params, kể cả `model_kwargs`), chỉ thêm `preprocess: true` và `preprocess_reuse: sherif_handwriting`, `enabled: true`; (2) `sherif_handwriting_pre__pp__long` = chép NGUYÊN mục tài liệu dài mà sherif đã dùng xong (`sherif_handwriting__pp__long` hoặc `__pp2g__long`), thêm `preprocess: true` và `preprocess_reuse: <tên mục đó>`. Tool tự lấy kết quả sherif cho mẫu ảnh không đổi (log: "dùng lại kết quả ... còn N mẫu phải chạy model"); khác params → tool từ chối (mã 4): sửa cho giống hệt rồi chạy lại. Trọng số sherif đã bị clean-cache xóa ở bước F → tải lại (~7 GB), bình thường. Đi đủ A→G như model thường (bước D: lệnh run với biến thể `sherif_handwriting_pre__pp__long`, `--per-category 12`; tài liệu dài ảnh sạch nên thường xong ngay không cần nạp model). CẤM sửa ngưỡng tiền xử lý, CẤM tự tạo biến thể tiền xử lý cho model khác. |
| **dots_mocr** (sau sherif_handwriting_pre) | Chép NGUYÊN mục `dots_mocr` từ `ocr-bench/configs/kaggle_example.yaml` vào config (prompt nhiều dòng, `patches: [dots_vision]`, `output_format: layout_json`, `model_class: AutoModelForCausalLM`, `local_alias: DotsMOCR` — KHÔNG sửa gì). Model vừa 1 GPU → `--gpus 0,1` tự chia 2 GPU. Ở bước B (chạy thử) kiểm tra thêm: trong `predictions.jsonl`, `extra.layout` phải là `ok` hoặc `repaired` ở đa số mẫu; chữ đọc được (không rỗng, không ký tự rác). Nạp model lỗi vì phiên bản transformers (ImportError / AttributeError trong `modeling_dots_*.py` hoặc `configuration_dots.py`) → được tạo venv riêng: `python -m pip install -q uv && uv venv /kaggle/working/venvs/dots --system-site-packages && uv pip install -p /kaggle/working/venvs/dots/bin/python "transformers==4.56.1" -e /kaggle/working/ocr-bench`, rồi thêm `python: /kaggle/working/venvs/dots/bin/python` vào mục (đổi tên thành `dots_mocr__venv`). **Mọi lỗi khác của dots_mocr (lỗi trong `adapters/patches.py`, `extra.layout = failed` ở đa số mẫu, chữ rác, hết VRAM): KHÔNG bỏ qua model, DỪNG và báo** kèm 40 dòng cuối run.log — người dùng sẽ sửa code. Tài liệu dài: biến thể `dots_mocr__pp__long` theo bước D. |
| **qari_0_4** | `hf_vlm`, Qwen3-VL-4B + LoRA (`adapter_id`). Lỗi nạp model nhắc `qwen3_vl` / model type không hỗ trợ → được chạy ĐÚNG MỘT lệnh `python -m pip install -q -U "transformers>=4.57"` rồi thử lại (cấm đụng torch/CUDA). Model này hay lặp vòng: đo nguyên cấu hình gốc, KHÔNG tự thêm repetition_penalty. Ước 4–5 giờ. |
| **amad_vlm6** | 8,3 tỉ tham số, không vừa 1 T4: dùng mục `amad_vlm6__2gpu` mà vram in ra (bậc B). Hết bộ nhớ ở bậc B → DỪNG và hỏi (bậc C trở đi giảm độ chính xác). Không chia mẫu cho 2 GPU được (1 bản model chiếm cả 2 card) → chậm, ước 6–10 giờ. Mục `amad_vlm6_4bit` trong config KHÔNG thuộc hàng đợi — để nguyên `enabled` như cũ, không chạy. |
| **paddleocr_ar** | Cần cài trước: `python -m pip install -q "paddleocr>=3.3"`. Config dùng `engine: transformers` (không cần paddlepaddle-gpu). Xung đột thư viện → venv riêng theo README mục 4. |
| **paddleocr_vl** | Cùng gói `paddleocr`. VLM ~0,9 tỉ tham số. |
| **baseer** | `hf_vlm`, Qwen2-VL-2B, trả JSON (`json_field: full_text` đã có sẵn trong config). |
| **hunyuan_ocr** | Đang `enabled: false` trong config → bật lên. Cần `trust_remote_code` (đã có). Có thể cần transformers mới; nạp lỗi 3 lần → bỏ qua với `LOAD_FAILED_3X`. Giấy phép riêng của Tencent: dùng nội bộ, không thương mại → được chạy. |
| **surya** | Đang `enabled: false`. Cần venv riêng: `python -m pip install -q uv && uv venv /kaggle/working/venvs/surya --system-site-packages && uv pip install -p /kaggle/working/venvs/surya/bin/python surya-ocr -e /kaggle/working/ocr-bench`, rồi bật. Surya 2 tự khởi động vLLM / llama-server, vLLM không ổn trên T4 → hỏng sau 3 lần sửa thì bỏ qua (`DEPENDENCY_CONFLICT` hoặc `LOAD_FAILED_3X`). |

Bỏ qua một model: CHỈ với 4 lý do `OOM_ALL_LEVELS`, `LOAD_FAILED_3X`, `GATED_OR_LICENSE`, `DEPENDENCY_CONFLICT`, tối đa 3
model, ghi vào `/kaggle/working/goal_skips.yaml` theo mẫu docs/GOAL_PROMPT.md mục 4, rồi `benchmark` → `status --push` →
`clean-cache --item M --force`. (Lưu ý: `OOM_ALL_LEVELS` cần người dùng đồng ý thử bậc C–H trước; chưa hỏi thì DỪNG hỏi.)

## 8. Luật cứng (tóm tắt — bản đầy đủ: AGENT_PROMPT.md mục 3)

**CẤM:**
- dùng holdout (`--split holdout`, `--confirm-holdout`), mở đáp án holdout;
- sửa dữ liệu test, `goal/queue.yaml`, `goal/GOAL_LOCK.sha256`, code tool (`ocrbench/`, `tests/`, `docs/`, `scripts/`),
  mục `normalization` trong config, `BENCHMARK.md`; chạy `goal-check --write-lock`;
- đổi tham số của model đã có kết quả mà giữ tên cũ; xóa thư mục kết quả; xóa trọng số bằng `rm`;
- tự bật `preprocess` cho model nào ngoài `sherif_handwriting_pre`, hoặc đổi ngưỡng tiền xử lý;
- giảm `max_new_tokens`, đổi `dtype: float32`, thêm `max_pixels`/`max_image_side`, nén model (bậc C–H) khi chưa hỏi;
- cài/nâng/hạ `torch`, CUDA, driver; gửi ảnh/chữ của bộ test ra ngoài (API bên thứ ba);
- tự tính / làm tròn số liệu — mọi con số chép nguyên văn từ output của tool;
- kill tiến trình không do bạn khởi động; chạy `ocrbench run` ngoài tmux.

**PHẢI:**
- chạy `goal-check` sau MỖI lệnh run và làm theo `VIỆC TIẾP THEO`;
- ghi nhật ký `/kaggle/working/EXPERIMENTS.md` sau MỖI lệnh run (mẫu: AGENT_PROMPT mục 8), rồi `status --push`;
- sau mỗi model kết thúc: thấy dòng `✔ STATUS`.

## 9. DỪNG và hỏi khi

1. `goal-check` báo `KHÓA KHÔNG HỢP LỆ` hoặc vượt giới hạn bỏ qua model.
2. Đẩy status thất bại 2 lần liên tiếp, hoặc `restore` báo `✘ KHÔI PHỤC THẤT BẠI`.
3. Cần dùng bậc VRAM C trở đi (giảm ảnh / nén / đẩy sang RAM).
4. Ổ đĩa còn dưới 15 GB sau khi đã `clean-cache`.
5. Phiên Kaggle còn dưới 1 giờ (giờ bắt đầu phiên do người dùng cho biết). Trước khi dừng: push kết quả mới nhất.
6. Lỗi không có trong bảng sự cố (AGENT_PROMPT mục 7), hoặc tình huống tài liệu không nói tới.

Mẫu tin nhắn: `DỪNG: <lý do>. goal-check: <dán nguyên văn output>. Cần anh: <việc cụ thể>.`

## 10. Khi `GOAL: ĐẠT`

```bash
ocrbench benchmark --config C
ocrbench status --config C --push --note "GOAL: ĐẠT"
```

Dán nguyên văn **bảng tổng hợp** trong `BENCHMARK.md` + link
`https://github.com/khanhkhmt/ocr-bench/blob/results/BENCHMARK.md`. Không tự kết luận chọn model.
