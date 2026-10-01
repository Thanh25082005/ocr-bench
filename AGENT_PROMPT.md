# Prompt cho agent: đánh giá các model OCR bằng `ocrbench`

> **Người giao việc:** điền mọi chỗ `<<...>>` ở mục 2, rồi đưa nguyên file này cho agent (Claude Code hoặc agent khác có
> quyền chạy lệnh shell trên máy có GPU, ví dụ terminal của Kaggle notebook).

---

## 0. Cách đọc file này

- **PHẢI** = bắt buộc làm. **CẤM** = tuyệt đối không làm. **CHỈ ĐƯỢC** = mọi cách khác đều bị cấm.
- Lệnh trong khung `bash` phải chạy **đúng như viết**; chỉ thay các chỗ trong `<...>`.
- Không có quy tắc nào cho phép "tự phán đoán". Gặp tình huống file này không nói tới: **DỪNG và hỏi** (mục 9).
- Trước khi làm bất cứ việc gì, **PHẢI** đọc hết: file này, `README.md`, `docs/METRICS.md`.

## 1. Nhiệm vụ

Chạy các model OCR trong config trên split **dev**, dùng `ocrbench decide` để loại và chọn model **cho từng nhóm tài
liệu**, rồi viết `FINAL_REPORT.md`. Sau đó dừng và chờ con người.

Bạn **KHÔNG** làm các việc sau: chọn ngưỡng pass/fail, chấm trên holdout, fine-tune model, thêm model ngoài config.

## 2. Thông số (người giao việc điền)

| Thông số | Giá trị |
|---|---|
| Thư mục code tool | `<<vd. /kaggle/working/ocr-bench>>` |
| File config (gọi tắt là `C` trong các lệnh) | `<<vd. /kaggle/working/config.yaml>>` |
| Thư mục output (`output_dir` trong config, gọi tắt là `OUT`) | `<<vd. /kaggle/working/runs>>` |
| GPU | `<<vd. 2× T4 16 GB (Compute Capability 7.5: không có bf16, không dùng vLLM)>>` |
| Ngân sách GPU cho toàn bộ nhiệm vụ | `<<vd. 20 giờ GPU>>` |
| Thời lượng tối đa một phiên | `<<vd. 12 giờ>>` |
| Phiên Kaggle hiện tại bắt đầu lúc | `<<vd. 2026-10-02 08:00>>` (phiên tối đa 12 giờ, hết giờ là mất `/kaggle/working`) |
| Nhóm tài liệu dài | `syn_longtable,syn_longtext` |
| Nhóm thường (mọi nhóm trừ nhóm dài) | `<<chép từ kết quả ocrbench validate, phân cách bằng dấu phẩy>>` |

## 3. Luật cứng

Cột "Tool chặn" cho biết tool đã tự chặn hoặc tự phát hiện vi phạm hay chưa. Luật **không** có tool chặn vẫn bắt buộc như nhau.

| # | Luật | Tool chặn |
|---|---|---|
| L1 | **CẤM** dùng split holdout dưới mọi hình thức: không `--split holdout`, không `--confirm-holdout`, không mở ảnh hay đáp án của mẫu holdout. | Có: thiếu `--confirm-holdout` thì từ chối chạy |
| L2 | **CẤM** sửa, xóa, thêm bất kỳ file nào trong bộ dữ liệu (manifest, ảnh, đáp án). Thấy đáp án nghi sai: ghi id vào mục "Nghi vấn dữ liệu" trong nhật ký, rồi đi tiếp. | Có: sửa dữ liệu thì báo cáo in cảnh báo đỏ |
| L3 | **CẤM** sửa code của tool (`ocrbench/`, `tests/`, `docs/`, `README.md`, file này). Tool báo lỗi mà không phải do config: **DỪNG và hỏi**. | Không |
| L4 | **CẤM** sửa mục `normalization` trong config. | Không |
| L5 | **CẤM** đổi bất kỳ thứ gì trong mục `params`, `adapter`, `env` của một model **đã có kết quả**. Muốn thử cấu hình khác: tạo **mục model mới với tên mới** (vd. `baseer__8bit`). | Có: tool từ chối chạy (mã thoát 3) |
| L6 | **CẤM** xóa thư mục kết quả `OUT/dev/<model>`. **Ngoại lệ duy nhất:** ở giai đoạn 1, được xóa thư mục của model đang thử để sửa cấu hình (xem 4.2). | Không |
| L7 | **CẤM** gửi ảnh hoặc chữ trong bộ dữ liệu ra ngoài máy: không gọi API của OpenAI, Gemini, Claude..., không tải lên dịch vụ nào. | Không |
| L8 | **CẤM** tự tính, tự làm tròn hay tự suy ra con số. Mọi con số trong nhật ký và báo cáo **CHỈ ĐƯỢC** chép nguyên văn từ `report.md`, `summary.csv`, `decision_*.md`, `meta.json`, hoặc output của `ocrbench vram`. | Không |
| L9 | **CẤM** tự quyết định loại hay chọn model. **CHỈ ĐƯỢC** dùng kết quả của `ocrbench decide`. | Không |
| L10 | **PHẢI** chạy `ocrbench vram` và `nvidia-smi` trước khi chạy model lần đầu, và **PHẢI** xử lý hết bộ nhớ đúng theo mục 6. | Không |
| L11 | **CẤM** dừng (kill) tiến trình không do bạn khởi động. | Không |
| L12 | **CẤM** đưa bất kỳ chữ nào lấy từ đáp án vào prompt của model. Prompt **CHỈ ĐƯỢC** lấy từ config có sẵn hoặc từ trang Hugging Face của model đó. | Không |
| L13 | **CẤM** cài, nâng cấp hay hạ cấp `torch`, CUDA, driver. Chỉ được cài gói nêu trong `README.md`. Gặp xung đột thư viện: dùng venv riêng theo README mục 4. | Không |
| L14 | **PHẢI** ghi nhật ký (mục 8) ngay sau **mỗi** lệnh `ocrbench run`, trước khi chạy lệnh tiếp theo. | Không |
| L15 | **PHẢI** cộng dồn giờ GPU đã dùng vào nhật ký. Khi tổng đạt **80% ngân sách**: **DỪNG và hỏi**. | Không |
| L16 | **CẤM** giảm `max_new_tokens`, và **CẤM** đổi sang `dtype: float32` để chữa lỗi hết bộ nhớ. | Không |
| L17 | **PHẢI** chạy mọi lệnh `ocrbench run` bên trong phiên tmux `bench` (`tmux new -s bench`, hoặc `tmux attach -t bench` nếu đã có). **CẤM** chạy `ocrbench run` trực tiếp trong phiên SSH. | Không |
| L18 | **CẤM** bắt đầu lệnh `ocrbench run` mới nếu phiên Kaggle còn **dưới 1 giờ** (tính từ giờ bắt đầu ở mục 2). Khi còn dưới 1 giờ: dừng sau lệnh đang chạy, cập nhật nhật ký, rồi **DỪNG và hỏi** để con người kéo kết quả về. | Không |

## 4. Quy trình

Làm **đúng thứ tự**, không bỏ bước. Mỗi giai đoạn có "Điều kiện để sang giai đoạn sau"; chưa đạt thì không được đi tiếp.

### 4.0. Giai đoạn 0: Kiểm tra (không chạy model)

```bash
cd <thư mục code tool>
nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_cap --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
ocrbench validate --config C
ocrbench vram --config C
ocrbench vram --config C --long
```

Ghi vào nhật ký (chép nguyên văn):
1. Bảng GPU và danh sách tiến trình đang chiếm GPU.
2. Bảng số mẫu theo nhóm từ `validate`, và mọi dòng bắt đầu bằng `⚠` hoặc `✘`.
3. Với mỗi model: bậc được đánh dấu `← BẮT ĐẦU TỪ ĐÂY` ở cả hai lần chạy `vram` (thường và `--long`).

**Nếu** `validate` có dòng `✘`, **hoặc** GPU đang có tiến trình không phải của bạn chiếm > 500 MiB: **DỪNG và hỏi**.

**Nếu** `vram` in ra một mục config mới (dòng `→ Thêm mục sau vào config`): chép **nguyên văn** mục đó vào cuối danh
sách `models` trong config, và đặt `enabled: false` cho mục gốc tương ứng.

**Điều kiện sang giai đoạn 1:** `validate` có mã thoát 0, mọi model có trạng thái `vram` đã ghi vào nhật ký.

### 4.1. Giai đoạn 1: Chạy thử, 1 mẫu mỗi nhóm

```bash
ocrbench run --config C --per-category 1 --categories <nhóm thường> --gpus 0,1
```

Với **từng** model, kiểm tra lần lượt 4 câu hỏi. Lấy số liệu từ dòng của model đó trong bảng "Tổng quan" của
`OUT/dev/_report/report.md`:

| # | Kiểm tra | Đạt khi |
|---|---|---|
| 1 | Mã thoát của model (dòng `✔`/`✘` khi lệnh chạy xong) | `✔` (mã 0) |
| 2 | Cột "Lỗi/rỗng" | 0 |
| 3 | Cột "Lặp/thừa" | 0 |
| 4 | Chạy lệnh dưới đây, đọc 3 kết quả đầu | chữ đọc được, đúng ngôn ngữ, không phải ký tự rác |

```bash
python -c "import json; [print(json.loads(l)['id'], '|', json.loads(l)['text'][:300].replace(chr(10),' ')) for l in list(open('OUT/dev/<model>/predictions.jsonl'))[:3]]"
```

**Model đạt cả 4:** sang giai đoạn 2.

**Model không đạt:** tra bảng ở mục 7, thử **một** cách sửa, rồi chạy lại riêng model đó:

```bash
rm -rf OUT/dev/<model>        # chỉ được làm ở giai đoạn 1 (ngoại lệ của L6)
ocrbench run --config C --per-category 1 --categories <nhóm thường> --models <model>
```

Mỗi model **tối đa 3 lần sửa**. Sau lần thứ 3 vẫn không đạt: đặt `enabled: false`, ghi vào nhật ký 3 cách đã thử và
lỗi của từng lần, rồi bỏ qua model đó.

**Điều kiện sang giai đoạn 2:** mọi model `enabled` đều đạt cả 4 kiểm tra.

### 4.2. Giai đoạn 2: Sàng lọc trên 20 mẫu mỗi nhóm

```bash
ocrbench run    --config C --per-category 20 --categories <nhóm thường> --gpus 0,1
ocrbench decide --config C --stage screening --per-category 20 --categories <nhóm thường>
mkdir -p reports && cp -r OUT/dev/_report reports/phase2_screening
```

- Danh sách vào chung kết của từng nhóm **CHỈ ĐƯỢC** lấy từ dòng `**Vào chung kết:**` trong
  `OUT/dev/_report/decision_screening.md`. Chép nguyên văn vào nhật ký.
- Trong cột "Lý do" có `bị cắt` **hoặc** báo cáo có cột "Bị cắt" > 0 cho model nào: tạo mục mới
  `<model>__tok8k` với `max_new_tokens: 8192`, chạy lại **chỉ** mục mới đó với cùng lệnh `run` ở trên (thêm
  `--models <model>__tok8k`), rồi chạy lại lệnh `decide`. Mỗi model chỉ được làm việc này **1 lần**.

**Điều kiện sang giai đoạn 2b:** đã có `decision_screening.md` của nhóm thường, đã chép danh sách chung kết vào nhật ký.

### 4.3. Giai đoạn 2b: Sàng lọc tài liệu dài

Chỉ xét những model **có tên trong danh sách chung kết của ít nhất một nhóm** ở giai đoạn 2.

1. Với mỗi model đó, tạo mục mới `<model>__long`: giữ nguyên mọi thứ của mục gốc, đặt `max_new_tokens: 8192`, thêm
   `processor_kwargs: {max_pixels: 1003520}` (chỉ với adapter `hf_vlm`).
2. Chạy `ocrbench vram --config C --long --models <model>__long`, rồi xử lý theo mục 6 nếu không vừa.
3. Chạy:

```bash
ocrbench run    --config C --per-category 12 --categories syn_longtable,syn_longtext --models <danh sách __long> --gpus 0,1
ocrbench decide --config C --stage screening --per-category 12 --categories syn_longtable,syn_longtext
cp -r OUT/dev/_report reports/phase2b_long
```

4. Chép nguyên văn vào nhật ký: danh sách chung kết của hai nhóm dài, và bảng "Tài liệu dài" trong `report.md`
   (gồm cột Q1–Q4, "Bị cắt", "Hàng sai số cột").

**Điều kiện sang giai đoạn 3:** đã có danh sách chung kết cho **mọi** nhóm (nhóm thường và nhóm dài).

### 4.4. Giai đoạn 3: Chung kết trên toàn bộ dev

Ước lượng giờ GPU theo mục 5 trước. Nếu vượt ngân sách còn lại: **DỪNG và hỏi**.

Với **mỗi nhóm**, chạy các model chung kết **của nhóm đó**:

```bash
ocrbench run --config C --categories <nhóm> --models <danh sách chung kết của nhóm> --gpus 0,1
```

Chạy xong mọi nhóm thì:

```bash
ocrbench decide --config C --stage final
cp -r OUT/dev/_report reports/phase3_final
```

Chép nguyên văn vào nhật ký dòng `**Thắng:**` của từng nhóm trong `decision_final.md`.

**Điều kiện sang giai đoạn 4 hoặc 5:** có `decision_final.md`, và **mọi** nhóm có người thắng. Nhóm nào ghi
`(không có)`: **DỪNG và hỏi**.

### 4.5. Giai đoạn 4: Tinh chỉnh (CHỈ ĐƯỢC làm khi ngân sách còn ≥ 30%)

- Chỉ áp dụng cho **người thắng** của mỗi nhóm.
- Mỗi người thắng tối đa **2 biến thể**, mỗi biến thể một mục mới (L5). **CHỈ ĐƯỢC** đổi đúng một trong các tham số sau:

| Khi báo cáo có | Tham số đổi | Giá trị thử |
|---|---|---|
| CER raw > 2 × CER (norm) trong nhóm đó | `prompt` | prompt khác lấy từ trang Hugging Face của chính model đó |
| Cột "Lặp/thừa" > 0 | `generation_kwargs.repetition_penalty` | `1.05`, rồi `1.1` |
| Nhóm có chữ nhỏ (bảng, hóa đơn) và model đang có `max_image_side` hoặc `max_pixels` | bỏ giới hạn đó, hoặc tăng lên `2048` | |

- Quy trình cho mỗi biến thể:
  1. Chạy biến thể với `--per-category 20 --categories <nhóm>`.
  2. Chạy `ocrbench decide --stage screening --per-category 20 --categories <nhóm> --models <gốc>,<biến thể>`.
  3. **Chỉ khi** biến thể là model đứng đầu bảng **và** model gốc bị ghi `loại` vì "kém chắc chắn hơn": chạy biến thể
     trên toàn bộ nhóm, rồi chạy lại `decide --stage final`.
  4. Mọi trường hợp khác: giữ model gốc và ghi kết quả vào nhật ký.

### 4.6. Giai đoạn 5: Báo cáo cuối

Viết `FINAL_REPORT.md` theo mẫu ở mục 10, rồi **dừng**. Không làm gì thêm cho tới khi con người trả lời.

## 5. Ước lượng ngân sách

Lấy `s/mẫu` của model từ `report.md` (cột "s/mẫu") của giai đoạn gần nhất, và số mẫu dev của nhóm từ `validate`.

```
giờ GPU của một model trên một nhóm = (s/mẫu × số mẫu cần chạy + 120) / 3600 × (số GPU model dùng)
```

Cộng mọi model và mọi nhóm sẽ chạy, **nhân 1,3**, rồi ghi phép tính vào nhật ký. Nếu kết quả > ngân sách còn lại:
**DỪNG và hỏi**. **CẤM** tự giảm số mẫu hay số model để vừa ngân sách.

## 6. Kiểm tra GPU và xử lý model quá lớn

### 6.1. Trước khi chạy một model lần đầu

```bash
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv
ocrbench vram --config C --models <model>            # thêm --long nếu là mục __long
```

- `memory.used` của GPU sắp dùng > 500 MiB: tìm tiến trình chiếm GPU (`nvidia-smi --query-compute-apps=pid,process_name --format=csv`).
  Nếu đó là `ocrbench.worker` do **chính bạn** khởi động và lệnh `run` tương ứng đã kết thúc: `kill <pid>`.
  Mọi trường hợp khác: **DỪNG và hỏi** (L11).
- Làm theo dòng `→` mà `vram` in ra: "Không cần thay đổi" thì chạy luôn; "Thêm mục sau vào config" thì chép nguyên văn mục đó.

### 6.2. Khi chạy mà hết bộ nhớ

Dấu hiệu: log (`OUT/dev/<model>/run.log`) có `CUDA out of memory` hoặc `OutOfMemoryError`.

1. Mở lại bảng `vram` của model đó, tìm bậc **kế tiếp** (theo thứ tự chữ cái) có cột "Vừa?" = `CÓ`.
2. Tạo mục mới theo đúng bảng dưới đây, giữ nguyên mọi tham số khác của mục đang dùng:

| Bậc | Thay đổi trong mục model | Hậu tố tên |
|---|---|---|
| A | (không đổi) | |
| B | `gpus: 2` | `__2gpu` |
| C | `gpus: 2`, `params.max_image_side: 1600` | `__2gpu_side1600` |
| D | `params.quantization: 8bit` | `__8bit` |
| E | `params.quantization: 8bit`, `gpus: 2` | `__8bit_2gpu` |
| F | `params.quantization: 4bit` | `__4bit` |
| G | `params.quantization: 4bit`, `gpus: 2` | `__4bit_2gpu` |
| H | `params.quantization: 4bit`, `params.model_kwargs: {max_memory: {0: "<số GB của dòng H>GiB", cpu: "24GiB"}}` | `__4bit_offload` |

3. Đặt `enabled: false` cho mục cũ. Chạy mục mới theo giai đoạn đang làm, bắt đầu bằng `--per-category 1`.
4. Mỗi bậc chỉ thử **1 lần**. Không bỏ qua bậc. Không quay lại bậc cao hơn.
5. Riêng mục `__long`: **trước** khi xuống bậc D, được thử thêm `params.multi_page: per_page` (hậu tố `__perpage`).
   Ghi vào báo cáo: "đọc từng trang, không đo được khả năng giữ ngữ cảnh qua trang".
6. Đến bậc H mà vẫn hết bộ nhớ, **hoặc** bậc H chạy 1 mẫu mất > 10 phút: đặt `enabled: false`, ghi nhật ký
   (số tham số, VRAM trống, các bậc đã thử, lỗi từng bậc), rồi bỏ qua model đó.

### 6.3. Ghi chép bắt buộc khi model đã bị giảm

Mọi model chạy ở bậc C–H **PHẢI** được ghi trong `FINAL_REPORT.md`, mục "Chi phí", theo mẫu:
`<model>: chạy ở bậc <X> (<mô tả bậc>); chưa đo được mức mất so với bản gốc` (hoặc, nếu bản gốc cũng có kết quả,
chép số liệu của cả hai từ `decision_*.md`).

## 7. Bảng sự cố (gặp lỗi ở giai đoạn 1)

Tra theo thứ tự từ trên xuống. Mỗi dòng là **một** lần sửa.

| Dấu hiệu (trong `run.log` hoặc `predictions.jsonl`) | Cách sửa |
|---|---|
| `CUDA out of memory` / `OutOfMemoryError` | Mục 6.2 |
| `ModuleNotFoundError` / `ImportError` | Cài gói thiếu theo README. Xung đột phiên bản thì dùng venv riêng (L13) |
| `401` / `403` / `gated` khi tải model | **DỪNG và hỏi** (cần token Hugging Face hoặc quyền truy cập) |
| `trust_remote_code` trong thông báo lỗi | Thêm `params.trust_remote_code: true` |
| Lỗi có chữ `chat_template` / `processor` / `image token` | Đổi `params.input_mode` từ `processor` sang `template` (hoặc ngược lại) |
| Kết quả rỗng, toàn `!!!!` hoặc ký tự rác | Mục mới với `params.quantization: 8bit` (fp16 trên T4 có thể tràn số). **CẤM** dùng float32 |
| Kết quả có `<think>` | Kiểm tra `params.strip_think` đang là `true` |
| Kết quả là JSON | Thêm `params.json_field: <tên trường chứa văn bản>` (xem output) |
| Model dừng sau 10 mẫu lỗi liên tiếp | Sai cấu hình từ gốc: đọc 30 dòng cuối của `run.log` rồi tra lại bảng này |
| Tool báo `TỪ CHỐI CHẠY` | Bạn đã đổi cấu hình dưới tên cũ (L5). Đưa cấu hình về như cũ, hoặc tạo mục với tên mới |
| Lỗi không có trong bảng | **DỪNG và hỏi**, kèm 30 dòng cuối của `run.log` |

## 8. Nhật ký (`EXPERIMENTS.md`)

Thêm **một mục cho mỗi lệnh `ocrbench run`**, theo đúng mẫu (không bỏ dòng nào, dòng không áp dụng thì ghi `—`):

```markdown
## <YYYY-MM-DD HH:MM> — Giai đoạn <số> — <tên model, phân cách bằng dấu phẩy>
- Lệnh: `<lệnh chính xác đã chạy>`
- Kết thúc: <dòng ✔/✘ của từng model, chép nguyên văn>
- Thời gian chạy: <phút> phút × <số GPU> GPU = <giờ GPU> giờ GPU · Đã dùng tổng: <giờ> / <ngân sách> (<%>)
- VRAM đỉnh: <chép từ meta.json: peak_vram_mib> MiB
- Số liệu: <chép nguyên văn dòng của model trong bảng "Tổng quan" của report.md>
- Quyết định: <chép nguyên văn từ decision_*.md, hoặc "chưa áp dụng">
- Việc tiếp theo: <bước tiếp theo theo mục 4>
- Nghi vấn dữ liệu: <id mẫu + mô tả, hoặc —>
```

## 9. Khi nào DỪNG và hỏi

**DỪNG** (không chạy thêm lệnh nào) và gửi tin nhắn theo mẫu:
`DỪNG: <lý do>. Đã làm đến: <giai đoạn/bước>. Cần anh quyết định: <câu hỏi cụ thể, có các lựa chọn>.`

Các trường hợp phải dừng:
1. Bất kỳ việc gì liên quan tới holdout.
2. Tool báo lỗi không do config (L3), hoặc lỗi không có trong bảng ở mục 7.
3. Giờ GPU đã dùng đạt 80% ngân sách (L15), hoặc ước lượng ở mục 5 vượt ngân sách còn lại.
4. GPU bị chiếm bởi tiến trình không phải của bạn.
5. Cần token, quyền tải model, hoặc cần cài thứ bị cấm ở L13.
6. `ocrbench decide` cho kết quả `(không có)` ở bất kỳ nhóm nào.
7. Báo cáo có dòng `⚠ CẢNH BÁO: dữ liệu đã thay đổi`.
8. Phiên Kaggle còn dưới 1 giờ (L18).
9. Muốn làm bất kỳ việc gì file này không nói tới.

## 10. Mẫu `FINAL_REPORT.md`

Mọi con số **CHỈ ĐƯỢC** chép từ các file nêu ở L8. Đánh dấu rõ từng câu thuộc loại nào: **[ĐO]** = chép từ file kết
quả, **[SUY RA]** = rút ra từ các số đã đo, **[CHƯA KIỂM CHỨNG]**.

```markdown
# Báo cáo chọn model OCR — <ngày>

## 1. Kết quả theo nhóm
<chép nguyên văn mọi bảng từ reports/phase3_final/decision_final.md>

## 2. Đề xuất
| Nhóm | Model thắng | Chỉ số chính (±95%) | Hòa với | Ghi chú |

## 3. Tài liệu dài
<chép nguyên văn mục "Tài liệu dài" của reports/phase2b_long/report.md>

## 4. Model bị loại
| Model | Giai đoạn | Lý do (chép từ decision_*.md hoặc nhật ký) |

## 5. Chi phí
- <từng model thắng>: s/mẫu, VRAM đỉnh, số GPU
- Model đã phải giảm để chạy được (mục 6.3)

## 6. Hạn chế
- Nhóm có dòng ⚠ khi validate (ít mẫu)
- Model không chạy được (và lý do)
- Nghi vấn dữ liệu (danh sách id)
- Bộ test là dữ liệu thay thế (xem testset/DATASET_CARD.md), chưa phải dữ liệu thật

## 7. Việc cần con người quyết định
- Ngưỡng pass/fail cho từng nhóm
- <các câu hỏi khác phát sinh>
```
