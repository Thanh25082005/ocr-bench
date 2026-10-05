# Web demo dots.mocr

Kéo thả PDF / ảnh → model **dots.mocr** đọc bố cục từng trang (tiêu đề, đoạn, bảng HTML, thứ tự đọc)
→ xem **ảnh bố cục** (khung + số thứ tự từng khối), xem trước chữ, **tải DOCX**.

| File | Vai trò |
|---|---|
| `start.sh` | Cài thư viện → tự kiểm tra model trên 1 trang mẫu → mở web |
| `check.py` | Tự kiểm tra (hoá đơn mẫu có bảng); in ✔/✘ |
| `config.yaml` | Mục `dots_mocr` (chép nguyên từ `configs/kaggle_example.yaml`, adapter `dots` = giống source gốc) |

## Vì sao có venv riêng

Code của dots (trust_remote_code) viết cho **transformers 4.56.1**. Trên Kaggle (transformers 5.0.0) model nạp
được, nhận ảnh đúng nhưng phần ngôn ngữ trả **chuỗi rỗng** (chẩn đoán: `inference/diagnose_dots.py`).
`start.sh` tự tạo `/kaggle/working/venvs/dots` (dùng chung torch/CUDA hệ thống, chỉ thay transformers).

## Chạy trên Kaggle (2× T4)

```bash
cd /kaggle/working/ocr-bench && git pull          # (lệnh pull có token: xem docs/RUNBOOK.md)
tmux new -s demo
bash inference/start.sh                           # lần đầu tải model ~6 GB, nạp ~1–2 phút
```

Từ máy mình mở web qua SSH: `ssh -L 7860:localhost:7860 kaggle-ngrok` → http://localhost:7860

Tùy chọn: `GPUS=1` (chỉ dùng GPU 1, khi GPU 0 đang chạy benchmark) · `PORT=7861` · `SKIP_CHECK=1` ·
`SHARE=1` (link công khai `*.gradio.live` — **chỉ dùng với tài liệu mẫu**, không dùng với tài liệu thật).

## Trên web

- **Chế độ đọc**: *Bố cục đầy đủ* (mặc định: khối + chữ + bảng HTML) · *Chỉ chữ* (prompt_ocr) ·
  *Chỉ bố cục (không chữ)* (chỉ khung, nhanh).
- **Dùng lớp chữ có sẵn của PDF**: mặc định TẮT để mọi trang đều qua dots (xem được bố cục).
- **Trang**: vd. `1-3` để demo nhanh tài liệu dài.
- Kết quả: DOCX (tiêu đề → heading, bảng → bảng Word, tiếng Ả Rập phải-sang-trái), bảng trạng thái từng trang
  (⚠ CẦN SOÁT nếu JSON bố cục hỏng / bị lặp / bị cắt), ảnh bố cục từng trang.

Tốc độ ước tính trên T4: ~30–90 giây/trang tùy độ dày chữ (2 GPU = 2 trang song song).

## Lưu ý

- GPU dùng chung với benchmark: dừng agent / lệnh `ocrbench run` trước, hoặc chạy demo với `GPUS=1`.
- Giấy phép trọng số dots.mocr (mục 5.2): cấm xử lý dữ liệu cá nhân nhạy cảm khi chưa có đồng ý + ẩn danh hoá.
  Demo bằng tài liệu mẫu; dùng tài liệu thật phải hỏi pháp lý trước.
- Muốn ra file giống hệt tool gốc của họ (.json, .jpg bố cục, .md, _nohf.md):
  `ocrbench dots-parse file.pdf --config inference/config.yaml --model dots_mocr -o dots_out`

## Tài liệu mẫu để demo (`samples/`)

Tự tạo hoàn toàn (`python inference/make_samples.py`, seed riêng, không lấy từ bộ test). Đáp án đúng ở
`samples/dap_an/` (`.html` = bảng, `.txt` = văn bản) để so khi demo.

| File | Thử điều gì |
|---|---|
| `01_hoa_don_tieng_anh.png` | Bảng tiếng Anh → bảng Word |
| `02_hoa_don_tieng_a_rap.png` | Bảng tiếng Ả Rập, số Ả Rập-Ấn (٠١٢…), phải-sang-trái |
| `03_hoa_don_tron_anh_a_rap.png` | Trộn Anh + Ả Rập trong cùng trang |
| `04_hop_dong_tieng_a_rap.png` | Văn bản dài tiếng Ả Rập, tiêu đề + đoạn |
| `05_thu_tieng_anh.png` | Thư tiếng Anh |
| `06_bieu_mau_viet_tay_a_rap.png` | Biểu mẫu điền tay tiếng Ả Rập |
| `07_bieu_mau_viet_tay_anh.png` | Biểu mẫu điền tay tiếng Anh |
| `08_anh_chup_xau_hoa_don_a_rap.jpg` | Ảnh chụp nghiêng, tối góc, mờ (mức vừa) |
| `09_anh_chup_rat_xau_hop_dong_anh.jpg` | Ảnh chụp rất xấu (mức khó) |
| `10_pdf_scan_bang_dai_3_trang.pdf` | PDF scan 3 trang, bảng sao kê dài — giá trị có lệch hàng không |
| `11_pdf_scan_2_trang_hop_dong_va_hoa_don.pdf` | PDF nhiều trang, mỗi trang một loại |
| `12_pdf_co_lop_chu_thu_tieng_anh.pdf` | PDF có lớp chữ: bật "Dùng lớp chữ" → lấy thẳng, không qua model |

Gợi ý thứ tự demo: 01 (bảng đẹp) → 06 (chữ viết tay) → 08 (ảnh xấu) → 10 với Trang `1-3` (tài liệu dài).
