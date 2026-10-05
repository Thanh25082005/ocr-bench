# Web demo dots.mocr

Kéo thả PDF / ảnh → model **dots.mocr** đọc bố cục từng trang (tiêu đề, đoạn, bảng HTML, thứ tự đọc)
→ xem **ảnh bố cục** (khung + số thứ tự từng khối), xem trước chữ, **tải DOCX**.

| File | Vai trò |
|---|---|
| `start.sh` | Cài thư viện → tự kiểm tra model trên 1 trang mẫu → mở web |
| `check.py` | Tự kiểm tra (hoá đơn mẫu có bảng); in ✔/✘ |
| `config.yaml` | Mục `dots_mocr` (chép nguyên từ `configs/kaggle_example.yaml`, adapter `dots` = giống source gốc) |

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
