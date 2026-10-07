# Thí nghiệm giấy tờ cũ (ảnh thật của khách) — giai đoạn 1

Mục tiêu: trước khi dựng nhánh "giấy tờ cũ", đo xem hệ thống nào **đọc bằng mắt** (bám ảnh) và hệ thống nào
**đoán chữ** trên loại tài liệu chính của khách: giấy tờ pháp lý 1970s, đánh máy / viết tay, ảnh chụp WhatsApp,
giấy ố, nếp gấp, có Hebrew, con dấu, vân tay.

| Hệ thống | Loại | Vì sao thử |
|---|---|---|
| dots_mocr (bố cục / chỉ chữ) | VLM đọc cả trang | model đang dùng cho web |
| churro_3b | VLM huấn luyện trên 99k trang tài liệu LỊCH SỬ (có Ả Rập, Hebrew) | arXiv 2509.19768 |
| qari_0_4 | VLM chuyên tiếng Ả Rập | arXiv 2506.02295 |
| sherif_handwriting | VLM chữ viết tay Ả Rập | đứng đầu benchmark chữ tay |
| kraken_openiti | OCR từng dòng, chữ in Ả Rập (OpenITI) | CER 3,1% trên trang Ả Rập in thật, VLM 10–21% (arXiv 2605.27750) |
| kraken_ppocrv6 | OCR từng dòng, 44 ngôn ngữ (có Hebrew) | arXiv 2609.20064 |

Mỗi ảnh chạy 2 biến thể: `__goc` (nguyên) và `__cat` (cắt tờ giấy khỏi nền + làm nét nhẹ). Không nhị phân hoá.

## Dữ liệu cá nhân — luật cứng
- Ảnh gốc: `/kaggle/working/real_docs/`; mọi kết quả: `/kaggle/working/real_docs_out/` — cả hai NGOÀI repo,
  nằm trong `.gitignore`. KHÔNG ghi vào `runs/`, `reports/` (bị `ocrbench status --push` đẩy lên GitHub).
- `so_sanh.html` có chữ OCR (tên, số căn cước) → chỉ xem trên server. Nhật ký chỉ chép `tom_tat.md` (số liệu).

## Chạy
```bash
O=/kaggle/working/real_docs_out
python experiments/real_docs_probe/prep.py /kaggle/working/real_docs $O
/kaggle/working/venvs/dots/bin/python experiments/real_docs_probe/run_vlm.py $O dots_mocr --gpus 0
/kaggle/working/venvs/dots/bin/python experiments/real_docs_probe/run_vlm.py $O dots_mocr --mode "Chỉ chữ" --gpus 0
python experiments/real_docs_probe/run_vlm.py $O churro_3b --gpus 0
python experiments/real_docs_probe/run_vlm.py $O qari_0_4 --gpus 0
python experiments/real_docs_probe/run_vlm.py $O sherif_handwriting --gpus 0
bash experiments/real_docs_probe/run_kraken.sh $O
python experiments/real_docs_probe/compare.py $O
```
Xoá thí nghiệm: `rm -rf experiments/real_docs_probe` (không đụng ocrbench).
