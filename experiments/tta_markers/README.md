# Thí nghiệm: TTA + ghép ký hiệu nhỏ (†, ‡, §, *, chữ mũ) cho dots.mocr

**Cô lập hoàn toàn**: chỉ *gọi* ocrbench (DotsAdapter, preprocess, metrics), KHÔNG sửa `ocrbench/`, web demo,
config hay benchmark. Bỏ thí nghiệm: `rm -rf experiments/tta_markers` (kết quả chạy nằm ở `/kaggle/working/exp_tta`).

## Vấn đề
Ảnh bảng độ phân giải thấp (vd. pubtabnet__552595, 503 px): dots bỏ mất ký hiệu chú thích nhỏ trước số
(†105 → 105). Phóng cả ảnh ×5 thì đọc ra ký hiệu nhưng phá cấu trúc bảng (mất cột), chậm ×3,5.

## Cách làm (test-time augmentation + bỏ phiếu, chỉ cho ký hiệu)
1. Lượt **gốc** = đúng web demo → giữ cấu trúc bảng và mọi con số.
2. Các lượt **phụ** (`--passes`, mặc định `x4_ocr,x5_ocr,x5_layout`): ảnh phóng ×k (cạnh dài tối đa 2.600 px),
   chế độ chữ (`prompt_ocr`) hoặc bố cục.
3. `fuse.py`: ô có giá trị *v* ở lượt gốc; nếu cùng dòng (khớp nhãn ô đầu) ở lượt phụ có token = *ký hiệu + v*
   → thêm ký hiệu. Phần thêm là chữ số (146 vs 46) → không sửa. Các lượt phụ mâu thuẫn → không sửa.
   `min_votes` 1 hoặc 2.

## Chạy (Kaggle, venv dots, 1 GPU)
```bash
cd /kaggle/working/ocr-bench
CUDA_VISIBLE_DEVICES=0 /kaggle/working/venvs/dots/bin/python experiments/tta_markers/run.py   # → /kaggle/working/exp_tta/report.md
```
Mặc định: 8 ảnh bảng (552595 — holdout ĐÃ LOẠI; 7 ảnh dev có < ≥ ± chữ mũ, danh sách lồng) + 2 ảnh đối chứng
(`inference/samples` 01, 08 — không được tệ đi). Script từ chối ảnh holdout chưa nằm trong `goal/holdout_excluded.yaml`.

## Đọc kết quả (report.md)
- **Ghép v1/v2 (sai)**: số ký hiệu trước số đúng sau khi ghép (số ký hiệu ghép thêm SAI).
- **CER gốc → v1/v2**: phải không tăng ở mọi ảnh, nhất là ảnh đối chứng.
- **Giây phụ**: chi phí thêm.
Đạt khi: thêm đúng > 0, thêm sai = 0, CER không tăng → mới cân nhắc đưa thành tuỳ chọn "Soát kỹ ký hiệu nhỏ" trên web.

## Kiểm thử (không cần GPU)
`python -m pytest -q experiments/tta_markers/test_fuse.py`
