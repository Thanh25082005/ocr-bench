# Bộ đo "ký hiệu nhỏ trước số" (tự sinh)

20 bảng TỰ TẠO kiểu PubTabNet, độ phân giải thấp (mỗi dòng ~10–12 px như pubtabnet__552595 → ký hiệu mũ chỉ ~3–4 px),
121 ký hiệu chú thích trước số (†, ‡, §, ¶, *, #, chữ mũ a–d) + nhiều ô không ký hiệu (để đo thêm sai).
Dùng chung cho các thí nghiệm ký hiệu nhỏ (hướng 2 siêu phân giải, hướng 3 TTA) để so công bằng.

- `make.py`: sinh lại (`python experiments/marker_bench/make.py`, cần Playwright + Chromium).
- `data/`: ảnh `mb_XX.png` + `manifest.jsonl` (đáp án HTML, số ký hiệu từng bảng).

Cô lập: không đụng bộ test, không sửa ocrbench. Xoá: `rm -rf experiments/marker_bench`.
