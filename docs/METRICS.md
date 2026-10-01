# Giải thích các chỉ số đánh giá

Tài liệu này giải thích mọi con số xuất hiện trong `report.md` và `summary.csv`: nó đo cái gì, tính thế nào, đọc
ra sao. Mọi ví dụ bên dưới là kết quả chạy thật bằng code chấm điểm của tool, không phải số minh họa.

---

## Phần A — Độ chính xác của chữ

### A1. CER: tỉ lệ lỗi ký tự (Character Error Rate)

**Câu hỏi nó trả lời:** trong 100 ký tự của đáp án, model đọc sai bao nhiêu ký tự?

**Cách tính:** đếm số thao tác *ít nhất* để biến kết quả model thành đáp án. Có 3 loại thao tác: **thay** một ký tự,
**xóa** một ký tự thừa, **thêm** một ký tự bị thiếu. Lấy số thao tác chia cho số ký tự của đáp án.

```
CER = (số ký tự bị thay + bị thiếu + bị thừa) / số ký tự của đáp án
```

**Ví dụ:**

| Đáp án | Model đọc | Lỗi | CER |
|---|---|---|---|
| `Total due: 1250 USD` (19 ký tự) | `Total due: 1205 USD` | đảo `50` → `05` = 2 ký tự bị thay | 2/19 = **10,5%** |
| `Total due: 1250 USD` | `Tota1 due 1250 USD` | `l` → `1`, thiếu `:` | 2/19 = **10,5%** |

**Đọc kết quả (gợi ý, ngưỡng thật do anh chốt):**

| CER | Với chữ in | Với chữ viết tay |
|---|---|---|
| < 1% | Rất tốt, gần như không phải sửa | Hiếm gặp |
| 1–3% | Tốt | Rất tốt |
| 3–8% | Dùng được nếu có người duyệt | Tốt |
| 8–15% | Kém | Trung bình (mức thường gặp của chữ tay tiếng Ả Rập) |
| > 15% | Không dùng được | Kém |

**Lưu ý:** CER có thể **lớn hơn 100%**. Ví dụ đáp án có 50 ký tự mà model bịa thêm 200 ký tự, thì CER = 400%.
Một trang như vậy kéo trung bình lên rất mạnh, và đó là chủ ý: bịa chữ phải bị phạt nặng.

### A2. CER (norm) và CER raw: có chuẩn hóa hay không

Trước khi so sánh, tool chạy **cùng một bộ quy tắc chuẩn hóa** cho cả đáp án lẫn kết quả model, để không phạt những
khác biệt không quan trọng:

- bỏ định dạng Markdown/HTML (`**`, `#`, `|`, thẻ `<td>`...): model hay trả về Markdown, đáp án thì là chữ thường;
- bỏ dấu tashkeel (`ـَ ـِ ـُ`) và ký tự kéo dài tatweel (`ـ`);
- quy chữ số Ả Rập-Ấn về chữ số thường (`١٢٥٠` → `1250`), kể cả dấu `٫` `٬`;
- gộp các khoảng trắng liên tiếp thành một.

Bộ quy tắc nằm ở mục `normalization` trong file config.

| | Đáp án | Model đọc |
|---|---|---|
| Nguyên bản | `المبلغ الإجمالي: ١٢٥٠ ريال` | `**المبلغُ الإجمالي:** 1250 ريال` |
| Sau chuẩn hóa | `المبلغ الإجمالي: 1250 ريال` | `المبلغ الإجمالي: 1250 ريال` |
| | **CER raw = 34,6%** | **CER norm = 0%** |

Model này thực ra đọc **đúng hoàn toàn**; nó chỉ khác ở định dạng (in đậm, thêm tashkeel, dùng chữ số khác).

- **CER (norm)**: dùng để **so sánh và ra quyết định**.
- **CER raw**: chỉ để **chẩn đoán**. Nếu raw cao hơn norm nhiều thì model đọc đúng nhưng trình bày khác; việc này
  sửa được bằng prompt hoặc hậu xử lý, không phải lỗi đọc.

### A3. CER micro

**Cách tính:** cộng tất cả lỗi của mọi trang, chia cho tổng ký tự của mọi trang.

Khác với CER thường (là **trung bình các trang**, trang nào cũng nặng như nhau), CER micro để trang dài nặng ký hơn
trang ngắn. Hai số chênh nhau nhiều nghĩa là model đọc trang ngắn và trang dài rất khác nhau. Chỉ dùng để tham khảo.

### A4. WER: tỉ lệ lỗi từ (Word Error Rate)

Giống CER nhưng tính theo **từ**: một từ sai dù chỉ một ký tự thì cả từ đó tính là sai.

| Đáp án | Model đọc | CER | WER |
|---|---|---|---|
| `Total due: 1250 USD` (4 từ) | `Total due: 1205 USD` | 10,5% | 1/4 = **25%** |
| `Total due: 1250 USD` | `Tota1 due 1250 USD` | 10,5% | 2/4 = **50%** |

Hai trường hợp có cùng CER nhưng WER khác nhau: lỗi rải ra nhiều từ thì WER cao hơn. WER quan trọng khi sau OCR còn
**tìm kiếm theo từ khóa**, vì sai một chữ là không tìm ra từ đó. Với tiếng Ả Rập, WER thường gấp 3–4 lần CER.

### A5. Dấu "±": khoảng tin cậy 95%

Báo cáo ghi kiểu `4,2% ±1,1`. Nghĩa là: đo trên mẫu này được trung bình 4,2%, và nếu đo trên **toàn bộ** tài liệu
cùng loại thì CER thật có khả năng cao nằm trong khoảng **3,1% – 5,3%**.

**Cách dùng khi so sánh hai model:**

| Model A | Model B | Kết luận |
|---|---|---|
| 4,2% ±1,1 (3,1–5,3) | 9,0% ±1,5 (7,5–10,5) | Hai khoảng **không chồng** nhau → A **chắc chắn** tốt hơn B |
| 4,2% ±1,1 (3,1–5,3) | 5,0% ±1,4 (3,6–6,4) | Hai khoảng **chồng** nhau → **coi như hòa**; chênh 0,8% có thể chỉ là may rủi |

Càng nhiều mẫu thì "±" càng nhỏ. Vì vậy nhóm chỉ có vài chục mẫu thường cho kết quả "hòa".

---

## Phần B — Bảng

Với bảng, đọc đúng chữ là chưa đủ: **mỗi giá trị phải nằm đúng ô**. Ví dụ dưới đây dùng bảng 4 hàng × 3 cột:

```
| Item | Qty | Price |
| Pen  | 2   | 5.00  |
| Book | 1   | 12.50 |
| Bag  | 3   | 30.00 |
```

và ba kiểu lỗi model thường mắc:

| Model đọc | TEDS | TEDS cấu trúc | Ô đúng vị trí | Ô đúng sau căn hàng | Hàng sai số cột |
|---|---:|---:|---:|---:|---:|
| ① Đọc sai 1 số: `12.50` → `12.80` | 0,989 | 1,000 | 91,7% | 91,7% | 0 |
| ② Bỏ sót cả hàng `Pen` | 0,778 | 0,778 | **25,0%** | **75,0%** | 0 |
| ③ Bỏ sót ô `1` của hàng `Book`, nên `12.50` bị dồn sang cột Qty | 0,944 | 0,944 | 83,3% | 83,3% | **1** |

### B1. TEDS: độ giống nhau của bảng

**Cách tính:** biến cả hai bảng thành cây (bảng → hàng → ô), rồi đếm số bước ít nhất để biến cây này thành cây kia
(thêm/xóa/sửa ô). Kết quả quy về thang 0–1, với **1,0 = giống hệt**.

- **TEDS** xét cả cấu trúc lẫn chữ trong ô.
- **TEDS cấu trúc** chỉ xét khung bảng (bao nhiêu hàng, mỗi hàng bao nhiêu ô, ô gộp), bỏ qua chữ.

⚠ **Điểm yếu quan trọng:** ở ví dụ ①, model đọc **sai số tiền** mà TEDS vẫn đạt **0,989**, gần như hoàn hảo. Lý do là
TEDS chấm mỗi ô theo mức "gần giống", và `12.80` chỉ khác `12.50` một ký tự. **Vì vậy TEDS không đủ để đánh giá bảng
tài chính.**

### B2. Ô đúng vị trí (`cell_exact`)

**Câu hỏi:** bao nhiêu % ô có giá trị **đúng hoàn toàn** (không sai ký tự nào) **và** nằm đúng hàng, đúng cột như đáp án?

Đây là chỉ số khắt khe nhất, và gần với rủi ro nghiệp vụ nhất. Nó bắt được lỗi ① (91,7%), trong khi TEDS thì không.

### B3. Ô đúng sau căn hàng (`cell_aligned`)

**Câu hỏi:** nếu bỏ qua việc thiếu hoặc thừa hàng (tool tự ghép mỗi hàng của model với hàng giống nó nhất trong đáp án),
thì bao nhiêu % ô đúng?

**Đọc cùng với B2:**

- B2 ≈ B3 → các lỗi nằm rải rác trong từng ô (ví dụ ①).
- **B2 thấp hơn B3 rất nhiều → model bỏ sót hoặc thêm hàng, làm mọi hàng phía sau bị đẩy lệch** (ví dụ ②: 25% so
  với 75%). Đây chính là lỗi "lệch giá trị". Với bảng 100 hàng mà mất một hàng ở đầu, gần như toàn bộ giá trị phía
  sau nằm sai hàng.

### B4. Hàng sai số cột (`rows_wrong_ncols`)

Số hàng mà model trả về **nhiều hoặc ít ô hơn** đáp án. Lý do thường gặp nhất: model **bỏ qua ô trống**, nên các giá
trị phía sau bị dồn sang cột bên cạnh (ví dụ ③: số tiền `12.50` rơi vào cột Số lượng). Bảng sao kê có cột Nợ/Có để
trống xen kẽ rất dễ mắc lỗi này. **Nên bằng 0.**

### B5. Sai số hàng (`row_count_off`)

Số bảng mà model đọc ra **số hàng khác** đáp án, do thiếu hoặc thừa hàng. Thường đi kèm với B2 ≪ B3.

---

## Phần C — Tài liệu dài

### C1. Độ chính xác theo vị trí: Q1 → Q4

**Câu hỏi:** model đọc phần cuối tài liệu có tốt bằng phần đầu không?

**Cách tính:**
- Với văn bản: cắt đáp án thành các đoạn khoảng 8 từ, kiểm tra từng đoạn có xuất hiện (gần đúng) trong kết quả
  model không. Q1 là 1/4 đầu tài liệu, Q4 là 1/4 cuối.
- Với bảng: tỉ lệ ô đúng (sau căn hàng) của 1/4 số hàng đầu, 1/4 tiếp theo, v.v.

**Ví dụ:** một model đọc văn bản 400 từ nhưng dừng ở từ thứ 250:

| Q1 | Q2 | Q3 | Q4 |
|---:|---:|---:|---:|
| 100% | 100% | 46% | **0%** |

**Đọc kết quả:**

- Q1 ≈ Q4 → tốt, không mất ngữ cảnh.
- **Tụt dần về Q4 → model mất ngữ cảnh, đọc sót hoặc dừng sớm khi tài liệu dài.**
- Q4 = 0% kèm cột "Bị cắt" > 0 → model bị cắt vì hết `max_new_tokens`. Đây là lỗi cấu hình (tăng giới hạn token),
  chưa chắc là lỗi của model.

### C2. Bị cắt (`hit_max_tokens`)

Số mẫu mà model viết đến đúng giới hạn `max_new_tokens` thì bị dừng. Kết quả của những mẫu này chắc chắn thiếu phần
cuối. **Phải bằng 0 trước khi kết luận về model**; nếu không, tăng `max_new_tokens` rồi chạy lại.

### C3. Có và không lặp tiêu đề

Trong `syn_longtable`, một nửa số bảng nhiều trang **không in lại hàng tiêu đề** ở trang 2–3. Nếu một model kém hẳn ở
những bảng này (xem trường `header_repeated` trong `scores.jsonl`), nghĩa là nó không nhớ được cột ở trang 1 khi đọc
sang trang sau.

---

## Phần D — Độ tin cậy và an toàn

### D1. Lỗi/rỗng

Số mẫu mà model **báo lỗi** (hết bộ nhớ, lỗi code...) hoặc **trả về chuỗi rỗng**. Nên bằng 0. Nếu khác 0, phải xem log
trước khi tin bất cứ con số nào khác của model đó.

### D2. Lặp/thừa (dấu hiệu bịa chữ)

Số mẫu bị một trong hai cờ:

- **`repetition`**: kết quả có một đoạn bị lặp liền nhau từ 10 lần trở lên (model "kẹt vòng lặp", rất hay gặp ở VLM);
- **`too_long`**: kết quả dài hơn đáp án quá 1,5 lần, tức model viết ra nhiều chữ không có trong ảnh.

**Đây là lỗi nguy hiểm nhất.** Đọc sai một chữ thì người duyệt còn thấy; còn chữ bịa thì trông hợp lý, rất khó phát hiện.

### D3. Cận trên 95% (số trong ngoặc ở cột Lặp/thừa)

Báo cáo ghi kiểu `0 (≤6,0%)`. Nghĩa là: trong mẫu chưa thấy lỗi nào, **nhưng với số mẫu này chỉ dám khẳng định tỉ lệ
lỗi thật không quá 6%**. Muốn khẳng định chặt hơn thì cần nhiều mẫu hơn:

| Số lỗi / số mẫu | Có thể khẳng định tỉ lệ thật ≤ |
|---|---:|
| 0 / 20 | 16,1% |
| 0 / 60 | 6,0% |
| 0 / 300 | 1,3% |
| 1 / 60 | 8,9% |
| 3 / 60 | 13,7% |

Vì vậy muốn đặt ngưỡng kiểu "bịa chữ ≤ 1%", phải có **khoảng 300 mẫu sạch hoàn toàn**.

### D4. Các cờ khác (xem trong `scores.jsonl` và bảng "5 mẫu tệ nhất")

- **`too_short`**: kết quả ngắn hơn một nửa đáp án; model bỏ sót nhiều chữ.
- **`hit_max_tokens`**: bị cắt (xem C2).

---

## Phần E — Chi phí

### E1. Giây mỗi mẫu (s/mẫu)

Thời gian model đọc một mẫu, chưa tính thời gian nạp model. Quy đổi sang số trang mỗi giờ:

```
trang/giờ trên 1 GPU = 3600 / (s/mẫu)
```

Ví dụ: 6 s/mẫu → 600 trang/giờ/GPU → khoảng 1.200 trang/giờ trên 2× T4 (nếu chạy 2 bản song song). Với tài liệu
nhiều trang, s/mẫu là thời gian cho **cả tài liệu**.

### E2. VRAM đỉnh

Bộ nhớ GPU cao nhất mà model dùng trong lúc chạy, đo bằng `nvidia-smi`. Dưới khoảng 7 GB thì chạy được 2 bản trên 2 card
T4. Gần 16 GB thì dễ hết bộ nhớ khi gặp trang lớn hoặc tài liệu dài.

---

## Phần F — Dùng chỉ số nào để quyết định

| Loại tài liệu | Chỉ số chính | Xem thêm |
|---|---|---|
| Chữ in, hợp đồng, công văn | CER (norm) ± | WER, CER raw (chẩn đoán) |
| Chữ viết tay | CER (norm) ± | WER |
| Bảng, hóa đơn | **Ô đúng vị trí** | Ô đúng sau căn hàng, Hàng sai số cột, TEDS |
| Tài liệu dài | Q4 so với Q1; ô đúng vị trí theo độ dài | Bị cắt |
| Mọi loại (điều kiện bắt buộc) | Lặp/thừa, Lỗi/rỗng | Cận trên 95% |
| Mọi loại (chi phí) | s/mẫu, VRAM đỉnh | |

**Những gì KHÔNG dùng để quyết định:**
- CER raw: phạt cả khác biệt định dạng vô hại.
- TEDS một mình với bảng tài chính: bỏ qua sai số tiền (ví dụ ①).
- Trung bình không kèm "±": chênh 1–2% trên vài chục mẫu thường chỉ là may rủi.
- Độ tin cậy (confidence) do model tự báo: mỗi model tính một kiểu, không so sánh được.

**So sánh nhóm để phát hiện vấn đề:**
- `syn_degraded` so với nhóm sạch tương ứng: model có chịu được ảnh chụp kém không.
- `pub_*` so với `syn_*` cùng loại: model tốt bất thường ở `pub_*` có thể đã học thuộc dữ liệu công khai đó.
- `syn_form_*` là **chữ viết tay giả bằng font**, dễ hơn chữ tay thật. Đánh giá chữ viết tay phải dựa vào `pub_handwriting_*`.
