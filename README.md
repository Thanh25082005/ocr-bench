# ocrbench

Công cụ chạy **nhiều model OCR trên cùng một bộ test** (tiếng Anh, tiếng Ả Rập, bảng, chữ viết tay) rồi so sánh
bằng cùng một cách chấm. Thiết kế để chạy trên Kaggle (GPU T4 ×2).

- Mỗi model là một mục trong file YAML. Muốn thêm hoặc bớt model thì sửa config, không cần sửa code.
- Mỗi model chạy trong **tiến trình riêng**: model chạy xong thì VRAM được trả hết, model lỗi không kéo model khác chết theo.
- Với `--gpus 0,1`, **hai model chạy song song**, mỗi model một card.
- **Chạy tiếp được** khi bị ngắt giữa chừng: trang nào đã có kết quả thì bỏ qua.
- Kết quả thô lưu riêng với phần chấm điểm, nên đổi cách chấm thì chỉ cần `ocrbench score`, không phải chạy lại model.
- Holdout được khóa: phải thêm `--confirm-holdout` thì mới chạy hoặc chấm trên holdout.

## 1. Chuẩn bị dữ liệu

### Chưa có dữ liệu thật: tự dựng bộ test

```bash
pip install -e ".[testset]" && playwright install chromium
ocrbench build-testset testset --public-n 120 --synth-n 120
```

Lệnh này tạo `testset/manifest.jsonl` gồm hai phần:

- **`pub_*`**: mẫu lấy từ các bộ công khai trên Hugging Face, gồm Misraj-DocOCR (chữ in tiếng Ả Rập), bảng tiếng Ả Rập
  của KITAB-Bench, PubTabNet (bảng tiếng Anh), KHATT (chữ viết tay tiếng Ả Rập) và IAM (chữ viết tay tiếng Anh).
- **`syn_*`**: tài liệu tự sinh, gồm hợp đồng, công văn, hóa đơn và biểu mẫu điền tay, bằng tiếng Ả Rập, tiếng Anh và
  trộn hai thứ tiếng. Có thêm bản làm xấu như ảnh chụp điện thoại. Ảnh được dựng bằng Chromium nên chữ Ả Rập nối và
  chạy phải-sang-trái đúng, còn đáp án sinh cùng lúc với ảnh nên chính xác tuyệt đối.

Giấy phép, nguồn gốc và các hạn chế được ghi trong `testset/DATASET_CARD.md`. **Đọc file này trước khi dùng kết quả.**
Chạy lại lệnh với số lớn hơn sẽ dùng lại các mẫu đã có. Cùng `--seed` thì luôn ra cùng một bộ test.

### Đã có dữ liệu thật

Cách đơn giản nhất là xếp thư mục như sau:

```
data/
  printed_en/      invoice01__p1.png  invoice01__p1.txt
  printed_ar/      hoso03__p1.png     hoso03__p1.txt
  tables_ar/       bang07__p1.png     bang07__p1.html     <- bảng: đáp án là HTML <table>
  handwriting_ar/  nguoi05__l1.png    nguoi05__l1.txt
```

- Mỗi thư mục con là một **nhóm** (category). Kết quả được báo cáo riêng cho từng nhóm.
- Phần trước `__` trong tên file là **mã tài liệu** (`doc_id`), ví dụ mã hồ sơ hoặc mã người viết. Các trang có cùng
  mã luôn nằm cùng phía dev hoặc holdout, nhờ vậy không bị rò rỉ dữ liệu.
- Đáp án là file `.txt` (văn bản) hoặc `.html` (bảng), đặt cùng tên với ảnh.

Tạo manifest:

```bash
ocrbench make-manifest data/ -o data/manifest.jsonl --holdout-ratio 0.5
```

Nếu dữ liệu có sẵn ở dạng khác, anh có thể tự viết `manifest.jsonl`, mỗi dòng một mẫu:

```json
{"id": "hoso12_p3", "image": "images/hoso12_p3.png", "category": "handwriting_ar", "gt_file": "gt/hoso12_p3.txt", "doc_id": "hoso12", "split": "dev"}
```

Các trường `gt` (đáp án ghi thẳng), `gt_type` (`text` / `table_html`) và `lang` là tùy chọn. Trường nào khác
được giữ lại trong `meta`.

## 2. Chạy

```bash
pip install -e ".[teds,hf]"             # thêm: tesseract, easyocr, paddle, surya, api
ocrbench validate --config config.yaml   # đếm mẫu theo nhóm, báo ảnh thiếu, nhóm quá ít mẫu
ocrbench run --config config.yaml --limit 5 --gpus 0,1   # thử nhanh 5 mẫu để bắt lỗi cấu hình
ocrbench run --config config.yaml --gpus 0,1             # chạy hết (tự chạy tiếp phần còn thiếu)
ocrbench run --config config.yaml --models baseer,qari_0_4   # chỉ vài model
ocrbench run --config config.yaml --per-category 20       # tập con 20 mẫu/nhóm, cố định cho mọi model
ocrbench run --config config.yaml --retry-errors          # chạy lại các trang bị lỗi
ocrbench score --config config.yaml                       # chỉ chấm lại và tạo báo cáo
ocrbench vram --config config.yaml [--long]               # VRAM trống, VRAM từng model cần, cấu hình nên dùng
ocrbench decide --config config.yaml --stage screening --per-category 20   # loại/chọn model theo luật cố định
ocrbench decide --config config.yaml --stage final                         # chọn model thắng cho từng nhóm
```

Ý nghĩa của từng chỉ số trong báo cáo, kèm ví dụ: **[docs/METRICS.md](docs/METRICS.md)**.

Hai cơ chế tool tự bảo vệ:
- **Đổi tham số của một model đã có kết quả mà vẫn giữ tên cũ thì tool từ chối chạy** (mã thoát 3), để không trộn kết
  quả của hai cấu hình khác nhau. Cấu hình mới phải mang tên mới.
- Mỗi lượt chạy ghi lại dấu vân tay của bộ dữ liệu. Nếu dữ liệu bị sửa sau khi model đã chạy, báo cáo in cảnh báo.

Kết quả nằm trong `output_dir/<split>/`:

```
runs/dev/
  baseer/predictions.jsonl   kết quả thô từng trang (văn bản, thời gian, lỗi)
  baseer/scores.jsonl        điểm từng trang
  baseer/meta.json           cấu hình, GPU, thời gian nạp model, VRAM đỉnh
  baseer/run.log             log đầy đủ
  _report/report.md          bảng so sánh các model
  _report/summary.csv        số liệu theo model × nhóm
```

## 3. Chỉ số

| Chỉ số | Ý nghĩa |
|---|---|
| CER / WER (norm) | Tỉ lệ lỗi ký tự / từ sau chuẩn hóa, trung bình theo trang, kèm khoảng tin cậy 95% |
| CER raw | Không chuẩn hóa. Chênh nhiều so với norm thường là do định dạng, không phải đọc sai |
| CER micro | Tổng lỗi / tổng ký tự, nên trang dài nặng ký hơn |
| TEDS | Độ giống của bảng (1.0 = khớp hoàn toàn), tính cả cấu trúc lẫn nội dung ô. TEDS-struct chỉ xét cấu trúc |
| Lỗi/rỗng | Model báo lỗi hoặc không trả về gì |
| Lặp/thừa | Kẹt vòng lặp, hoặc dài hơn đáp án quá 1,5 lần (dấu hiệu bịa chữ), kèm cận trên 95% của tỉ lệ thật |
| s/mẫu, VRAM đỉnh | Tốc độ và bộ nhớ, để biết model có kham nổi khối lượng thật không |

Quy tắc chuẩn hóa (bỏ tashkeel, gộp hamza, đổi chữ số...) nằm ở mục `normalization` trong config và **áp dụng giống
nhau cho cả đáp án lẫn kết quả model**.

## 4. Thêm model

**Model VLM trên Hugging Face** (phần lớn model OCR mới): dùng adapter `hf_vlm`, chỉ cần sửa config.

```yaml
- name: my_vlm
  adapter: hf_vlm
  params:
    model_id: org/model
    adapter_id: org/my-lora      # tùy chọn: LoRA/QLoRA của anh sau khi fine-tune
    quantization: 4bit           # tùy chọn: nhét model to vào 1 GPU
    prompt:                      # prompt riêng theo nhóm hoặc theo loại đáp án
      default: "Extract all text..."
      table_html: "Convert every table to HTML..."
    max_new_tokens: 4096
```

**Model có script riêng:** dùng adapter `command`, mỗi ảnh chạy một lệnh và lấy stdout làm kết quả.

**Model cần code riêng:** viết một class kế thừa `ocrbench.adapters.base.Adapter`:

```python
from ocrbench.adapters.base import Adapter, Prediction

class MyOCR(Adapter):
    def load(self):
        self.model = ...                       # nạp model
    def predict(self, image, item):            # image: PIL.Image, item.category / item.gt_type
        return Prediction(text=self.model.read(image))
```

Rồi khai báo trong config: `adapter: "my_module:MyOCR"`.

**Model cần thư viện xung đột** (ví dụ Surya ghim phiên bản `transformers` riêng): cài vào venv riêng và trỏ `python:`
tới venv đó.

```bash
uv venv /kaggle/working/venvs/surya --system-site-packages
uv pip install -p /kaggle/working/venvs/surya/bin/python surya-ocr -e /kaggle/working/ocr-bench
```

## 5. Lưu ý khi chạy trên Kaggle

- **T4 không có bf16** nên model sẽ chạy ở fp16. Một số model huấn luyện bằng bf16 có thể ra chữ rác hoặc rỗng ở fp16.
  Khi gặp, thử `dtype: float32` hoặc `quantization: 4bit`.
- **vLLM cần GPU Compute Capability ≥ 8.0, mà T4 chỉ có 7.5.** Vì vậy tool chạy model qua `transformers` là chính.
- Model ≥ 8 tỷ tham số không vừa một card T4 ở fp16: dùng `gpus: 2` (chia trên 2 card) hoặc `quantization: 4bit`.
- **Phiên Kaggle tối đa 12 giờ.** Với lượt chạy dài, dùng *Save Version → Save & Run All* để chạy nền. Lần sau, gắn output
  của lần trước làm input rồi copy thư mục `runs` về `/kaggle/working`; tool sẽ tự chạy tiếp phần còn thiếu.
- Dữ liệu tải lên Kaggle là tải lên dịch vụ của bên thứ ba. Cần kiểm tra lại cam kết bảo mật với bên cung cấp dữ liệu.

## 5b. Chạy trên Kaggle qua SSH

Kaggle không có SSH chính thức. Anh tự mở kết nối SSH vào notebook (qua tunnel); cách này có thể bị Kaggle chặn,
và rủi ro với tài khoản do anh tự cân nhắc. Khi đã SSH vào được:

```bash
# Trên máy Kaggle (repo private nên cần token chỉ-đọc):
curl -sL -H "Authorization: token <TOKEN>" \
  https://raw.githubusercontent.com/khanhkhmt/ocr-bench/main/scripts/kaggle/setup.sh -o setup.sh
GITHUB_TOKEN=<TOKEN> bash setup.sh
```

`setup.sh` tự tải bộ test từ Google Drive (`scripts/get_testset.py`, khoảng 20–25 phút; bị ngắt thì chạy lại để tải
tiếp) và kiểm tra dấu vân tay. Muốn dựng lại trên Kaggle thì dùng `TESTSET=build`; muốn tự đưa lên từ máy mình thì
dùng `TESTSET=none` rồi chạy `bash scripts/kaggle/push_testset.sh <ssh-host>`.

```bash
# Trên Kaggle: mọi lệnh chạy lâu đặt trong tmux, để mất SSH vẫn chạy tiếp
tmux new -s bench
ocrbench run --config /kaggle/working/config.yaml --per-category 1 --gpus 0,1
```

**Phiên Kaggle tối đa 12 giờ; hết phiên là `/kaggle/working` bị xóa.** Vì vậy:

| Khi nào | Lệnh (chạy trên máy của anh) |
|---|---|
| Trong lúc chạy, định kỳ (vd. 30 phút một lần) | `bash scripts/kaggle/pull_results.sh <ssh-host>` → lưu vào `kaggle_results/` |
| Mở phiên Kaggle mới, sau khi chạy lại `setup.sh` | `bash scripts/kaggle/push_results.sh <ssh-host>`, rồi chạy lại đúng lệnh `ocrbench run` cũ: tool bỏ qua mẫu đã có |

Hai máy có cùng **dấu vân tay dữ liệu** (dòng thứ hai của `ocrbench validate`) nghĩa là cùng một bộ test, và kết
quả cũ dùng tiếp được.

## 6. Giao cho agent chạy

`AGENT_PROMPT.md` là prompt hướng dẫn agent xoay model theo vòng loại: chạy thử → sàng lọc trên tập con →
chung kết trên toàn bộ dev → tinh chỉnh có giới hạn → báo cáo. Điền các chỗ `<<...>>` rồi đưa cho agent.

## 7. Test

```bash
pip install -e ".[dev]" && pytest
```

Bộ test chạy toàn bộ pipeline bằng model giả lập (`adapter: dummy`), không cần GPU.
