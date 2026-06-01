# DPTTT---GK

Đồ án môn **Độ phức tạp thuật toán**: xây dựng và mô phỏng cơ chế
**Multi-Head Self-Attention** từ đầu bằng Python + NumPy, kèm benchmark
so sánh bản naive dùng vòng lặp và bản vectorized dùng phép toán ma trận.

## Điểm đã triển khai

- Scaled Dot-Product Attention:
  `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`
- Numerically stable softmax, xử lý được hàng bị mask toàn bộ mà không sinh `NaN`
- Causal mask cho bài toán autoregressive
- Padding mask cho batch có câu dài/ngắn khác nhau
- Multi-Head Attention hoàn chỉnh với `W_Q`, `W_K`, `W_V`, `W_O`
- Sinusoidal positional encoding để bổ sung thông tin thứ tự token
- Tokenizer word-level/char-level, batch padding và padding mask
- Benchmark naive vs vectorized
- Unit tests cho attention core, multi-head, positional encoding và tokenizer mask

## Cấu trúc thư mục

```text
DPTTT---GK/
├── attention/
│   ├── scaled_dot_product.py   # Attention core, causal mask, padding mask
│   └── multi_head.py           # Multi-Head Attention coordinator
├── core/
│   ├── layers.py               # Linear layer + Xavier init
│   └── math_utils.py           # Softmax, Xavier, positional encoding
├── data/
│   └── tokenizer.py            # Tokenizer + embedding lookup + padding mask
├── experiments/
│   ├── benchmark.py            # Naive vs vectorized benchmark
│   └── plots/                  # Biểu đồ benchmark
├── tests/
│   └── test_attention.py       # Unit tests
├── main.py                     # Demo pipeline Attention + n-gram generation
└── requirements.txt
```

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Trên Windows terminal, nếu gặp lỗi in tiếng Việt, chạy thêm:

```powershell
$env:PYTHONIOENCODING="utf-8"
```

## Chạy demo

```bash
python main.py
```

Demo sẽ chạy pipeline:

```text
Text -> Tokenizer -> Embedding + Positional Encoding
     -> Multi-Head Self-Attention -> Output Logits -> Sampling token tiếp theo
```

Lưu ý: phần sinh văn bản dùng n-gram để câu đầu ra dễ đọc hơn vì project này
tập trung vào mô phỏng forward pass và phân tích độ phức tạp của attention,
không huấn luyện một Transformer language model hoàn chỉnh.

## Chạy test

```bash
python -m unittest discover -s tests
```

## Chạy benchmark

Kiểm tra nhanh pipeline benchmark:

```bash
python experiments/benchmark.py --quick
```

Chạy benchmark đầy đủ:

```bash
python experiments/benchmark.py
```

Benchmark đầy đủ có thể tốn nhiều RAM ở các sequence length lớn vì attention
cổ điển phải tạo ma trận attention kích thước `L x L` cho mỗi head.

## Phân tích độ phức tạp

Với `B` là batch size, `H` là số head, `L` là sequence length và `d_k` là
kích thước mỗi head:

- Tính score `QK^T`: `O(B * H * L^2 * d_k)`
- Softmax trên ma trận attention: `O(B * H * L^2)`
- Nhân attention weights với `V`: `O(B * H * L^2 * d_k)`
- Không gian cho score/weights: `O(B * H * L^2)`

Vì vậy khi `L` tăng gấp đôi, chi phí attention chuẩn tăng xấp xỉ bậc hai.
Đây là lý do các mô hình ngữ cảnh dài thường cần các kỹ thuật tối ưu như
FlashAttention hoặc sparse/sliding-window attention.
