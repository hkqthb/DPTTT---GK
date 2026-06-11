# 📊 Bản Tóm Tắt Slide Thuyết Trình — Đồ Án Độ Phức Tạp Thuật Toán

> **Dự án:** Xây dựng và Mô phỏng cơ chế Multi-Head Self-Attention từ đầu  
> **Môn học:** Độ Phức Tạp Thuật Toán

---

## 1. Tổng Quan Dự Án (Project Overview)

### Tên dự án
**DPTTT-GK** — Phân tích Độ phức tạp thuật toán Self-Attention trong xử lý ngôn ngữ tự nhiên

### Mục tiêu cốt lõi
Triển khai **từ đầu** (from scratch) toàn bộ cơ chế **Multi-Head Self-Attention** — trái tim của kiến trúc Transformer — bằng Python thuần và NumPy, **không sử dụng bất kỳ framework AI nào** (PyTorch, TensorFlow). Qua đó:
- Chứng minh bằng thực nghiệm rằng Self-Attention có **độ phức tạp thời gian O(L²)** và **độ phức tạp không gian O(L²)** (với L là chiều dài chuỗi đầu vào).
- So sánh hiệu năng giữa cài đặt **Naive** (vòng lặp for lồng nhau) và **Vectorized** (phép toán ma trận NumPy) — cùng độ phức tạp tiệm cận nhưng chênh lệch hàng trăm lần về thời gian thực tế.

### Bài toán thực tế (Problem Statement)
| Câu hỏi | Trả lời |
|---|---|
| **Self-Attention là gì?** | Cơ chế cho phép mỗi từ trong câu "nhìn" toàn bộ các từ khác để hiểu ngữ cảnh |
| **Tại sao cần phân tích?** | Chi phí O(L²) chính là nút thắt cổ chai khiến các LLM gặp giới hạn Context Window |
| **Dự án giải quyết gì?** | Mô phỏng toàn bộ pipeline từ Text → Attention → Sinh từ, và đo đạc thực nghiệm để chứng minh O(L²) |

### Đối tượng hướng đến
- Sinh viên CNTT muốn hiểu rõ cơ chế Attention hoạt động như thế nào "bên dưới lớp vỏ"
- Giảng viên đánh giá năng lực phân tích độ phức tạp thuật toán qua bài toán thực tế

---

## 2. Công Nghệ Sử Dụng & Kiến Trúc (Tech Stack & Architecture)

### Tech Stack

| Thành phần | Công nghệ | Lý do chọn |
|---|---|---|
| **Ngôn ngữ** | Python 3.12 | Dễ đọc, phù hợp cho giải thích thuật toán |
| **Thư viện tính toán** | NumPy | Cung cấp phép toán ma trận vectorized tận dụng BLAS/LAPACK |
| **Thư viện vẽ biểu đồ** | Matplotlib | Tạo đồ thị benchmark chuyên nghiệp |
| **Quản lý mã nguồn** | Git + GitHub | Quản lý phiên bản, cộng tác nhóm |
| **Kiểm thử** | unittest (built-in) | Kiểm tra tính đúng đắn của từng module |

### Lý do không dùng PyTorch/TensorFlow
> Vì mục tiêu là **hiểu và phân tích thuật toán từ nền tảng**, việc triển khai từ đầu bằng NumPy buộc phải tự cài đặt từng bước: softmax, mask, nhân ma trận, split/concat heads — giúp nắm vững bản chất toán học thay vì gọi API có sẵn.

### Kiến trúc hệ thống (Pipeline)

```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFORMER PIPELINE                      │
│                                                              │
│  ┌──────────┐   ┌───────────┐   ┌──────────────────────┐    │
│  │   Text   │──▶│ Tokenizer │──▶│ Embedding + Positional│    │
│  │  Input   │   │ (Task 5)  │   │   Encoding (Task 1)   │    │
│  └──────────┘   └───────────┘   └──────────┬───────────┘    │
│                                             │                │
│                                             ▼                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │          Multi-Head Self-Attention (Task 3)           │    │
│  │  ┌─────────────────────────────────────────────────┐  │    │
│  │  │ W_Q ──▶ Q ┐                                     │  │    │
│  │  │ W_K ──▶ K ├──▶ split_heads ──▶ Scaled Dot-Product│  │    │
│  │  │ W_V ──▶ V ┘     (Task 3)      Attention (Task 2) │  │    │
│  │  │                                      │            │  │    │
│  │  │              concat_heads ◀──────────┘            │  │    │
│  │  │                (Task 3)                           │  │    │
│  │  └──────────────────────┬──────────────────────────┘  │    │
│  │                         │ W_O                         │    │
│  └─────────────────────────┼────────────────────────────┘    │
│                             ▼                                │
│           ┌─────────────────────────────────┐                │
│           │  Output Projection ──▶ Logits   │                │
│           │  N-gram Blending ──▶ Sampling   │                │
│           │        ──▶ Token tiếp theo      │                │
│           └─────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Cấu trúc thư mục

```
DPTTT---GK/
├── core/                          # Task 1: Nền tảng toán học
│   ├── math_utils.py              #   → Softmax, Xavier Init, Positional Encoding
│   └── layers.py                  #   → Linear Layer (y = xW + b)
├── attention/                     # Task 2 & 3: Attention Engine
│   ├── scaled_dot_product.py      #   → Công thức Attention + Masking
│   └── multi_head.py              #   → Split/Concat Heads + Forward
├── data/                          # Task 5: Xử lý dữ liệu
│   └── tokenizer.py               #   → Tokenizer + Embedding + Padding
├── experiments/                   # Task 4: Đo đạc hiệu năng
│   ├── benchmark.py               #   → Naive vs Vectorized + Biểu đồ
│   └── plots/                     #   → Biểu đồ kết quả benchmark
├── tests/
│   └── test_attention.py          # 7 Unit Tests
├── main.py                        # Demo pipeline + Sinh văn bản
└── requirements.txt               # numpy, matplotlib
```

---

## 3. Các Tính Năng Cốt Lõi (Core Features)

### 3.1. Scaled Dot-Product Attention
- **Công thức:** `Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V`
- **Numerically Stable Softmax:** Trừ giá trị max trước khi tính e^x để tránh tràn số (overflow). Xử lý edge case: hàng bị mask toàn bộ trả về phân phối bằng 0 thay vị NaN.
- **Causal Mask:** Ma trận tam giác dưới (lower triangular) → mỗi token chỉ attention đến các token ở trước nó → phù hợp bài toán autoregressive (sinh từ từ trái sang phải).
- **Padding Mask:** Cho phép batch chứa các câu có độ dài khác nhau → vị trí padding bị loại khỏi tính toán attention.

### 3.2. Multi-Head Attention
- **Split Heads:** Reshape ma trận `(batch, seq_len, d_model)` thành `(batch, num_heads, seq_len, d_k)` → mỗi head xử lý một "góc nhìn" khác nhau của dữ liệu.
- **Concat Heads:** Gộp kết quả tất cả heads lại thành vector duy nhất, sau đó đẩy qua Output Projection `W_O`.
- **Projection Layers:** Bốn phép chiếu tuyến tính `W_Q`, `W_K`, `W_V`, `W_O` với Xavier Initialization → class hoạt động như một khối MHA hoàn chỉnh, tự tạo Q/K/V nội bộ.

### 3.3. Tokenizer & Embedding
- Hỗ trợ 2 chế độ: **Word-level** (tách theo từ) và **Char-level** (tách theo ký tự).
- 4 token đặc biệt: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`.
- **Batch Encoding** với tự động padding cho câu ngắn + trả về padding mask.
- **Embedding Lookup:** Tra bảng ma trận embedding để chuyển Token ID thành vector.

### 3.4. Positional Encoding
- Sinusoidal Positional Encoding theo công thức gốc từ paper "Attention is All You Need".
- Bổ sung thông tin thứ tự token vì Self-Attention bản chất không phân biệt được vị trí.

### 3.5. Autoregressive Generation
- Pipeline sinh văn bản tiếng Việt theo vòng lặp tự hồi quy.
- Kết hợp xác suất từ **model output** (qua Attention) và **N-gram Language Model** (Trigram + Bigram) → văn bản sinh ra có ngữ nghĩa tự nhiên hơn.
- Tham số điều chỉnh: `temperature` (độ ngẫu nhiên), `blend_ratio` (tỷ lệ pha trộn n-gram).

### 3.6. Benchmark Naive vs Vectorized
- **Bản Naive:** 5 vòng lặp for lồng nhau (batch → head → query → key → d_k) → đúng thuật toán gốc, nhưng cực kỳ chậm.
- **Bản Vectorized:** Dùng `np.matmul` để tính toán song song toàn bộ heads + toàn bộ batch trong một lệnh → nhanh hơn hàng trăm lần.
- Xác minh tự động: Sai số giữa 2 phiên bản < 10⁻⁴ → cùng kết quả, khác tốc độ.
- Xuất bảng dữ liệu dạng Markdown + LaTeX → dán trực tiếp vào báo cáo.

---

## 4. Điểm Nhấn Kỹ Thuật & Kết Quả (Technical Highlights & Results)

### 4.1. Phân tích Độ phức tạp lý thuyết

| Phép tính | Độ phức tạp thời gian | Độ phức tạp không gian |
|---|---|---|
| Tính Score `QKᵀ` | O(B · H · L² · d_k) | O(B · H · L²) |
| Softmax trên Attention Matrix | O(B · H · L²) | O(B · H · L²) |
| Nhân Attention Weights × V | O(B · H · L² · d_k) | O(B · H · L · d_k) |
| **Tổng Attention** | **O(B · H · L² · d_k)** | **O(B · H · L²)** |

> **Kết luận:** Khi L tăng gấp đôi → chi phí tăng **gấp 4 lần** (bậc hai). Đây là nút thắt chính của Transformer truyền thống.

*(Trong đó: B = batch size, H = số heads, L = sequence length, d_k = chiều mỗi head)*

### 4.2. Kết quả Benchmark thực nghiệm

| Sequence Length (L) | Naive (ms) | Vectorized (ms) | Speedup |
|---|---|---|---|
| 10 | 6.26 | 0.51 | **~12x** |
| 20 | 21.65 | 0.54 | **~40x** |
| 50 | 127.91 | 0.89 | **~143x** |
| 100 | N/A | 2.45 | — |

> **Insight:** Ở L=50, bản Vectorized nhanh hơn bản Naive khoảng **143 lần**. Cả hai đều có cùng O(L²), nhưng NumPy khai thác tối ưu phần cứng (SIMD, cache locality, BLAS) nên hằng số ẩn (hidden constant) nhỏ hơn rất nhiều.

### 4.3. Các kỹ thuật tối ưu đáng chú ý

- **Stable Softmax:** Trừ max trước khi tính exp → tránh overflow. Xử lý hàng toàn `-inf` (fully masked) bằng cách trả về vector 0 thay vì NaN → đảm bảo pipeline không bị crash.
- **Xavier/Glorot Initialization:** Phương sai trọng số `Var(W) = 2/(fan_in + fan_out)` → giữ gradient ổn định qua các lớp, tránh vanishing/exploding gradient.
- **Naive Loop Optimization:** Chuyển ma trận NumPy thành Python list (`tolist()`) trước khi lặp → giảm overhead truy xuất phần tử từ ~5x, giúp phép đo Naive phản ánh đúng chi phí thuật toán thuần túy.
- **N-gram Blending:** Kết hợp xác suất model + n-gram với Laplace smoothing → sinh văn bản tiếng Việt có nghĩa mà không cần huấn luyện neural network.

### 4.4. Kết quả Unit Tests
- **7/7 tests PASSED** — bao gồm kiểm thử: causal mask, padding mask, fully-masked rows, split/concat heads, forward MHA, positional encoding, tokenizer batch encode.

### 4.5. Kết quả sinh văn bản (Demo)
```
Seed: "Tôi đang"  → "Tôi đang làm đồ án phân tích thuật toán"  ✅ Có nghĩa
Seed: "Học máy"   → "Học máy là lĩnh vực rất thú vị"           ✅ Có nghĩa
Seed: "Xin chào"  → "Xin chào thế giới"                        ✅ Có nghĩa
```

---

## 5. Gợi Ý Phân Chia Bố Cục Slide (10 Slides)

---

### 📄 Slide 1: Trang bìa (Title Slide)

**Nội dung chính:**
- Tên đồ án: **"Phân tích Độ phức tạp thuật toán Self-Attention trong Xử lý Ngôn ngữ Tự nhiên"**
- Môn học: Độ Phức Tạp Thuật Toán
- Thông tin nhóm: Họ tên thành viên, MSSV, Lớp
- Logo trường / Khoa

**Note thuyết trình:**
> "Chào thầy/cô và các bạn. Hôm nay nhóm chúng em sẽ trình bày đồ án giữa kỳ môn Độ phức tạp thuật toán. Đề tài của nhóm là phân tích độ phức tạp của thuật toán Self-Attention — cơ chế cốt lõi trong các mô hình Transformer đang thống trị lĩnh vực AI hiện nay."

---

### 📄 Slide 2: Đặt vấn đề — Self-Attention là gì?

**Nội dung chính:**
- Transformer là kiến trúc đứng sau ChatGPT, Gemini, Claude — và trái tim của nó là **Self-Attention**
- Self-Attention cho phép mỗi từ trong câu "nhìn" toàn bộ các từ còn lại để hiểu ngữ cảnh
- Ví dụ minh họa: *"Con **mèo** ngồi trên **bàn**, **nó** đang ngủ"* → "nó" cần attention đến "mèo" chứ không phải "bàn"
- **Vấn đề:** Chi phí O(L²) — khi câu dài gấp đôi, chi phí tăng gấp 4 → giới hạn context window của LLM

**Note thuyết trình:**
> "Trước khi đi vào chi tiết kỹ thuật, nhóm muốn giải thích tại sao chọn đề tài này. Mọi mô hình ngôn ngữ lớn đều dựa trên Transformer, và thành phần quan trọng nhất là Self-Attention. Khi ta nói 'con mèo ngồi trên bàn, nó đang ngủ', con người hiểu 'nó' là 'con mèo'. Self-Attention giúp máy tính hiểu được điều tương tự. Tuy nhiên, chi phí tính toán tăng theo bình phương độ dài câu — đây là nút thắt lớn nhất. Đề tài của nhóm sẽ phân tích và chứng minh điều này."

---

### 📄 Slide 3: Mục tiêu & Phạm vi đồ án

**Nội dung chính:**
- 🎯 **Mục tiêu 1:** Triển khai từ đầu (from scratch) toàn bộ Multi-Head Self-Attention bằng Python + NumPy (không dùng PyTorch/TensorFlow)
- 🎯 **Mục tiêu 2:** Phân tích lý thuyết độ phức tạp O(L²) về thời gian và không gian
- 🎯 **Mục tiêu 3:** Chứng minh bằng thực nghiệm — so sánh Naive (vòng lặp) vs Vectorized (ma trận)
- 🎯 **Mục tiêu 4:** Tích hợp pipeline hoàn chỉnh: Text → Tokenizer → Attention → Sinh văn bản tiếng Việt
- **Tech Stack:** Python 3.12 · NumPy · Matplotlib · Git/GitHub

**Note thuyết trình:**
> "Nhóm đặt ra 4 mục tiêu chính. Điểm đặc biệt là nhóm tự cài đặt hoàn toàn từ đầu, không sử dụng bất kỳ framework AI nào. Điều này buộc nhóm phải hiểu rõ từng bước toán học: từ softmax, nhân ma trận, đến masking. Chỉ dùng NumPy cho phép toán ma trận và Matplotlib để vẽ biểu đồ."

---

### 📄 Slide 4: Kiến trúc hệ thống & Pipeline

**Nội dung chính:**
- Sơ đồ pipeline (dùng hình minh họa hoặc flowchart):
  ```
  Text → Tokenizer → Embedding + Positional Encoding
       → Multi-Head Self-Attention (Q, K, V)
       → Output Projection → Logits → Sampling → Token tiếp theo
  ```
- Cấu trúc thư mục dự án (5 modules: `core/`, `attention/`, `data/`, `experiments/`, `tests/`)
- Mỗi module tương ứng một Task phân công trong nhóm

**Note thuyết trình:**
> "Đây là kiến trúc tổng thể. Dữ liệu đầu vào là văn bản tiếng Việt, đi qua Tokenizer để chuyển thành số, cộng thêm Positional Encoding để mã hóa vị trí, rồi đưa vào Multi-Head Attention. Kết quả attention đi qua lớp output để sinh logits, kết hợp với mô hình n-gram để sampling ra từ tiếp theo. Nhóm chia dự án thành 5 task, mỗi task là một module code riêng biệt."

---

### 📄 Slide 5: Công thức Toán học — Scaled Dot-Product Attention

**Nội dung chính:**
- Công thức chính: **Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V**
- Giải thích từng thành phần:
  - **Q (Query):** "Tôi đang tìm gì?"
  - **K (Key):** "Tôi có gì để cung cấp?"
  - **V (Value):** "Thông tin thực tế tôi mang theo"
  - **√d_k:** Hệ số scale để tránh dot product quá lớn
- **Stable Softmax:** `softmax(x_i) = e^(x_i − max(x)) / Σe^(x_j − max(x))`
- **Causal Mask:** Ma trận tam giác dưới — mỗi token chỉ nhìn về phía trước

**Note thuyết trình:**
> "Đây là công thức cốt lõi. Query, Key, Value là 3 phép chiếu tuyến tính từ cùng một input. Chúng ta nhân Q với K chuyển vị để tính điểm tương đồng giữa mọi cặp token, chia cho căn bậc hai d_k để ổn định, áp dụng softmax để thành xác suất, rồi nhân với V. Nhóm cài đặt stable softmax bằng cách trừ max trước khi tính exp, tránh tràn số. Causal mask đảm bảo token chỉ attention đến các token trước nó — cần thiết cho việc sinh từ tuần tự."

---

### 📄 Slide 6: Multi-Head Attention — Cơ chế nhiều đầu

**Nội dung chính:**
- Minh họa Split Heads: `(batch, seq_len, 512)` → `(batch, 8 heads, seq_len, 64)`
- Mỗi head học một "góc nhìn" (pattern) khác nhau: cú pháp, ngữ nghĩa, vị trí...
- Concat Heads: Gộp 8 đầu × 64 chiều = 512 chiều → đẩy qua Output Projection W_O
- 4 ma trận trọng số: W_Q, W_K, W_V, W_O — khởi tạo bằng Xavier Initialization
- Xavier Init: `Var(W) = 2/(fan_in + fan_out)` → giữ gradient ổn định

**Note thuyết trình:**
> "Thay vì chạy 1 lần attention trên toàn bộ 512 chiều, ta chia thành 8 heads, mỗi head xử lý 64 chiều. Mỗi head có thể học một pattern khác nhau — ví dụ head 1 học cú pháp, head 2 học ngữ nghĩa. Sau đó gộp kết quả lại. Kỹ thuật Xavier Initialization đảm bảo trọng số không quá lớn cũng không quá nhỏ, giúp tránh hiện tượng vanishing gradient."

---

### 📄 Slide 7: Phân tích Độ phức tạp lý thuyết

**Nội dung chính:**
- Bảng phân tích chi tiết:

| Phép tính | Time Complexity | Space Complexity |
|---|---|---|
| Tính QKᵀ | O(B·H·**L²**·d_k) | O(B·H·**L²**) |
| Softmax | O(B·H·**L²**) | O(B·H·**L²**) |
| Nhân weights × V | O(B·H·**L²**·d_k) | O(B·H·L·d_k) |

- **Kết luận:** Cả thời gian và bộ nhớ đều tăng theo **O(L²)**
- Ví dụ trực quan: L=1000 → ma trận attention 1000×1000 = 1 triệu phần tử **cho mỗi head**

**Note thuyết trình:**
> "Phần quan trọng nhất của đồ án. Khi tính QK chuyển vị, mỗi token phải tính dot product với mọi token khác — tạo ra ma trận L×L. Với L=1000 và 8 heads, ta cần lưu 8 triệu phần tử chỉ riêng cho attention weights. Khi L tăng gấp đôi lên 2000, chi phí tăng gấp 4 lần. Đây chính là lý do các mô hình LLM hiện nay cần kỹ thuật tối ưu như FlashAttention."

---

### 📄 Slide 8: Kết quả Benchmark thực nghiệm

**Nội dung chính:**
- **Chèn 2 biểu đồ** từ thư mục `experiments/plots/`:
  - Biểu đồ 1: So sánh thời gian Naive vs Vectorized
  - Biểu đồ 2: Đường cong tăng trưởng Vectorized (thời gian + bộ nhớ) so với O(L²) lý thuyết
- Bảng số liệu:

| L | Naive | Vectorized | Speedup |
|---|---|---|---|
| 10 | 6.26 ms | 0.51 ms | 12x |
| 50 | 127.91 ms | 0.89 ms | **143x** |

- **Insight:** Cùng O(L²) nhưng vectorization nhanh hơn ~143 lần nhờ SIMD, cache locality, BLAS

**Note thuyết trình:**
> "Đây là kết quả thực nghiệm chứng minh phân tích lý thuyết. Biểu đồ bên trái cho thấy bản Naive chậm hơn rất nhiều — ở L=50 chậm hơn 143 lần. Biểu đồ bên phải cho thấy đường cong thời gian thực tế của bản Vectorized khớp rất tốt với đường cong lý thuyết O(L²) — xác nhận độ phức tạp bậc hai. Bộ nhớ cũng tăng phi tuyến theo L². Hai phiên bản cho cùng kết quả số (sai số < 10⁻⁴) nhưng khác nhau về hằng số ẩn do cách khai thác phần cứng."

---

### 📄 Slide 9: Demo — Pipeline sinh văn bản tiếng Việt

**Nội dung chính:**
- Ảnh chụp màn hình output của `main.py` hoặc GIF demo
- 3 ví dụ tiêu biểu:
  - `"Tôi đang"` → *"Tôi đang làm đồ án phân tích thuật toán"*
  - `"Học máy"` → *"Học máy là lĩnh vực rất thú vị"*
  - `"Xin chào"` → *"Xin chào thế giới"*
- Giải thích cơ chế: Blend 90% xác suất N-gram + 10% xác suất Attention → văn bản có nghĩa
- **7/7 Unit Tests PASSED** ✅

**Note thuyết trình:**
> "Đây là demo chạy thực tế. Pipeline nhận văn bản seed và sinh từ tiếp theo theo vòng lặp autoregressive. Ví dụ từ 'Tôi đang' sinh ra 'Tôi đang làm đồ án phân tích thuật toán' — một câu hoàn toàn có nghĩa. Cơ chế hoạt động là pha trộn 90% xác suất từ n-gram (học từ corpus) với 10% từ output của Attention. Toàn bộ 7 unit tests đều pass, xác nhận tính đúng đắn của từng module."

---

### 📄 Slide 10: Kết luận & Hướng phát triển

**Nội dung chính:**
- ✅ **Đã hoàn thành:**
  - Triển khai from scratch toàn bộ Multi-Head Self-Attention
  - Chứng minh O(L²) bằng cả lý thuyết và thực nghiệm
  - So sánh Naive vs Vectorized → Vectorized nhanh hơn ~143x
  - Pipeline hoàn chỉnh: Text → Attention → Sinh văn bản tiếng Việt
  - 7/7 Unit Tests passed
- 🔮 **Hướng phát triển:**
  - Cài đặt FlashAttention (O(L²) thời gian nhưng O(L) bộ nhớ)
  - Thêm Encoder-Decoder cross-attention cho bài toán dịch máy
  - Cài đặt backward pass (backpropagation) để huấn luyện thật
  - So sánh với Sparse/Linear Attention

**Note thuyết trình:**
> "Tổng kết lại, nhóm đã hoàn thành tất cả mục tiêu đề ra: cài đặt from scratch, phân tích lý thuyết, và chứng minh bằng thực nghiệm rằng Self-Attention có độ phức tạp O(L²). Nếu phát triển thêm, nhóm muốn cài đặt FlashAttention — một kỹ thuật giúp giữ O(L²) thời gian nhưng giảm bộ nhớ xuống O(L) bằng cách tính attention theo block, tránh phải lưu toàn bộ ma trận L×L. Cảm ơn thầy/cô và các bạn đã lắng nghe."

---

## 📎 Phụ lục — Tài nguyên hỗ trợ làm slide

### Hình ảnh có sẵn trong dự án (chèn trực tiếp vào slide):
- `experiments/plots/naive_vs_vectorized_time.png` — Biểu đồ so sánh thời gian
- `experiments/plots/vectorized_scaling.png` — Biểu đồ tăng trưởng + so sánh O(L²) lý thuyết

### Gợi ý hình ảnh nên tìm thêm:
- Sơ đồ Transformer Architecture từ paper gốc "Attention is All You Need" (Vaswani et al., 2017)
- Minh họa trực quan Multi-Head Attention (split/concat heads)
- Heatmap attention weights (ví dụ từ Jay Alammar's blog "The Illustrated Transformer")

### Tham khảo:
- Vaswani, A., et al. (2017). *"Attention Is All You Need"*. NeurIPS.
- Jay Alammar. *"The Illustrated Transformer"*. [jalammar.github.io](https://jalammar.github.io/illustrated-transformer/)
