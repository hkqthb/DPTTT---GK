# DPTTT---GK (Đồ án Môn Phân tích Độ phức tạp Thuật toán)
Đồ án nghiên cứu và mô phỏng cơ chế **Multi-Head Self-Attention** từ đầu (from scratch) bằng Python và NumPy, thực hiện phân tích thực nghiệm độ phức tạp thuật toán (Time & Space Complexity) của mô hình.

---

## 📁 Cấu trúc Thư mục Dự án

Hiện tại dự án được tổ chức theo các nhánh phát triển và cấu trúc phân rã module như sau:

```plaintext
transformer_project/
│
├── src/
│   └── multi_head.py       # TASK 3: Tách/gộp Head, Điều phối Multi-Head Attention
│
├── experiments/            # TASK 4: Đo lường & Thực nghiệm (Branch: task4-profiling)
│   ├── benchmark.py        # Kịch bản đo đạc RAM và Thời gian (Naive vs Vectorized)
│   └── plots/              # Thư mục lưu đồ thị biểu diễn độ phức tạp xuất ra
│       ├── naive_vs_vectorized_time.png   # So sánh tốc độ Naive vs Vectorized (L <= 500)
│       └── vectorized_scaling.png         # Sự tăng trưởng của bản Vectorized (L <= 5000)
│
├── tests/
│   └── test_multi_head.py  # Script kiểm thử nhanh khối Multi-Head Attention
│
├── README.md               # Hướng dẫn dự án và Báo cáo nhanh kết quả thực nghiệm
└── .gitignore
```

---

## 📉 TASK 4: Thực nghiệm & Đo đạc Hiệu năng (Empirical Profiling & Benchmarking)

Nhánh `task4-profiling` triển khai chi tiết việc đo đạc thực nghiệm để chứng minh độ phức tạp Big-O trên máy tính thực tế.

### 1. Thuật toán so sánh
*   **Naive Implementation (Attention "Ngây thơ"):** Sử dụng các vòng lặp `for` lồng nhau để tính Attention Scores, tính xác suất Softmax và nhân với ma trận Value $V$. Để cải thiện tốc độ của Python thuần, các mảng NumPy đã được chuyển đổi thành cấu trúc Python list trước khi lặp nhằm giảm thiểu overhead truy cập.
*   **Vectorized Implementation (Attention tối ưu):** Sử dụng các phép toán ma trận song song thông qua thư viện NumPy (`np.matmul` và `np.transpose`), khai thác tối đa sức mạnh của phần cứng.

### 2. Cách chạy thực nghiệm
Yêu cầu hệ thống cần cài đặt thư viện `numpy` và `matplotlib`. Từ thư mục gốc của dự án, chạy lệnh sau:
```bash
python3 experiments/benchmark.py
```

Sau khi chạy xong, chương trình sẽ tự động thực hiện:
1.  **Kiểm tra độ chính xác (Unit Test):** Đối chiếu kết quả đầu ra của bản Naive và Vectorized (chấp nhận sai số $< 10^{-5}$) để đảm bảo logic tính toán hoàn toàn trùng khớp.
2.  **Đo đạc hiệu năng:** Đo thời gian trung bình (ms) và bộ nhớ RAM đỉnh (MB) tiêu tốn cho các chiều dài ngữ cảnh $L$ tăng dần (Naive chạy từ $L=10$ đến $L=500$, Vectorized chạy từ $L=10$ đến $L=5000$).
3.  **Vẽ đồ thị:** Xuất ra 2 biểu đồ trực quan trong thư mục `experiments/plots/`.
4.  **In bảng báo cáo:** Xuất bảng dữ liệu dưới định dạng Markdown và LaTeX.

---

## 📊 Kết quả Thực nghiệm Thực tế

Dưới đây là dữ liệu đo đạc thực tế chạy trên hệ thống:

### 1. Bảng so sánh hiệu năng (Markdown format)

| Độ dài câu (L) | Thời gian Naive (ms) | Bộ nhớ Naive (MB) | Thời gian Vectorized (ms) | Bộ nhớ Vectorized (MB) | Tốc độ tăng (Speedup) |
|:---|:---|:---|:---|:---|:---|
| 10             | 8.64                 | 0.0000            | 0.19                      | 0.0602                 | 45.7x                 |
| 50             | 143.07               | 0.0000            | 0.71                      | 0.5310                 | 201.6x                |
| 100            | 411.08               | 0.0000            | 2.94                      | 1.9172                 | 140.0x                |
| 250            | 2376.08              | 0.0000            | 15.18                     | 11.5979                | 156.5x                |
| 500            | 9692.43              | 0.0000            | 67.09                     | 46.1395                | 144.5x                |
| 1000           | N/A                  | N/A               | 123.39                    | 184.2449               | N/A                   |
| 1500           | N/A                  | N/A               | 244.95                    | 414.3760               | N/A                   |
| 2000           | N/A                  | N/A               | 415.68                    | 736.5444               | N/A                   |
| 3000           | N/A                  | N/A               | 1120.91                   | 1656.9469              | N/A                   |
| 4000           | N/A                  | N/A               | 2440.19                   | 2945.4983              | N/A                   |
| 5000           | N/A                  | N/A               | 3380.81                   | 4602.0916              | N/A                   |

> *Ghi chú: Bộ nhớ đỉnh của bản Naive được bỏ qua (0.0000 MB) để tránh overhead ghi nhận của thư viện `tracemalloc` lên hàng trăm ngàn phần tử list.*

### 2. Dạng LaTeX Table (Sao chép để dán vào file báo cáo `.tex` của nhóm)

```latex
\begin{table}[H]
    \centering
    \begin{tabular}{|r|r|r|r|r|c|}
        \hline
        \rowcolor{darkblue} \textcolor{white}{\textbf{Độ dài L}} & \textcolor{white}{\textbf{Time Naive (ms)}} & \textcolor{white}{\textbf{RAM Naive (MB)}} & \textcolor{white}{\textbf{Time Vector (ms)}} & \textcolor{white}{\textbf{RAM Vector (MB)}} & \textcolor{white}{\textbf{Tốc độ tăng}} \\
        \hline
        10 & 8.64 & 0.0000 & 0.19 & 0.0602 & 45.7x \\
        \hline
        50 & 143.07 & 0.0000 & 0.71 & 0.5310 & 201.6x \\
        \hline
        100 & 411.08 & 0.0000 & 2.94 & 1.9172 & 140.0x \\
        \hline
        250 & 2376.08 & 0.0000 & 15.18 & 11.5979 & 156.5x \\
        \hline
        500 & 9692.43 & 0.0000 & 67.09 & 46.1395 & 144.5x \\
        \hline
        1000 & -- & -- & 123.39 & 184.2449 & -- \\
        \hline
        1500 & -- & -- & 244.95 & 414.3760 & -- \\
        \hline
        2000 & -- & -- & 415.68 & 736.5444 & -- \\
        \hline
        3000 & -- & -- & 1120.91 & 1656.9469 & -- \\
        \hline
        4000 & -- & -- & 2440.19 & 2945.4983 & -- \\
        \hline
        5000 & -- & -- & 3380.81 & 4602.0916 & -- \\
        \hline
    \end{tabular}
    \caption{Bảng so sánh hiệu năng giữa hai phương pháp Attention}
    \label{tab:attention_comparison}
\end{table}
```

---

## 📈 Phân tích Kết quả Thực nghiệm

1.  **Về Độ phức tạp Thời gian (Time Complexity):**
    *   Cả hai phiên bản đều chứng minh rõ ràng quy luật tăng trưởng bậc hai $O(L^2)$ theo lý thuyết. Khi $L$ tăng gấp đôi (ví dụ từ $250 \to 500$), thời gian chạy tăng xấp xỉ 4 lần (Naive từ $2.37 \to 9.69$ giây, Vectorized từ $15 \to 67$ mili-giây).
    *   Phiên bản Vectorized cho thấy tốc độ vượt trội gấp **140x - 200x** lần so với bản Naive. Điều này chứng minh hiệu quả cực lớn của tính toán song song ma trận trong NumPy (được tối ưu hóa bằng BLAS/LAPACK viết bằng C/Fortran) so với việc sử dụng các vòng lặp tuần tự trong Python.
2.  **Về Độ phức tạp Không gian (Space Complexity):**
    *   Dung lượng RAM tiêu tốn cho bản Vectorized tăng trưởng theo hàm parabol $O(L^2)$ (do phải lưu trữ ma trận điểm số tương quan kích thước $L \times L$). 
    *   Khi $L=5000$, bộ nhớ đỉnh cần cấp phát để tính toán attention pass đã lên tới **4.6 GB** (với kiểu dữ liệu `float32`). Điều này minh chứng cho "bức tường bộ nhớ" (Memory Wall) của cơ chế Attention cổ điển, giải thích vì sao các mô hình LLMs cần các giải thuật cải tiến như FlashAttention để giảm Space Complexity xuống $O(L)$.
# DPTTT---GK
Đồ án môn Độ phức tạp thuật toán - Xây dựng Self-Attention

## Cách chạy  
1. Tạo môi trường ảo
   `python3 -m venv .venv`
   `source .venv/bin/activate`
2. Cài thư viện
   `pip install -r requirements.txt`
3. Chạy chương trình
   `python3 main.py`
