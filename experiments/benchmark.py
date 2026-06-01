import os
import sys
import time
import tracemalloc
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Thêm thư mục gốc vào path để import các package attention/core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention.multi_head import MultiHeadAttention
from attention.scaled_dot_product import scaled_dot_product_attention
from core.math_utils import stable_softmax

# =====================================================================
# 1. SOFTMAX & ATTENTION IMPLEMENTATIONS
# =====================================================================

def vectorized_attention_core(Q, K, V, mask=None, causal=False):
    """
    TASK 2: Phiên bản Vector hóa sử dụng toán ma trận của NumPy.
    Input shape: (batch_size, num_heads, seq_length, d_k)
    Output shape: (batch_size, num_heads, seq_length, d_k)
    """
    output, _ = scaled_dot_product_attention(Q, K, V, mask=mask, causal=causal)
    return output


def _build_valid_mask(mask, causal, target_shape):
    """
    Tạo mask bool cho bản naive.
    Quy ước: True = vị trí được phép attention.
    """
    if isinstance(mask, (bool, np.bool_)):
        causal = causal or bool(mask)
        mask = None

    batch_size, num_heads, query_len, key_len = target_shape
    valid_mask = np.ones(target_shape, dtype=bool)

    if causal:
        causal_mask = np.tril(np.ones((query_len, key_len), dtype=bool))
        valid_mask &= causal_mask[np.newaxis, np.newaxis, :, :]

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim == 2 and mask.shape == (batch_size, key_len):
            mask = mask[:, np.newaxis, np.newaxis, :]
        elif mask.ndim == 2 and mask.shape == (query_len, key_len):
            mask = mask[np.newaxis, np.newaxis, :, :]
        elif mask.ndim == 3 and mask.shape == (batch_size, query_len, key_len):
            mask = mask[:, np.newaxis, :, :]
        valid_mask &= np.broadcast_to(mask, target_shape)

    return valid_mask


def naive_attention_core(Q, K, V, mask=None, causal=False):
    """
    TASK 4: Phiên bản Attention "ngây thơ" dùng vòng lặp for lồng nhau (O(L^2) Naive).
    Không dùng các hàm nhân ma trận song song của NumPy ở chiều seq_length.
    Để tối ưu hóa tốc độ lặp của Python, ta chuyển các ma trận NumPy sang Python list
    trước khi lặp, giúp tránh overhead truy xuất của NumPy (bound checking, wrap scalars).
    Input shape: (batch_size, num_heads, seq_length, d_k)
    Output shape: (batch_size, num_heads, seq_length, d_k)
    """
    batch_size, num_heads, seq_length, d_k = Q.shape
    output = np.zeros_like(V)
    valid_mask = _build_valid_mask(mask, causal, (batch_size, num_heads, seq_length, seq_length))
    
    # Chuyển đổi sang Python list để tăng tốc độ truy cập phần tử trong vòng lặp Python thuần
    Q_list = Q.tolist()
    K_list = K.tolist()
    V_list = V.tolist()
    output_list = output.tolist()
    
    for b in range(batch_size):
        for h in range(num_heads):
            q_head = Q_list[b][h]
            k_head = K_list[b][h]
            v_head = V_list[b][h]
            out_head = output_list[b][h]
            
            # Tính Attention Scores từng phần tử bằng vòng lặp
            scores = [[-1e30] * seq_length for _ in range(seq_length)]
            for i in range(seq_length):
                q_i = q_head[i]
                for j in range(seq_length):
                    if not valid_mask[b, h, i, j]:
                        continue
                    k_j = k_head[j]
                    dot_prod = 0.0
                    for d in range(d_k):
                        dot_prod += q_i[d] * k_j[d]
                    scores[i][j] = dot_prod / np.sqrt(d_k)
            
            # Áp dụng Softmax theo hàng bằng vòng lặp
            probs = [[0.0] * seq_length for _ in range(seq_length)]
            for i in range(seq_length):
                # Tìm max của hàng i để ổn định số học
                row_max = -1e30
                for j in range(seq_length):
                    if scores[i][j] > row_max:
                        row_max = scores[i][j]
                
                # Tính tổng lũy thừa exp
                sum_exp = 0.0
                for j in range(seq_length):
                    sum_exp += np.exp(scores[i][j] - row_max)
                
                # Tính xác suất
                for j in range(seq_length):
                    probs[i][j] = np.exp(scores[i][j] - row_max) / sum_exp
            
            # Nhân với ma trận Value V bằng vòng lặp
            for i in range(seq_length):
                prob_i = probs[i]
                for d in range(d_k):
                    val = 0.0
                    for j in range(seq_length):
                        val += prob_i[j] * v_head[j][d]
                    out_head[i][d] = val
                    
    return np.array(output_list, dtype=Q.dtype)

# =====================================================================
# 2. ACCURACY VERIFICATION (UNIT TEST)
# =====================================================================

def verify_correctness():
    """
    Kiểm tra xem bản Vectorized và Naive có cho kết quả hoàn toàn trùng khớp không.
    """
    print("=== ĐANG KIỂM TRA ĐỘ CHÍNH XÁC CỦA HAI THUẬT TOÁN ===")
    b, h, l, d_k = 2, 4, 16, 32
    Q = np.random.randn(b, h, l, d_k).astype(np.float32)
    K = np.random.randn(b, h, l, d_k).astype(np.float32)
    V = np.random.randn(b, h, l, d_k).astype(np.float32)
    
    out_vec = vectorized_attention_core(Q, K, V, causal=True)
    out_naive = naive_attention_core(Q, K, V, causal=True)
    
    # So sánh sai số tuyệt đối
    diff = np.max(np.abs(out_vec - out_naive))
    print(f"Sai số lớn nhất giữa Vectorized và Naive: {diff:.2e}")
    assert diff < 1e-4, "Lỗi: Kết quả giữa 2 phiên bản không khớp nhau!"
    print("Chúc mừng! Kết quả của hai phiên bản trùng khớp hoàn toàn.\n")

# =====================================================================
# 3. BENCHMARKING SUITE
# =====================================================================

def run_benchmarks(quick=False):
    # Tham số cố định
    d_model = 256
    num_heads = 8
    batch_size = 1
    
    # Khởi tạo mô hình Multi-Head Attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # Tập kích thước Sequence Length cần đo đạc
    # Naive sẽ chạy với tập kích thước nhỏ hơn vì cực kỳ chậm ở L lớn
    if quick:
        Ns_naive = [10, 20, 50]
        Ns_vectorized = [10, 20, 50, 100]
    else:
        Ns_naive = [10, 50, 100, 250, 500]
        # Vectorized chạy hết dải rộng để chứng minh O(L^2) thực tế
        Ns_vectorized = [10, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]
    
    naive_times = []
    naive_memories = []
    vectorized_times = []
    vectorized_memories = []
    
    print("=== BẮT ĐẦU ĐO ĐẠC HIỆU NĂNG ===")
    print(f"Tham số: d_model={d_model}, num_heads={num_heads}, batch_size={batch_size}\n")
    
    # --- 3.1 Đo đạc bản Naive ---
    print("--- Đang đo đạc Naive Attention ---")
    for N in Ns_naive:
        # Khởi tạo ma trận ngẫu nhiên
        X = np.random.randn(batch_size, N, d_model).astype(np.float32)
        
        # Xác định số lần chạy (runs) để lấy trung bình (giảm số lần ở L lớn để tránh chờ lâu)
        runs = 3 if N <= 200 else 1
        
        # Đo thời gian
        start_time = time.perf_counter()
        for _ in range(runs):
            _ = mha.forward(X, causal=True, attention_fn=naive_attention_core)
        end_time = time.perf_counter()
        avg_time = ((end_time - start_time) / runs) * 1000  # ms
        
        # Đo bộ nhớ đỉnh (Bỏ qua đo bộ nhớ cho Naive để tránh overhead của tracemalloc trên Python lists)
        peak_mb = 0.0
        
        naive_times.append(avg_time)
        naive_memories.append(peak_mb)
        print(f"L = {N:<5} | Thời gian: {avg_time:>10.2f} ms | Bộ nhớ đỉnh: {peak_mb:>8.4f} MB")
        
    print("\n--- Đang đo đạc Vectorized Attention ---")
    # --- 3.2 Đo đạc bản Vectorized ---
    for N in Ns_vectorized:
        X = np.random.randn(batch_size, N, d_model).astype(np.float32)
        
        runs = 10 if N <= 1000 else 5
        
        # Đo thời gian
        start_time = time.perf_counter()
        for _ in range(runs):
            _ = mha.forward(X, causal=True, attention_fn=vectorized_attention_core)
        end_time = time.perf_counter()
        avg_time = ((end_time - start_time) / runs) * 1000  # ms
        
        # Đo bộ nhớ đỉnh
        tracemalloc.start()
        _ = mha.forward(X, causal=True, attention_fn=vectorized_attention_core)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)  # MB
        
        vectorized_times.append(avg_time)
        vectorized_memories.append(peak_mb)
        print(f"L = {N:<5} | Thời gian: {avg_time:>10.2f} ms | Bộ nhớ đỉnh: {peak_mb:>8.4f} MB")
        
    return Ns_naive, naive_times, naive_memories, Ns_vectorized, vectorized_times, vectorized_memories

# =====================================================================
# 4. PLOTTING & SAVE GRAPH
# =====================================================================

def plot_results(Ns_naive, naive_times, naive_memories, Ns_vectorized, vectorized_times, vectorized_memories):
    # Đảm bảo thư mục lưu đồ thị tồn tại
    plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # -----------------------------------------------------------------
    # ĐỒ THỊ 1: So sánh thời gian thực thi (Naive vs Vectorized)
    # -----------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(Ns_naive, naive_times, marker='o', color='#d62728', label='Naive (Vòng lặp for O(L^2))', linewidth=2)
    # Chỉ vẽ phần Vectorized có cùng trục độ dài với Naive để so sánh trực quan
    plt.plot(Ns_naive, vectorized_times[:len(Ns_naive)], marker='s', color='#1f77b4', label='Vectorized (NumPy O(L^2))', linewidth=2)
    
    plt.title('So sánh Thời gian Thực thi: Naive vs Vectorized Attention', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Độ dài Sequence (L)', fontsize=12)
    plt.ylabel('Thời gian thực thi trung bình (ms)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    # Annotate tỷ lệ tăng tốc (Speedup) tại L lớn nhất
    ratio = naive_times[-1] / vectorized_times[len(Ns_naive)-1]
    plt.annotate(f'Vectorized nhanh hơn {ratio:.1f}x\ntại L={Ns_naive[-1]}', 
                 xy=(Ns_naive[-1], vectorized_times[len(Ns_naive)-1]), 
                 xytext=(Ns_naive[-2], naive_times[-1] * 0.4),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#e6f2ff", ec="#1f77b4", lw=1))
    
    plt.tight_layout()
    plot1_path = os.path.join(plot_dir, 'naive_vs_vectorized_time.png')
    plt.savefig(plot1_path, dpi=300)
    print(f"\n[Đồ thị] Đã lưu biểu đồ so sánh thời gian tại: {plot1_path}")
    plt.close()

    # -----------------------------------------------------------------
    # ĐỒ THỊ 2: Sự tăng trưởng của bản Vectorized (Thời gian & Bộ nhớ)
    # -----------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = '#1f77b4'
    ax1.set_xlabel('Độ dài Sequence (L)', fontsize=12)
    ax1.set_ylabel('Thời gian thực thi trung bình (ms)', color=color, fontsize=12)
    line1 = ax1.plot(Ns_vectorized, vectorized_times, marker='o', color=color, label='Thời gian đo thực tế (ms)', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Vẽ đường cong lý thuyết O(L^2) để đối sánh (Scale dựa theo điểm cuối)
    c = vectorized_times[-1] / (Ns_vectorized[-1] ** 2)
    theoretical_times = [c * (n ** 2) for n in Ns_vectorized]
    line2 = ax1.plot(Ns_vectorized, theoretical_times, linestyle='--', color='#ff7f0e', label='Độ phức tạp lý thuyết O(L^2)', linewidth=1.5)
    
    # Trục phụ cho Bộ nhớ
    ax2 = ax1.twinx()  
    color = '#2ca02c'
    ax2.set_ylabel('Dung lượng Bộ nhớ đỉnh (MB)', color=color, fontsize=12)
    line3 = ax2.plot(Ns_vectorized, vectorized_memories, marker='^', linestyle='-.', color=color, label='Bộ nhớ RAM tiêu tốn (MB)', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Ghép tất cả các label vào chung một legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    plt.title('Phân tích Sự tăng trưởng của Vectorized Multi-Head Attention', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    plot2_path = os.path.join(plot_dir, 'vectorized_scaling.png')
    plt.savefig(plot2_path, dpi=300)
    print(f"[Đồ thị] Đã lưu biểu đồ tăng trưởng tại: {plot2_path}")
    plt.close()

# =====================================================================
# 5. GENERATE DATA TABLES FOR REPORT (MARKDOWN & LATEX)
# =====================================================================

def print_tables(Ns_naive, naive_times, naive_memories, Ns_vectorized, vectorized_times, vectorized_memories):
    print("\n" + "="*80)
    print(" BẢNG DỮ LIỆU THỰC NGHIỆM ĐỂ CHO VÀO BÁO CÁO")
    print("="*80)
    
    # --- Markdown Table ---
    print("\n### 1. Dạng Markdown (Dành cho Github README.md)")
    print("| Độ dài câu (L) | Thời gian Naive (ms) | Bộ nhớ Naive (MB) | Thời gian Vectorized (ms) | Bộ nhớ Vectorized (MB) | Tốc độ tăng (Speedup) |")
    print("|:---|:---|:---|:---|:---|:---|")
    for i, N in enumerate(Ns_vectorized):
        # Naive chỉ chạy đến Ns_naive
        if N in Ns_naive:
            idx = Ns_naive.index(N)
            n_t = f"{naive_times[idx]:.2f}"
            n_m = f"{naive_memories[idx]:.4f}"
            speedup = f"{naive_times[idx] / vectorized_times[i]:.1f}x"
        else:
            n_t = "N/A"
            n_m = "N/A"
            speedup = "N/A"
        print(f"| {N:<14} | {n_t:<20} | {n_m:<17} | {vectorized_times[i]:<25.2f} | {vectorized_memories[i]:<21.4f} | {speedup:<21} |")
        
    # --- LaTeX Table ---
    print("\n### 2. Dạng LaTeX Table (Dán trực tiếp vào file .tex của báo cáo)")
    print(r"""\begin{table}[H]
    \centering
    \begin{tabular}{|r|r|r|r|r|c|}
        \hline
        \rowcolor{darkblue} \textcolor{white}{\textbf{Độ dài L}} & \textcolor{white}{\textbf{Time Naive (ms)}} & \textcolor{white}{\textbf{RAM Naive (MB)}} & \textcolor{white}{\textbf{Time Vector (ms)}} & \textcolor{white}{\textbf{RAM Vector (MB)}} & \textcolor{white}{\textbf{Tốc độ tăng}} \\
        \hline""")
    for i, N in enumerate(Ns_vectorized):
        if N in Ns_naive:
            idx = Ns_naive.index(N)
            n_t = f"{naive_times[idx]:.2f}"
            n_m = f"{naive_memories[idx]:.4f}"
            speedup = f"{naive_times[idx] / vectorized_times[i]:.1f}x"
        else:
            n_t = "--"
            n_m = "--"
            speedup = "--"
        print(f"        {N} & {n_t} & {n_m} & {vectorized_times[i]:.2f} & {vectorized_memories[i]:.4f} & {speedup} \\\\")
        print(f"        \\hline")
    print(r"""    \end{tabular}
    \caption{Bảng so sánh hiệu năng giữa hai phương pháp Attention}
    \label{tab:attention_comparison}
\end{table}""")


# =====================================================================
# MAIN RUNNER
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Naive vs Vectorized Attention")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Chạy bộ benchmark nhỏ để kiểm tra nhanh pipeline",
    )
    args = parser.parse_args()

    verify_correctness()
    Ns_naive, naive_times, naive_memories, Ns_vectorized, vectorized_times, vectorized_memories = run_benchmarks(quick=args.quick)
    plot_results(Ns_naive, naive_times, naive_memories, Ns_vectorized, vectorized_times, vectorized_memories)
    print_tables(Ns_naive, naive_times, naive_memories, Ns_vectorized, vectorized_times, vectorized_memories)
