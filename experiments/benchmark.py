"""
Task 4: Empirical Profiling & Benchmarking
============================================
Đo lường độ phức tạp O(L^2) của Self-Attention.
So sánh vectorized NumPy vs naive for-loop.
"""

import time
import tracemalloc
import numpy as np
import os

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def run_benchmark(attention_fn, seq_lengths=None, d_model=64, num_heads=4):
    """
    Chạy benchmark đo thời gian và RAM theo độ dài sequence.

    :param attention_fn: Hàm attention cần đo
    :param seq_lengths: Danh sách các độ dài L cần thử
    :param d_model: Kích thước embedding
    :param num_heads: Số lượng head
    :return: dict với keys 'seq_lengths', 'times', 'memories'
    """
    if seq_lengths is None:
        seq_lengths = [10, 50, 100, 500, 1000]

    times = []
    memories = []

    for L in seq_lengths:
        Q = np.random.randn(1, L, d_model)
        K = np.random.randn(1, L, d_model)
        V = np.random.randn(1, L, d_model)

        tracemalloc.start()
        start = time.perf_counter()

        attention_fn(Q, K, V)

        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(elapsed)
        memories.append(peak / 1024)  # KB

        print(f"  L={L:5d} | Time={elapsed:.4f}s | Peak RAM={peak/1024:.1f} KB")

    return {"seq_lengths": seq_lengths, "times": times, "memories": memories}
