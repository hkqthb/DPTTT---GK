"""
Task 2: Scaled Dot-Product Attention Engine
=============================================
Trái tim của mô hình - triển khai công thức:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Bao gồm:
- Tính Attention Score
- Causal Masking (che tương lai)
"""

import numpy as np
from core.math_utils import stable_softmax


def scaled_dot_product_attention(Q, K, V, mask=True):
    """
    Tính Scaled Dot-Product Attention.
    
    Công thức: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    :param Q: Query matrix, shape (..., seq_len, d_k)
    :param K: Key matrix, shape (..., seq_len, d_k)
    :param V: Value matrix, shape (..., seq_len, d_k)
    :param mask: Nếu True, áp dụng Causal Masking (che tương lai)
    :return: tuple (attention_output, attention_weights)
             - attention_output: shape (..., seq_len, d_k)
             - attention_weights: shape (..., seq_len, seq_len)
    """
    # Lấy d_k từ chiều cuối cùng của K
    d_k = K.shape[-1]
    
    # Bước 1: Tính QK^T / sqrt(d_k)
    # Q shape: (..., seq_len, d_k)
    # K^T shape: (..., d_k, seq_len)
    # Kết quả shape: (..., seq_len, seq_len)
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)
    
    # Bước 2: Áp dụng Causal Masking nếu được yêu cầu
    if mask:
        seq_len = scores.shape[-1]
        # Tạo mặt nạ tam giác dưới: vị trí hợp lệ = 1, vị trí bị che = 0
        causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        # Biến vị trí bị che thành -inf (vì e^(-inf) = 0 sau softmax)
        scores = np.where(causal_mask, scores, -np.inf)
    
    # Bước 3: Áp dụng Softmax để chuyển thành xác suất
    attention_weights = stable_softmax(scores, axis=-1)
    
    # Bước 4: Nhân với V để lấy output
    attention_output = np.matmul(attention_weights, V)
    
    return attention_output, attention_weights
