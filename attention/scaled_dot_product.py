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


def _expand_attention_mask(mask, target_shape):
    """
    Chuẩn hóa mask về shape có thể broadcast với scores.

    Quy ước: True = vị trí được phép attention, False = vị trí bị che.
    Hỗ trợ các shape phổ biến cho cả 3D (batch_size, query_len, key_len) 
    và 4D (batch_size, num_heads, query_len, key_len):
    - (batch_size, key_len): padding mask cho key/value
    - (query_len, key_len): mask chung cho mọi batch/head
    - (batch_size, query_len, key_len): mask riêng cho từng batch
    """
    mask = np.asarray(mask, dtype=bool)
    
    if len(target_shape) == 4:
        batch_size, _, query_len, key_len = target_shape
        if mask.ndim == 2 and mask.shape == (batch_size, key_len):
            mask = mask[:, np.newaxis, np.newaxis, :]
        elif mask.ndim == 2 and mask.shape == (query_len, key_len):
            mask = mask[np.newaxis, np.newaxis, :, :]
        elif mask.ndim == 3 and mask.shape == (batch_size, query_len, key_len):
            mask = mask[:, np.newaxis, :, :]
    elif len(target_shape) == 3:
        batch_size, query_len, key_len = target_shape
        if mask.ndim == 2 and mask.shape == (batch_size, key_len):
            mask = mask[:, np.newaxis, :]
        elif mask.ndim == 2 and mask.shape == (query_len, key_len):
            mask = mask[np.newaxis, :, :]
    
    try:
        return np.broadcast_to(mask, target_shape)
    except ValueError as exc:
        raise ValueError(
            f"mask shape {mask.shape} không broadcast được với scores shape {target_shape}"
        ) from exc


def scaled_dot_product_attention(Q, K, V, mask=None, causal=False):
    """
    Tính Scaled Dot-Product Attention.
    
    Công thức: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    :param Q: Query matrix, shape (..., seq_len, d_k)
    :param K: Key matrix, shape (..., key_len, d_k)
    :param V: Value matrix, shape (..., key_len, d_v)
    :param mask: Optional attention/padding mask. True = được attention.
                 Để tương thích phiên bản cũ, mask=True tương đương causal=True.
    :param causal: Nếu True, áp dụng Causal Masking (che tương lai)
    :return: tuple (attention_output, attention_weights)
             - attention_output: shape (..., query_len, d_v)
             - attention_weights: shape (..., query_len, key_len)
    """
    if Q.shape[:-2] != K.shape[:-2] or K.shape[:-2] != V.shape[:-2]:
        raise ValueError("Q, K, V phải có cùng các chiều batch/head")
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Chiều d_k của Q và K phải bằng nhau")
    if K.shape[-2] != V.shape[-2]:
        raise ValueError("K và V phải có cùng key/value length")

    # Giữ tương thích với API cũ: mask=True nghĩa là causal mask.
    if isinstance(mask, (bool, np.bool_)):
        causal = causal or bool(mask)
        mask = None

    # Lấy d_k từ chiều cuối cùng của K
    d_k = K.shape[-1]
    
    # Bước 1: Tính QK^T / sqrt(d_k)
    # Q shape: (..., seq_len, d_k)
    # K^T shape: (..., d_k, seq_len)
    # Kết quả shape: (..., seq_len, seq_len)
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)
    
    # Bước 2: Áp dụng Causal Masking và/hoặc Padding Mask nếu được yêu cầu
    valid_mask = None
    if causal:
        query_len, key_len = scores.shape[-2], scores.shape[-1]
        causal_mask = np.tril(np.ones((query_len, key_len), dtype=bool))
        valid_mask = causal_mask

    if mask is not None:
        expanded_mask = _expand_attention_mask(mask, scores.shape)
        valid_mask = expanded_mask if valid_mask is None else (valid_mask & expanded_mask)

    if valid_mask is not None:
        scores = np.where(valid_mask, scores, -np.inf)
    
    # Bước 3: Áp dụng Softmax để chuyển thành xác suất
    attention_weights = stable_softmax(scores, axis=-1)
    
    # Bước 4: Nhân với V để lấy output
    attention_output = np.matmul(attention_weights, V)
    
    return attention_output, attention_weights
