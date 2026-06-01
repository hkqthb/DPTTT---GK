"""
Task 1: Math Utilities - Softmax & Xavier Initialization
=========================================================
Module chứa các hàm toán học nền tảng cho toàn bộ dự án.
"""

import numpy as np


def stable_softmax(x, axis=-1):
    """
    Tính Softmax ổn định số học (Numerically Stable Softmax).
    
    Khi tính lũy thừa e^x, các số lớn rất dễ gây tràn bộ nhớ (overflow).
    Giải pháp: Trừ đi giá trị max của mảng trước khi tính toán.
    
    Công thức: softmax(x_i) = e^(x_i - max(x)) / Σ e^(x_j - max(x))
    
    :param x: Ma trận đầu vào (numpy array)
    :param axis: Trục để tính softmax (mặc định: trục cuối cùng)
    :return: Ma trận sau khi áp dụng softmax, tổng mỗi hàng = 1
    """
    # Bước 1: Trừ đi giá trị lớn nhất theo trục để tránh overflow.
    # Nếu một hàng chỉ gồm -inf (ví dụ sau masking), giữ kết quả là 0 thay vì NaN.
    x_max = np.max(x, axis=axis, keepdims=True)
    safe_x_max = np.where(np.isfinite(x_max), x_max, 0)
    x_shifted = np.where(np.isfinite(x_max), x - safe_x_max, -np.inf)
    
    # Bước 2: Tính e^x cho mỗi phần tử
    exp_x = np.exp(x_shifted)
    
    # Bước 3: Chia cho tổng để chuẩn hóa thành xác suất.
    # Khi toàn bộ phần tử là -inf, tổng mũ bằng 0 và phân phối trả về toàn 0.
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return np.divide(exp_x, sum_exp_x, out=np.zeros_like(exp_x), where=sum_exp_x != 0)


def sinusoidal_positional_encoding(seq_len, d_model):
    """
    Tạo positional encoding dạng sin/cos như Transformer gốc.

    Self-Attention chỉ nhìn quan hệ giữa các vector, nên nếu không cộng thông tin
    vị trí thì mô hình không phân biệt được thứ tự token trong câu.

    :param seq_len: Độ dài chuỗi token
    :param d_model: Kích thước vector embedding
    :return: Ma trận positional encoding, shape (seq_len, d_model)
    """
    if seq_len < 0:
        raise ValueError("seq_len phải không âm")
    if d_model <= 0:
        raise ValueError("d_model phải lớn hơn 0")

    positions = np.arange(seq_len)[:, np.newaxis]
    div_terms = np.exp(
        np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
    )

    encoding = np.zeros((seq_len, d_model), dtype=np.float32)
    encoding[:, 0::2] = np.sin(positions * div_terms)
    if d_model > 1:
        encoding[:, 1::2] = np.cos(positions * div_terms[: encoding[:, 1::2].shape[1]])

    return encoding


def xavier_init(fan_in, fan_out):
    """
    Khởi tạo trọng số Xavier/Glorot.
    
    Trọng số không được random ngẫu nhiên hoàn toàn mà phải dựa trên 
    phương sai của kích thước input/output để tránh hiện tượng mất mát 
    đạo hàm (vanishing gradient).
    
    Phương sai: Var(W) = 2 / (fan_in + fan_out)
    
    :param fan_in: Số lượng neuron đầu vào
    :param fan_out: Số lượng neuron đầu ra
    :return: Ma trận trọng số (fan_in, fan_out)
    """
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std
