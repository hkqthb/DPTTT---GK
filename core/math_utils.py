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
    # Bước 1: Trừ đi giá trị lớn nhất theo trục để tránh overflow
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    
    # Bước 2: Tính e^x cho mỗi phần tử
    exp_x = np.exp(x_shifted)
    
    # Bước 3: Chia cho tổng để chuẩn hóa thành xác suất
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    
    return exp_x / sum_exp_x


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
