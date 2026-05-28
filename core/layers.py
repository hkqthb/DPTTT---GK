"""
Task 1: Linear Layer
=====================
Lớp tuyến tính cơ bản: y = x @ W + b
Sử dụng Xavier Initialization cho trọng số.
"""

import numpy as np
from .math_utils import xavier_init


class LinearLayer:
    """
    Lớp tuyến tính (Fully Connected Layer).
    
    Nhận input, nhân ma trận với trọng số W và cộng bias b.
    Trọng số được khởi tạo bằng phương pháp Xavier/Glorot.
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Khởi tạo LinearLayer.
        
        :param input_dim: Kích thước chiều đầu vào (fan_in)
        :param output_dim: Kích thước chiều đầu ra (fan_out)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Khởi tạo trọng số W bằng Xavier Initialization
        self.W = xavier_init(input_dim, output_dim)
        
        # Khởi tạo bias b = 0
        self.b = np.zeros((1, output_dim))
    
    def forward(self, x):
        """
        Tính toán y = x @ W + b
        
        :param x: Ma trận đầu vào, shape (..., input_dim)
        :return: Ma trận đầu ra, shape (..., output_dim)
        """
        return x @ self.W + self.b
