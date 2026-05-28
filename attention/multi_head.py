"""
Task 3: Multi-Head Attention Coordinator
==========================================
Quản lý việc tách/gộp các Head và điều phối tính toán Attention.

- split_heads: Cắt ma trận lớn thành nhiều Head nhỏ
- concat_heads: Gộp kết quả các Head lại
- forward: Luồng chạy chính
"""

import numpy as np
from core.layers import LinearLayer
from .scaled_dot_product import scaled_dot_product_attention


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        """
        Khởi tạo kiến trúc Multi-Head
        :param d_model: Kích thước vector của một từ (ví dụ: 512)
        :param num_heads: Số lượng đầu chú ý (ví dụ: 8)
        """
        self.d_model = d_model
        self.num_heads = num_heads
        
        # d_k là kích thước mới sau khi chia nhỏ. Phải chia hết!
        # Ví dụ: 512 / 8 = 64
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"
        self.d_k = d_model // num_heads

        # Khởi tạo Linear Layer cuối cùng (Output Projection)
        # Sử dụng LinearLayer từ Task 1 với Xavier Initialization
        self.W_O = LinearLayer(d_model, d_model)

    def split_heads(self, x):
        """
        Cắt bánh: Thay đổi kích thước ma trận từ 3 chiều thành 4 chiều
        Input shape:  (batch_size, seq_length, d_model)
        Output shape: (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, _ = x.shape
        
        # Bước 1: Reshape cắt d_model thành (num_heads, d_k)
        # Shape mới: (batch_size, seq_length, num_heads, d_k)
        x_reshaped = np.reshape(x, (batch_size, seq_length, self.num_heads, self.d_k))
        
        # Bước 2: Hoán đổi vị trí chiều (Transpose)
        # Ta cần đưa num_heads (chiều số 2) lên trước seq_length (chiều số 1)
        # Tại sao? Để mỗi Head đứng riêng biệt, dễ tính toán ma trận ở Task 2
        # (0: batch, 1: seq, 2: heads, 3: d_k) ---> (0, 2, 1, 3)
        x_transposed = np.transpose(x_reshaped, (0, 2, 1, 3))
        
        return x_transposed

    def concat_heads(self, x):
        """
        Gộp bánh: Ráp các Head lại với nhau sau khi tính toán xong
        Input shape:  (batch_size, num_heads, seq_length, d_k)
        Output shape: (batch_size, seq_length, d_model)
        """
        batch_size, num_heads, seq_length, d_k = x.shape
        
        # Bước 1: Trả num_heads về đúng vị trí cũ
        # Từ (0, 2, 1, 3) quay lại (0, 2, 1, 3) để thành (batch_size, seq_length, num_heads, d_k)
        x_transposed = np.transpose(x, (0, 2, 1, 3))
        
        # Bước 2: Ép phẳng (num_heads, d_k) trở lại thành d_model
        x_concatenated = np.reshape(x_transposed, (batch_size, seq_length, self.d_model))
        
        return x_concatenated

    def forward(self, Q, K, V, mask=True):
        """
        Luồng chạy chính của Multi-Head Attention.
        
        :param Q, K, V: Các ma trận gốc nhận từ Task 1
                         shape: (batch_size, seq_length, d_model)
        :param mask: Có áp dụng Causal Masking hay không
        :return: Ma trận đầu ra, shape (batch_size, seq_length, d_model)
        """
        # 1. Cắt bánh cho Q, K, V
        Q_split = self.split_heads(Q)
        K_split = self.split_heads(K)
        V_split = self.split_heads(V)

        # 2. Đưa cho Task 2 xử lý (Tính Attention cho từng Head song song)
        # scaled_dot_product_attention hỗ trợ tính toán vectorized trên tất cả heads
        head_outputs, attention_weights = scaled_dot_product_attention(
            Q_split, K_split, V_split, mask=mask
        )

        # 3. Gộp bánh lại
        concat_output = self.concat_heads(head_outputs)

        # 4. Đẩy qua LinearLayer cuối cùng (Output Projection)
        final_output = self.W_O.forward(concat_output)

        return final_output
