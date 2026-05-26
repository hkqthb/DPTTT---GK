import numpy as np
from src.multi_head import MultiHeadAttention # Lấy class từ thư mục src sang

# ==========================================
# TEST KHỐI CODE TASK 3
# ==========================================
if __name__ == "__main__":
    # 1. Tạo thông số giả định
    batch_size = 2   # 2 câu
    seq_length = 4   # Mỗi câu 4 từ
    d_model = 8      # Vector 8 chiều
    num_heads = 2    # Chia làm 2 Heads
    
    # 2. Tạo ma trận Q, K, V giả lập (Giả vờ nhận từ Task 1)
    # Shape = (2, 4, 8)
    Q = np.random.rand(batch_size, seq_length, d_model)
    K = np.random.rand(batch_size, seq_length, d_model)
    V = np.random.rand(batch_size, seq_length, d_model)

    # 3. Tạo 1 hàm giả lập của Task 2 (chỉ trả về chính V để test)
    def mock_task2_attention(q, k, v):
        print(f"-> Bên trong Task 2 nhận được shape: {q.shape}") # Sẽ in ra (2, 2, 4, 4)
        return v 

    # 4. Khởi tạo class của bạn và chạy
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    print(f"Kích thước ban đầu của Q: {Q.shape}")
    output = mha.forward(Q, K, V, mock_task2_attention)
    print(f"Kích thước sau khi gộp và xuất ra: {output.shape}") # Phải quay về (2, 4, 8)