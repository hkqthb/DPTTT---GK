import numpy as np

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

        # Khởi tạo Linear Layer cuối cùng (Task 1 sẽ làm phần chi tiết này, 
        # ở đây mình tạo sẵn ma trận W_O ngẫu nhiên để test)
        self.W_O = np.random.randn(d_model, d_model) * 0.01

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

    def forward(self, Q, K, V, attention_core_function):
        """
        Luồng chạy chính của Task 3.
        :param Q, K, V: Các ma trận gốc nhận từ Task 1
        :param attention_core_function: Hàm tính điểm của Task 2 truyền vào
        """
        # 1. Cắt bánh cho Q, K, V
        Q_split = self.split_heads(Q)
        K_split = self.split_heads(K)
        V_split = self.split_heads(V)

        # 2. Đưa cho Task 2 xử lý (Tính Attention cho từng Head song song)
        # (Task 2 sẽ nhận input là 4 chiều và trả ra output 4 chiều y hệt)
        head_outputs = attention_core_function(Q_split, K_split, V_split)

        # 3. Gộp bánh lại
        concat_output = self.concat_heads(head_outputs)

        # 4. Nhân với ma trận W_O cuối cùng để ra kết quả
        final_output = concat_output @ self.W_O

        return final_output
    
# ==========================================
# TEST KHỐI CODE TASK 3 CỦA BẠN
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