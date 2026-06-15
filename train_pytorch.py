"""
Huấn luyện mô hình Transformer bằng PyTorch, sau đó xuất trọng số sang NumPy.

Mục đích:
    Mô hình NumPy trong main.py chỉ có forward pass (không có backpropagation),
    nên các trọng số luôn là ngẫu nhiên → Attention phân bố đều, vô nghĩa.

    Script này giải quyết vấn đề bằng cách:
    1. Xây dựng mô hình PyTorch có kiến trúc GIỐNG HỆT mô hình NumPy
    2. Huấn luyện trên corpus tiếng Việt (autoregressive language modeling)
    3. Xuất trọng số đã train thành file .npy
    4. Mô hình NumPy nạp trọng số này → Attention có ý nghĩa ngữ nghĩa

Lưu ý quan trọng về chuyển đổi trọng số:
    - PyTorch Linear: y = x @ W^T + b  (W shape: output_dim × input_dim)
    - NumPy LinearLayer: y = x @ W + b  (W shape: input_dim × output_dim)
    → Phải TRANSPOSE ma trận W khi xuất từ PyTorch sang NumPy.
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Thêm thư mục gốc vào path để import được Tokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.tokenizer import Tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# CẤU HÌNH HUẤN LUYỆN (Hyperparameters)
# ══════════════════════════════════════════════════════════════════════════════
D_MODEL = 64        # Phải trùng với d_model trong main.py
NUM_HEADS = 4       # Phải trùng với num_heads trong main.py
EPOCHS = 300        # Số vòng huấn luyện (tăng vì corpus nhỏ)
LR = 0.003          # Learning rate
BATCH_SIZE = 16     # Kích thước mini-batch
WEIGHT_DECAY = 0.01 # Regularization
WARMUP_STEPS = 50   # Warmup cho learning rate scheduler


# ══════════════════════════════════════════════════════════════════════════════
# MÔ HÌNH PYTORCH (kiến trúc tương thích 1:1 với NumPy)
# ══════════════════════════════════════════════════════════════════════════════
class PyTorchTransformerLM(nn.Module):
    """
    Mô hình Transformer Language Model tối giản.
    
    Kiến trúc giống hệt TransformerGenerator trong main.py:
    - Embedding Layer
    - Sinusoidal Positional Encoding (giống nhau, không cần train)
    - Single-layer Multi-Head Self-Attention (có Causal Mask)
    - Output Projection → vocab logits
    """
    
    def __init__(self, vocab_size, d_model, num_heads, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Embedding (sẽ xuất sang tokenizer.embedding_matrix)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Multi-Head Attention projections (sẽ xuất sang mha.W_Q/K/V/O)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Output projection (sẽ xuất sang output_layer)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Pre-compute positional encoding (deterministic, giống NumPy)
        self.register_buffer('pe', self._sinusoidal_pe(max_seq_len, d_model))
    
    @staticmethod
    def _sinusoidal_pe(max_len, d_model):
        """Tạo Sinusoidal Positional Encoding (giống hàm trong math_utils.py)."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) 
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        return pe
    
    def forward(self, x):
        """
        Forward pass.
        
        :param x: Token IDs, shape (batch_size, seq_len)
        :return: logits, shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.size()
        
        # 1. Embedding + Positional Encoding
        emb = self.embedding(x)  # (batch, seq_len, d_model)
        x_enc = emb + self.pe[:seq_len].unsqueeze(0)
        
        # 2. Project Q, K, V
        Q = self.q_proj(x_enc)
        K = self.k_proj(x_enc)
        V = self.v_proj(x_enc)
        
        # 3. Split Heads: (batch, seq, d_model) → (batch, heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 4. Scaled Dot-Product Attention + Causal Mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Causal mask: che các token tương lai (upper triangle)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        # 5. Concat Heads: (batch, heads, seq, d_k) → (batch, seq, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 6. Output Projection
        out = self.out_proj(context)
        logits = self.output_layer(out)
        
        return logits


# ══════════════════════════════════════════════════════════════════════════════
# DỮ LIỆU HUẤN LUYỆN (lấy từ main.py)
# ══════════════════════════════════════════════════════════════════════════════
# Import trực tiếp training_texts sẽ chạy main.py → không muốn thế.
# Copy danh sách corpus ở đây để script chạy độc lập.

def get_training_texts():
    """Trả về corpus huấn luyện tiếng Việt (giống hệt trong main.py)."""
    return [
        "Xin chào các bạn",
        "Xin chào thế giới",
        "Xin chào tất cả mọi người",
        "Xin lỗi tôi đến muộn",
        "Chào buổi sáng các bạn",
        "Chào mừng bạn đến đây",
        "Chào buổi chiều mọi người",
        "Xin chào và chào mừng đến lớp học",
        "Rất vui được gặp các bạn",
        "Rất vui được làm quen với bạn",
        "Cảm ơn các bạn rất nhiều",
        "Cảm ơn thầy cô đã hướng dẫn",
        "Cảm ơn mọi người đã lắng nghe",
        "Xin phép được trình bày",
        "Chúc mọi người một ngày tốt lành",
        "Chúc các bạn học tốt và thành công",
        "Chúc bạn một ngày tốt lành",
        "Chúc các bạn luôn vui vẻ",
        "Chúc mọi người thành công",
        "Hẹn gặp lại các bạn vào tuần sau",
        "Tôi đang học lập trình",
        "Tôi là sinh viên năm ba",
        "Tôi là sinh viên ngành công nghệ thông tin",
        "Tôi yêu Việt Nam",
        "Tôi thích học máy rất nhiều",
        "Tôi đang làm đồ án môn học",
        "Tôi đang làm đồ án phân tích thuật toán",
        "Tôi đang nghiên cứu về trí tuệ nhân tạo",
        "Tôi đang tìm hiểu về mô hình Transformer",
        "Tôi rất thích lập trình bằng Python",
        "Tôi muốn trở thành kỹ sư phần mềm giỏi",
        "Tôi đã hoàn thành bài tập về nhà",
        "Tôi cần ôn thi cuối kỳ môn này",
        "Sinh viên cần học tốt và chăm chỉ",
        "Sinh viên năm ba phải làm đồ án",
        "Sinh viên công nghệ thông tin rất giỏi",
        "Đời sinh viên rất vui và nhiều kỷ niệm",
        "Chúng ta cùng học nhé",
        "Chúng ta là bạn tốt",
        "Chúng ta cần chuẩn bị bài thuyết trình",
        "Chúng ta hãy cùng nhau cố gắng",
        "Các bạn có thể thấy kết quả rất rõ ràng",
        "Các bạn hãy xem ví dụ sau đây",
        "Tất cả mọi người đều có thể học lập trình",
        "Bạn có thể học lập trình dễ dàng",
        "Hôm nay trời đẹp quá",
        "Hôm nay trời nắng đẹp",
        "Hôm nay tôi đi học",
        "Hôm nay là một ngày tốt lành",
        "Hôm nay chúng ta học bài mới",
        "Hôm nay tôi rất vui vì được gặp bạn",
        "Hôm nay lớp học rất sôi nổi",
        "Ngày mai chúng ta sẽ thi giữa kỳ",
        "Ngày mai tôi sẽ hoàn thành đồ án",
        "Buổi sáng hôm nay rất đẹp",
        "Buổi chiều chúng ta đi thư viện",
        "Cuối tuần tôi sẽ ôn bài",
        "Tối nay tôi sẽ viết code",
        "Một ngày tốt lành cho tất cả",
        "Một ngày mới bắt đầu với năng lượng tích cực",
        "Python là ngôn ngữ lập trình tuyệt vời",
        "Python là ngôn ngữ lập trình phổ biến nhất hiện nay",
        "Python rất dễ học và rất mạnh mẽ",
        "Ngôn ngữ lập trình Python rất phổ biến",
        "Lập trình là kỹ năng quan trọng của thế kỷ",
        "Lập trình giúp giải quyết nhiều vấn đề thực tế",
        "Học lập trình rất thú vị và bổ ích",
        "Học lập trình cần kiên nhẫn và thực hành",
        "Thuật toán là nền tảng của khoa học máy tính",
        "Thuật toán sắp xếp và tìm kiếm rất quan trọng",
        "Cấu trúc dữ liệu và thuật toán là môn học cơ sở",
        "Phân tích độ phức tạp thuật toán là kỹ năng cần thiết",
        "Đồ án phân tích thuật toán rất thú vị",
        "Đồ án môn học rất quan trọng và cần thiết",
        "Đồ án này giúp hiểu rõ cơ chế Attention",
        "Đây là một ví dụ đơn giản nhưng hiệu quả",
        "Đây là đồ án phân tích thuật toán của nhóm",
        "Mã nguồn được viết bằng Python và NumPy",
        "NumPy giúp tính toán ma trận rất nhanh",
        "Kết quả rất tốt và chính xác",
        "Kết quả thực nghiệm cho thấy thuật toán hoạt động tốt",
        "Kết quả benchmark chứng minh hiệu quả của vectorization",
        "Mô hình này rất đơn giản nhưng hiệu quả",
        "Mô hình đã được kiểm thử kỹ lưỡng",
        "Chúng tôi đã chạy thử nghiệm thành công",
        "Chương trình chạy ổn định và cho kết quả chính xác",
        "Hiệu suất tính toán được cải thiện đáng kể",
        "Bài toán này có độ phức tạp thời gian là bậc hai",
        "Độ phức tạp bậc hai là thách thức lớn nhất",
        "Tối ưu thuật toán là công việc rất quan trọng",
        "Transformer thay đổi thế giới trí tuệ nhân tạo",
        "Transformer là kiến trúc nền tảng của các mô hình ngôn ngữ lớn",
        "Transformer sử dụng cơ chế Self Attention để xử lý ngôn ngữ",
        "Attention là cơ chế quan trọng nhất trong Transformer",
        "Attention cho phép mô hình hiểu ngữ cảnh tốt hơn",
        "Self Attention giúp mỗi từ nhìn toàn bộ các từ khác",
        "Self Attention có độ phức tạp bậc hai theo chiều dài chuỗi",
        "Multi Head Attention chia thành nhiều đầu để học các pattern khác nhau",
        "Multi Head Attention là thành phần cốt lõi của Transformer",
        "Trí tuệ nhân tạo đang phát triển rất nhanh",
        "Trí tuệ nhân tạo thay đổi cuộc sống con người",
        "Trí tuệ nhân tạo được ứng dụng trong nhiều lĩnh vực",
        "Học máy là lĩnh vực rất thú vị",
        "Học máy là nhánh quan trọng của trí tuệ nhân tạo",
        "Học máy giúp máy tính học từ dữ liệu",
        "Học sâu là bước tiến lớn của học máy",
        "Mô hình ngôn ngữ lớn rất mạnh và thông minh",
        "Mô hình ngôn ngữ lớn có thể hiểu và sinh văn bản",
        "Mô hình ngôn ngữ lớn đang thay đổi thế giới",
        "Xử lý ngôn ngữ tự nhiên là lĩnh vực quan trọng",
        "Xử lý ngôn ngữ tự nhiên giúp máy hiểu tiếng người",
        "ChatGPT là ứng dụng nổi bật của mô hình ngôn ngữ lớn",
        "Dữ liệu là nhiên liệu của trí tuệ nhân tạo",
        "Ma trận Attention thể hiện mối quan hệ giữa các từ",
        "Softmax chuyển điểm số thành phân phối xác suất",
        "Embedding chuyển từ thành vector số để máy hiểu được",
        "Positional Encoding giúp mô hình biết thứ tự của từ",
        "Query Key Value là ba thành phần của Attention",
        "Causal Mask đảm bảo mỗi token chỉ nhìn về phía trước",
        "Vectorization giúp tính toán nhanh hơn hàng trăm lần",
        "Đồ án này trình bày về cơ chế Self Attention",
        "Đồ án gồm năm thành phần chính",
        "Nhóm chúng em xin trình bày đồ án giữa kỳ",
        "Nhóm đã hoàn thành tất cả các yêu cầu",
        "Nhóm đã kiểm thử kỹ lưỡng toàn bộ mã nguồn",
        "Chúng em xin cảm ơn thầy cô đã lắng nghe",
        "Phần tiếp theo là kết quả thực nghiệm",
        "Phần này trình bày về kiến trúc hệ thống",
        "Biểu đồ cho thấy thời gian tăng theo bậc hai",
        "Bảng so sánh cho thấy vectorized nhanh hơn nhiều",
        "Demo cho thấy pipeline hoạt động chính xác",
        "Kết luận là Self Attention có độ phức tạp bậc hai",
        "Hướng phát triển là cài đặt Flash Attention",
        "Mục tiêu của đồ án là phân tích độ phức tạp",
        "Báo cáo gồm mười slide chính",
        "Slide này trình bày công thức toán học",
        "Phần demo minh họa pipeline sinh văn bản",
        "Chúng ta có thể thấy kết quả rất rõ ràng",
        "Thực nghiệm chứng minh lý thuyết là đúng",
        "Cảm ơn thầy cô và các bạn đã lắng nghe",
        "Thế giới đang thay đổi nhanh chóng",
        "Thế giới công nghệ luôn đổi mới",
        "Công nghệ thông tin là ngành rất có triển vọng",
        "Công nghệ đang thay đổi cách chúng ta sống",
        "Khoa học máy tính phát triển rất nhanh",
        "Khoa học và công nghệ là chìa khóa thành công",
        "Việt Nam đang phát triển mạnh về công nghệ",
        "Việt Nam có nhiều kỹ sư phần mềm giỏi",
        "Tương lai thuộc về trí tuệ nhân tạo",
        "Tương lai của công nghệ rất tươi sáng",
        "Nghiên cứu khoa học cần sự kiên nhẫn",
        "Nghiên cứu về Attention đang rất sôi nổi",
        "Dự án này rất có ý nghĩa thực tiễn",
        "Dự án giúp hiểu rõ hơn về thuật toán",
        "Giáo dục là nền tảng phát triển đất nước",
        "Sáng tạo và đổi mới là chìa khóa thành công",
        "Làm việc nhóm giúp hoàn thành dự án tốt hơn",
        "Làm việc chăm chỉ sẽ mang lại kết quả tốt",
        "Thực hành nhiều sẽ giúp bạn giỏi hơn",
        "Thực hành là cách tốt nhất để học lập trình",
        "Kiến thức là sức mạnh",
        "Kiến thức nền tảng rất quan trọng",
        "Mỗi ngày một tiến bộ hơn",
        "Mỗi dự án là một bài học quý giá",
        "Thành công đến từ sự nỗ lực không ngừng",
        "Thành công cần kiên nhẫn và quyết tâm",
        "Sự kiên nhẫn là chìa khóa của thành công",
        "Đam mê công nghệ giúp tôi tiến bộ mỗi ngày",
        "Đam mê và nỗ lực sẽ dẫn đến thành công",
        "Hãy luôn cố gắng và không bao giờ bỏ cuộc",
        "Hãy tin vào bản thân và nỗ lực hết mình",
        "Nỗ lực hôm nay sẽ tạo nên thành công ngày mai",
        "Mọi thứ đều bắt đầu từ những bước nhỏ",
        "Mọi người đều có thể thành công nếu cố gắng",
        "Đây là kết quả sau nhiều ngày làm việc",
        "Đây là thành quả của cả nhóm",
        "Chúng tôi rất tự hào về dự án này",
        "Chúng tôi hy vọng thầy cô hài lòng",
        "Xin chân thành cảm ơn tất cả mọi người",
        "Xin cảm ơn và hẹn gặp lại",
        "Con chó là loài động vật rất trung thành",
        "Con mèo thích ngủ dưới ánh nắng mặt trời",
        "Tôi nuôi một con chó màu vàng rất đáng yêu",
        "Con mèo kêu meo meo đòi ăn cá",
        "Con chó sủa gâu gâu khi thấy người lạ",
        "Nuôi thú cưng giúp giảm bớt căng thẳng",
        "Con mèo thích chơi đùa với cuộn len",
        "Con chó thích chạy bộ cùng tôi mỗi sáng",
        "Chú mèo con lông trắng muốt rất tinh nghịch",
        "Tôi rất yêu thương con vật nuôi của mình",
        "Con chim hót líu lo trên cành cây",
        "Đàn cá bơi lội tung tăng dưới hồ nước",
        "Cuối tuần tôi thường đi uống trà sữa với bạn bè",
        "Tôi thích đọc sách khoa học vào ban đêm",
        "Nghe nhạc giúp tôi tập trung viết code hơn",
        "Chúng tôi đi xem phim chiếu rạp vào tối thứ bảy",
        "Chạy bộ mỗi ngày giúp nâng cao sức khỏe",
        "Tôi thích đi du lịch khắp đất nước Việt Nam",
        "Ăn cơm tối cùng gia đình rất ấm áp",
        "Tôi đang tập nấu ăn một số môn học mới",
        "Chụp ảnh phong cảnh là sở thích của tôi",
        "Hãy uống nhiều nước mỗi ngày để giữ sức khỏe",
        "Bóng đá là môn thể thao vua được yêu thích nhất",
        "Tôi thích ngắm hoàng hôn trên bãi biển",
        "Toán học và Vật lý là các môn học thú vị",
        "Học tiếng Anh giúp mở rộng cơ hội nghề nghiệp",
        "Thầy cô giáo luôn tận tâm truyền đạt kiến thức",
        "Trường đại học có khuôn viên rất rộng và đẹp",
        "Tôi cần vượt qua kỳ thi tiếng Anh tuần tới",
        "Học sinh cần làm bài tập đầy đủ trước khi lên lớp",
        "Thư viện trường có rất nhiều tài liệu quý giá",
        "Chúng tôi thảo luận nhóm rất tích cực trong giờ học",
        "Hoàn thành khóa học giúp tôi tự tin hơn",
        "Kiến thức lý thuyết cần đi đôi với thực hành",
        "Bạn nên ghi chép bài đầy đủ để dễ ôn tập",
        "Đăng ký môn học kỳ này rất cạnh tranh",
        "Gia đình là điểm tựa bình yên nhất của mỗi người",
        "Bố mẹ luôn là người ủng hộ mọi quyết định của tôi",
        "Tôi có một người anh trai rất thông minh",
        "Chị gái tôi nấu ăn cực kỳ ngon",
        "Hãy luôn trân trọng những người bạn chân thành",
        "Chúng tôi thường về thăm ông bà vào dịp Tết",
        "Chia sẻ khó khăn giúp tình bạn thêm bền chặt",
        "Mẹ tôi luôn chăm sóc gia đình rất chu đáo",
        "Bố tôi thích trồng cây và nuôi chim cảnh",
        "Gia đình tôi sum họp hạnh phúc bên mâm cơm",
        "Anh em trong nhà cần yêu thương và giúp đỡ nhau",
        "Tôi nhận được nhiều lời chúc tốt đẹp từ bạn bè",
        "Tôi mới mua một chiếc laptop cấu hình mạnh",
        "Điện thoại thông minh là vật bất ly thân ngày nay",
        "Xe máy là phương tiện phổ biến nhất ở Việt Nam",
        "Tôi đi học bằng xe đạp mỗi ngày để bảo vệ môi trường",
        "Ngồi trên xe buýt giúp tôi có thời gian đọc sách",
        "Lái ô tô đòi hỏi sự tập trung cao độ",
        "Chiếc bàn làm việc của tôi luôn được sắp xếp gọn gàng",
        "Quyển sách này chứa đựng nhiều bài học ý nghĩa",
        "Tôi cần sạc pin cho máy tính ngay lập tức",
        "Đồng hồ treo tường nhắc nhở tôi quản lý thời gian",
        "Tôi thích mang theo một cuốn sổ tay nhỏ",
        "Đèn bàn cung cấp ánh sáng tốt để học bài",
        "Tôi cảm thấy rất hạnh phúc khi đạt điểm cao",
        "Đừng lo lắng về những điều chưa xảy ra",
        "Sự kiên trì sẽ giúp bạn vượt qua mọi thử thách",
        "Tôi rất bất ngờ trước kết quả của cuộc thi",
        "Hãy luôn giữ tinh thần lạc quan trong cuộc sống",
        "Cảm giác hoàn thành công việc thật tuyệt vời",
        "Tôi cảm thấy tự hào về nỗ lực của bản thân",
        "Sự tự tin giúp bạn tỏa sáng trước đám đông",
        "Tôi luôn tò mò muốn khám phá những điều mới lạ",
        "Hãy học cách lắng nghe ý kiến của người khác",
        "Sự chân thành luôn chạm đến trái tim con người",
        "Tôi cảm thấy biết ơn vì những gì mình đang có",
        "Làm việc chăm chỉ là chìa khóa của thành công",
        "Tìm kiếm một công việc phù hợp không hề dễ dàng",
        "Kỹ năng giao tiếp rất quan trọng khi đi phỏng vấn",
        "Tôi muốn tích lũy thêm nhiều kinh nghiệm thực tế",
        "Đồng nghiệp ở công ty mới rất thân thiện",
        "Chúng tôi cùng nhau thảo luận kế hoạch dự án",
        "Viết báo cáo tuần là công việc bắt buộc",
        "Tôi hy vọng sẽ được thăng tiến trong tương lai",
        "Quản lý thời gian hiệu quả giúp giảm áp lực công việc",
        "Môi trường làm việc năng động giúp tôi phát triển",
        "Hãy chuẩn bị hồ sơ xin việc thật ấn tượng",
        "Tôi yêu thích công việc lập trình của mình",
        "Cơn mưa rào mùa hạ làm dịu đi cái nắng nóng",
        "Bầu trời đêm đầy sao lấp lánh cực kỳ lãng mạn",
        "Mùa thu lá vàng rơi khắp các con đường",
        "Mùa đông trời lạnh buốt khiến ai cũng muốn ở nhà",
        "Hoa mai hoa đào nở rộ báo hiệu mùa xuân về",
        "Không khí buổi sáng sớm ở quê rất trong lành",
        "Những ngọn núi trùng điệp hùng vĩ giữa mây trời",
        "Tiếng sóng biển rì rào mang lại cảm giác bình yên",
        "Ánh nắng ban mai ấm áp chiếu qua kẽ lá",
        "Cánh đồng lúa chín vàng óng trải dài vô tận",
        "Bảo vệ rừng là bảo vệ cuộc sống của chính chúng ta",
        "Thời tiết hôm nay se se lạnh rất dễ chịu",
        "Xin chào, bạn tên là gì?",
        "Hôm nay công việc của bạn thế nào?",
        "Bạn có muốn đi ăn trưa cùng tôi không?",
        "Cảm ơn bạn đã nhiệt tình giúp đỡ tôi",
        "Chúc mừng sinh nhật bạn thân yêu của tôi",
        "Mọi chuyện rồi sẽ ổn thôi, đừng quá lo lắng",
        "Thật tuyệt vời khi được đồng hành cùng bạn",
        "Tôi rất mong chờ chuyến đi sắp tới",
        "Hãy giữ liên lạc nhé, đừng quên tôi đấy",
        "Chúc bạn thượng lộ bình an và gặp nhiều may mắn",
        "Xin lỗi vì đã làm phiền bạn vào lúc này",
        "Không sao đâu, tôi rất vui lòng được hỗ trợ bạn",
        "Mỗi ngày là một cơ hội để bắt đầu lại",
        "Thất bại là mẹ của thành công, hãy tiếp tục cố gắng",
        "Hãy sống trọn vẹn từng khoảnh khắc của hiện tại",
        "Hành trình vạn dặm luôn bắt đầu từ một bước chân",
        "Ước mơ chỉ thành hiện thực khi bạn hành động",
        "Hãy tử tế với mọi người xung quanh bạn",
        "Sức mạnh lớn nhất nằm ở chính bên trong bạn",
        "Không có gì là không thể nếu bạn có quyết tâm",
        "Hãy học hỏi từ những sai lầm của quá khứ",
        "Cuộc sống là một bức tranh, hãy tự tô màu cho nó",
        "Hãy luôn mỉm cười và đón nhận mọi điều xảy ra",
        "Sự cho đi mang lại nhiều niềm vui hơn nhận lại",
    ]


def prepare_data(tokenizer, texts):
    """
    Chuẩn bị dữ liệu huấn luyện cho autoregressive language modeling.
    
    Mỗi câu "A B C D" → Input: [BOS, A, B, C] → Target: [A, B, C, D]
    Mô hình học dự đoán token tiếp theo tại mỗi vị trí.
    """
    input_seqs = []
    target_seqs = []
    
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)  # [BOS, ..., EOS]
        if len(ids) > 2:  # Cần ít nhất BOS + 1 token + EOS
            input_seqs.append(ids[:-1])   # [BOS, t1, t2, ..., tn]
            target_seqs.append(ids[1:])   # [t1, t2, ..., tn, EOS]
    
    # Padding tất cả sequences về cùng độ dài
    pad_id = tokenizer.token_to_id[Tokenizer.PAD_TOKEN]
    max_len = max(len(s) for s in input_seqs)
    
    X = [s + [pad_id] * (max_len - len(s)) for s in input_seqs]
    Y = [s + [pad_id] * (max_len - len(s)) for s in target_seqs]
    
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


def export_weights(model, output_path):
    """
    Xuất trọng số PyTorch sang định dạng NumPy (.npy).
    
    Chuyển đổi:
        PyTorch Linear: y = x @ W^T + b  →  W shape (out, in)
        NumPy LinearLayer: y = x @ W + b  →  W shape (in, out)
        → W_numpy = W_pytorch.T
    """
    weights = {
        # Embedding matrix
        "embedding_matrix": model.embedding.weight.detach().cpu().numpy(),
        
        # Multi-Head Attention: W_Q
        "W_Q_W": model.q_proj.weight.detach().cpu().numpy().T,   # (in, out)
        "W_Q_b": model.q_proj.bias.detach().cpu().numpy().reshape(1, -1),
        
        # Multi-Head Attention: W_K
        "W_K_W": model.k_proj.weight.detach().cpu().numpy().T,
        "W_K_b": model.k_proj.bias.detach().cpu().numpy().reshape(1, -1),
        
        # Multi-Head Attention: W_V
        "W_V_W": model.v_proj.weight.detach().cpu().numpy().T,
        "W_V_b": model.v_proj.bias.detach().cpu().numpy().reshape(1, -1),
        
        # Multi-Head Attention: W_O (Output Projection)
        "W_O_W": model.out_proj.weight.detach().cpu().numpy().T,
        "W_O_b": model.out_proj.bias.detach().cpu().numpy().reshape(1, -1),
        
        # Output Layer (vocab projection)
        "output_layer_W": model.output_layer.weight.detach().cpu().numpy().T,
        "output_layer_b": model.output_layer.bias.detach().cpu().numpy().reshape(1, -1),
    }
    
    np.save(output_path, weights)
    print(f"\n[Export] Đã lưu trọng số vào: {output_path}")
    print(f"  Các key: {list(weights.keys())}")
    for key, val in weights.items():
        print(f"    {key}: shape {val.shape}")


def train():
    """Hàm huấn luyện chính."""
    
    print("=" * 70)
    print("  HUẤN LUYỆN MÔ HÌNH TRANSFORMER BẰNG PYTORCH")
    print("  (Xuất trọng số sang NumPy để dùng trong demo)")
    print("=" * 70)
    
    # ── 1. Chuẩn bị Tokenizer và Dữ liệu ──
    training_texts = get_training_texts()
    
    tokenizer = Tokenizer(embed_dim=D_MODEL, mode="word")
    tokenizer.build_vocab(training_texts)
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.token_to_id[Tokenizer.PAD_TOKEN]
    
    X_train, Y_train = prepare_data(tokenizer, training_texts)
    print(f"\n[Data] Số câu: {len(training_texts)} | Vocab: {vocab_size} tokens")
    print(f"[Data] X shape: {X_train.shape} | Y shape: {Y_train.shape}")
    
    # ── 2. Khởi tạo Model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Sử dụng: {device}")
    
    model = PyTorchTransformerLM(vocab_size, D_MODEL, NUM_HEADS).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Tổng tham số: {total_params:,}")
    
    # ── 3. Optimizer & Loss ──
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LR, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler: warmup → cosine decay
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, EPOCHS * (len(X_train) // BATCH_SIZE + 1) - WARMUP_STEPS)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    # ── 4. Vòng lặp huấn luyện ──
    print(f"\n{'─' * 70}")
    print(f"  Bắt đầu huấn luyện: {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR}")
    print(f"{'─' * 70}\n")
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle dữ liệu mỗi epoch
        perm = torch.randperm(len(X_train))
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]
        
        for i in range(0, len(X_train), BATCH_SIZE):
            x_batch = X_shuffled[i:i+BATCH_SIZE].to(device)
            y_batch = Y_shuffled[i:i+BATCH_SIZE].to(device)
            
            optimizer.zero_grad()
            
            logits = model(x_batch)  # (batch, seq_len, vocab_size)
            
            # Flatten: (batch*seq_len, vocab_size) vs (batch*seq_len,)
            loss = criterion(
                logits.view(-1, vocab_size), 
                y_batch.view(-1)
            )
            
            loss.backward()
            
            # Gradient clipping để ổn định quá trình train
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # In progress mỗi 20 epoch hoặc epoch đầu/cuối
        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:>4}/{EPOCHS} │ Loss: {avg_loss:.4f} │ "
                  f"Best: {best_loss:.4f} │ LR: {current_lr:.6f}")
    
    print(f"\n{'─' * 70}")
    print(f"  Huấn luyện hoàn tất! Best Loss: {best_loss:.4f}")
    print(f"{'─' * 70}")
    
    # ── 5. Kiểm tra nhanh (Quick Validation) ──
    print(f"\n{'─' * 70}")
    print(f"  Kiểm tra nhanh: Sinh văn bản")
    print(f"{'─' * 70}")
    
    model.eval()
    test_prompts = ["Xin chào", "Tôi đang", "Học máy", "Chó màu vàng,"]
    
    with torch.no_grad():
        for prompt in test_prompts:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([ids], dtype=torch.long).to(device)
            
            generated = list(ids)
            for _ in range(12):
                logits = model(input_ids)
                next_logits = logits[0, -1, :] / 0.7  # temperature
                
                # Mask special tokens
                for special in [Tokenizer.PAD_TOKEN, Tokenizer.UNK_TOKEN, Tokenizer.BOS_TOKEN]:
                    if special in tokenizer.token_to_id:
                        next_logits[tokenizer.token_to_id[special]] = float('-inf')
                
                probs = F.softmax(next_logits, dim=-1)
                
                # Top-k sampling
                top_k = 10
                top_probs, top_indices = torch.topk(probs, top_k)
                top_probs = top_probs / top_probs.sum()
                idx = torch.multinomial(top_probs, 1)
                next_id = top_indices[idx].item()
                
                # Dừng nếu gặp EOS
                eos_id = tokenizer.token_to_id.get(Tokenizer.EOS_TOKEN)
                if next_id == eos_id:
                    break
                
                generated.append(next_id)
                input_ids = torch.tensor([generated], dtype=torch.long).to(device)
            
            result = " ".join(tokenizer.id_to_token.get(i, "?") for i in generated)
            print(f'  "{prompt}" → "{result}"')
    
    # ── 6. Xuất trọng số ──
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights.npy")
    export_weights(model, output_path)
    
    print(f"\n{'=' * 70}")
    print(f"  HOÀN TẤT! File trọng số: model_weights.npy")
    print(f"  Tiếp theo: Chạy main.py hoặc app.py, trọng số sẽ tự động được nạp.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    train()
