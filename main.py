"""
Task 5: Main - Tích hợp hệ thống & Vòng lặp sinh từ (Autoregressive Generation)
==================================================================================
File thực thi chính, ghép toàn bộ pipeline:
    Text -> Tokenizer -> Embedding + Positional Encoding
         -> Multi-Head Self-Attention -> Output Logits -> Sinh từ tiếp theo

Luồng Autoregressive:
    "Xin" -> Dự đoán "chào" -> "Xin chào" -> Dự đoán "các" -> ...

Cơ chế sinh từ:
    Model sử dụng n-gram language model (xác suất chuyển tiếp token) được
    học từ corpus huấn luyện, kết hợp với output của Attention pipeline.
    Điều này đảm bảo kết quả sinh ra là văn bản tiếng Việt có nghĩa,
    đồng thời vẫn demo đầy đủ luồng xử lý qua Self-Attention.
"""

import os
import numpy as np
from data.tokenizer import Tokenizer
from core.layers import LinearLayer
from core.math_utils import stable_softmax, sinusoidal_positional_encoding
from attention.multi_head import MultiHeadAttention


class TransformerGenerator:
    """
    Lớp tích hợp toàn bộ pipeline: Tokenizer + Attention + Generation.
    
    Bao gồm:
    - Tokenizer (Task 5): Chuyển text <-> token IDs
    - Multi-Head Attention (Task 3 + Task 2): Tạo Q/K/V nội bộ và xử lý attention
    - N-gram LM: Học xác suất chuyển tiếp token từ corpus
    """
    
    def __init__(self, d_model=64, num_heads=4):
        """
        Khởi tạo hệ thống.
        
        :param d_model: Kích thước vector embedding
        :param num_heads: Số lượng attention heads
        """
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Task 5: Tokenizer
        self.tokenizer = Tokenizer(embed_dim=d_model)
        
        # Task 3: Multi-Head Attention hoàn chỉnh (bao gồm W_Q, W_K, W_V, W_O)
        self.mha = MultiHeadAttention(d_model, num_heads)
        
        # Output projection: từ d_model -> vocab_size (tạo sau khi build vocab)
        self.output_layer = None
        
        # N-gram Language Model: học xác suất chuyển tiếp từ corpus
        # - Trigram: P(từ_tiếp | 2_từ_trước) -> câu mạch lạc hơn
        # - Bigram:  P(từ_tiếp | 1_từ_trước) -> fallback khi trigram không có
        self.trigram_probs = {}  # dict: (token_id_1, token_id_2) -> np.array(vocab_size)
        self.bigram_probs = None  # np.array: (vocab_size, vocab_size)
        
        # Tập hợp ID các token đặc biệt cần lọc khi sinh
        self._special_token_ids = set()
    
    def build(self, training_texts):
        """
        Xây dựng từ điển, khởi tạo output layer, và học bigram từ corpus.
        
        :param training_texts: Danh sách văn bản để xây dựng vocab và học
        """
        # Bước 1: Xây dựng từ điển (Tokenizer)
        self.tokenizer.build_vocab(training_texts)
        
        # Bước 2: Khởi tạo output projection layer
        self.output_layer = LinearLayer(self.d_model, self.tokenizer.vocab_size)
        
        # Bước 3: Đánh dấu các token đặc biệt cần lọc khi sinh
        for special in [Tokenizer.PAD_TOKEN, Tokenizer.UNK_TOKEN, Tokenizer.BOS_TOKEN]:
            if special in self.tokenizer.token_to_id:
                self._special_token_ids.add(self.tokenizer.token_to_id[special])
        
        # Bước 4: Học n-gram language model từ corpus
        self._learn_ngrams(training_texts)
        
        print(f"[Generator] Đã khởi tạo: d_model={self.d_model}, "
              f"heads={self.num_heads}, vocab={self.tokenizer.vocab_size}")
    
    def _learn_ngrams(self, texts):
        """
        Học xác suất chuyển tiếp n-gram (bigram + trigram) từ corpus.
        
        - Bigram:  P(từ_C | từ_B) — nhìn 1 từ trước
        - Trigram: P(từ_C | từ_A, từ_B) — nhìn 2 từ trước (mạch lạc hơn)
        
        Ví dụ: "Xin chào các bạn"
          Bigram:  Xin->chào, chào->các, các->bạn
          Trigram: (Xin,chào)->các, (chào,các)->bạn
        
        :param texts: Danh sách văn bản huấn luyện
        """
        vocab_size = self.tokenizer.vocab_size
        
        # ── Bigram: Ma trận đếm (vocab_size x vocab_size) ──
        bigram_counts = np.zeros((vocab_size, vocab_size))
        
        # ── Trigram: Dict đếm {(id_A, id_B): np.array(vocab_size)} ──
        trigram_counts = {}
        
        for text in texts:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Đếm bigram (cặp liên tiếp)
            for i in range(len(token_ids) - 1):
                bigram_counts[token_ids[i]][token_ids[i + 1]] += 1
            
            # Đếm trigram (bộ 3 liên tiếp)
            for i in range(len(token_ids) - 2):
                key = (token_ids[i], token_ids[i + 1])
                if key not in trigram_counts:
                    trigram_counts[key] = np.zeros(vocab_size)
                trigram_counts[key][token_ids[i + 2]] += 1
        
        # ── Chuẩn hóa Bigram ──
        # Alpha nhỏ giúp bigram tập trung vào các token thật sự xuất hiện
        # trong corpus, thay vì dàn đều xác suất sang toàn bộ vocab
        alpha = 0.001  # Laplace smoothing rất nhỏ
        bigram_counts += alpha
        for special_id in self._special_token_ids:
            bigram_counts[:, special_id] = 0
        row_sums = bigram_counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        self.bigram_probs = bigram_counts / row_sums
        
        # ── Chuẩn hóa Trigram ──
        self.trigram_probs = {}
        for key, counts in trigram_counts.items():
            for special_id in self._special_token_ids:
                counts[special_id] = 0
            total = counts.sum()
            if total > 0:
                self.trigram_probs[key] = counts / total
        
        print(f"[N-gram] Đã học {len(self.trigram_probs)} trigram + "
              f"bigram từ {len(texts)} câu")
    
    def forward(self, text):
        """
        Chạy một lượt forward pass qua toàn bộ pipeline.
        
        Text -> Embedding -> Q,K,V -> Multi-Head Attention -> Logits
        
        :param text: Chuỗi văn bản đầu vào
        :return: logits, shape (1, seq_len, vocab_size)
        """
        # Task 5: Text -> Embedding
        X = self.tokenizer.text_to_embedding(text, add_special_tokens=False)

        # Positional Encoding: thêm thông tin thứ tự token trước Self-Attention
        position_encoding = sinusoidal_positional_encoding(X.shape[1], self.d_model)
        X = X + position_encoding[np.newaxis, :, :]

        # Không có padding trong single-sentence forward, nhưng truyền mask rõ ràng
        # để cùng API với batch/padding mask.
        padding_mask = np.ones((X.shape[0], X.shape[1]), dtype=bool)

        # Task 3 + Task 2: Multi-Head Self-Attention với causal mask
        attn_output = self.mha.forward(X, mask=padding_mask, causal=True)

        # Output: Chuyển thành logits cho từng token trong vocab
        logits = self.output_layer.forward(attn_output)
        
        return logits
    
    def load_trained_weights(self, weights_path="model_weights.npy"):
        """
        Nạp trọng số đã huấn luyện từ PyTorch vào mô hình NumPy.
        
        File trọng số được tạo bởi train_pytorch.py, chứa các ma trận:
        - embedding_matrix: Ma trận embedding cho Tokenizer
        - W_Q_W, W_Q_b: Trọng số phép chiếu Query
        - W_K_W, W_K_b: Trọng số phép chiếu Key
        - W_V_W, W_V_b: Trọng số phép chiếu Value
        - W_O_W, W_O_b: Trọng số Output Projection
        - output_layer_W, output_layer_b: Trọng số lớp chiếu đầu ra
        
        :param weights_path: Đường dẫn tới file .npy chứa trọng số
        :return: True nếu nạp thành công, False nếu không
        """
        # Tìm file trọng số: thử đường dẫn tuyệt đối, rồi thử cùng thư mục với script
        if not os.path.exists(weights_path):
            alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), weights_path)
            if os.path.exists(alt_path):
                weights_path = alt_path
            else:
                print(f"[Generator] Không tìm thấy file trọng số '{weights_path}', "
                      f"giữ nguyên trọng số ngẫu nhiên (Xavier init).")
                return False
        
        try:
            weights = np.load(weights_path, allow_pickle=True).item()
            
            # 1. Nạp ma trận Embedding
            self.tokenizer.embedding_matrix = weights["embedding_matrix"]
            
            # 2. Nạp trọng số Multi-Head Attention (W_Q, W_K, W_V, W_O)
            self.mha.W_Q.W = weights["W_Q_W"]
            self.mha.W_Q.b = weights["W_Q_b"]
            
            self.mha.W_K.W = weights["W_K_W"]
            self.mha.W_K.b = weights["W_K_b"]
            
            self.mha.W_V.W = weights["W_V_W"]
            self.mha.W_V.b = weights["W_V_b"]
            
            self.mha.W_O.W = weights["W_O_W"]
            self.mha.W_O.b = weights["W_O_b"]
            
            # 3. Nạp trọng số Output Layer
            if self.output_layer is not None:
                self.output_layer.W = weights["output_layer_W"]
                self.output_layer.b = weights["output_layer_b"]
            
            print(f"[Generator] ✅ Đã nạp trọng số đã huấn luyện từ '{weights_path}'")
            return True
            
        except Exception as e:
            print(f"[Generator] ❌ Lỗi khi nạp trọng số: {e}")
            return False
    
    def generate(self, seed_text, max_new_tokens=20, temperature=1.0,
                 blend_ratio=0.85, repetition_penalty=0.3):
        """
        Vòng lặp tự hồi quy (Autoregressive Generation).
        
        Quy trình mỗi bước:
        1. Đưa chuỗi hiện tại vào pipeline Attention (forward pass)
        2. Lấy logits tại vị trí cuối cùng -> xác suất từ model (p_model)
        3. Lấy xác suất n-gram dựa trên ngữ cảnh cuối (p_ngram)
        4. Kết hợp: p_final = blend_ratio * p_ngram + (1 - blend_ratio) * p_model
        5. Áp dụng repetition penalty (giảm xác suất từ đã sinh gần đây)
        6. Sampling token tiếp theo từ p_final
        7. Nối token mới vào chuỗi, lặp lại
        
        :param seed_text: Văn bản khởi đầu (ví dụ: "Xin")
        :param max_new_tokens: Số token tối đa cần sinh
        :param temperature: Độ "ngẫu nhiên" khi sampling (cao = đa dạng hơn)
        :param blend_ratio: Tỷ lệ pha trộn n-gram (0.0 = chỉ model, 1.0 = chỉ n-gram)
        :param repetition_penalty: Hệ số phạt lặp (0.0 = không phạt, 1.0 = chặn hoàn toàn)
        :return: Chuỗi văn bản đã sinh
        """
        if temperature <= 0:
            raise ValueError("temperature phải lớn hơn 0")
        blend_ratio = float(np.clip(blend_ratio, 0.0, 1.0))
        repetition_penalty = float(np.clip(repetition_penalty, 0.0, 1.0))

        current_text = seed_text
        generated_ids = []
        
        print(f"\n{'='*60}")
        print(f"  Autoregressive Generation")
        print(f"  Seed: \"{seed_text}\" | max_tokens={max_new_tokens} | "
              f"temp={temperature} | blend={blend_ratio}")
        print(f"{'='*60}")
        
        for step in range(max_new_tokens):
            # ---- Nhánh 1: Forward pass qua Attention pipeline ----
            logits = self.forward(current_text)
            next_logits = logits[0, -1, :]  # shape: (vocab_size,)
            next_logits = next_logits / temperature
            
            # Mask token đặc biệt: đặt logits = -inf để softmax cho ra 0
            for sid in self._special_token_ids:
                next_logits[sid] = -1e9
            
            p_model = stable_softmax(next_logits)
            
            # ---- Nhánh 2: Xác suất n-gram (trigram > bigram) ----
            # Lấy các token cuối trong chuỗi hiện tại
            if self.tokenizer.mode == "word":
                words = current_text.split()
            else:
                words = list(current_text)
            
            unk_id = self.tokenizer.token_to_id[Tokenizer.UNK_TOKEN]
            
            # Thử trigram trước (nhìn 2 từ cuối) -> mạch lạc hơn
            p_ngram = None
            if len(words) >= 2:
                id_a = self.tokenizer.token_to_id.get(words[-2], unk_id)
                id_b = self.tokenizer.token_to_id.get(words[-1], unk_id)
                trigram_key = (id_a, id_b)
                if trigram_key in self.trigram_probs:
                    p_ngram = self.trigram_probs[trigram_key]
            
            # Fallback: bigram (nhìn 1 từ cuối)
            if p_ngram is None:
                last_id = self.tokenizer.token_to_id.get(words[-1], unk_id)
                p_ngram = self.bigram_probs[last_id]
            
            # ---- Kết hợp n-gram với model output ----
            p_final = blend_ratio * p_ngram + (1 - blend_ratio) * p_model
            
            # ---- Repetition Penalty ----
            # Giảm xác suất của các token đã sinh gần đây (cửa sổ 15 token)
            # Mức phạt giảm dần theo khoảng cách: từ mới sinh bị phạt nặng hơn
            if repetition_penalty > 0 and len(generated_ids) > 0:
                window = generated_ids[-15:]  # cửa sổ 15 token gần nhất
                for i, past_id in enumerate(reversed(window)):
                    # Token gần nhất bị phạt nặng nhất, xa hơn thì nhẹ hơn
                    decay = repetition_penalty * (1.0 - i * 0.06)
                    decay = max(decay, 0.05)  # giữ mức phạt tối thiểu
                    p_final[past_id] *= (1.0 - decay)
            
            # Chuẩn hóa lại (đảm bảo tổng = 1)
            p_sum = p_final.sum()
            if p_sum > 0:
                p_final = p_final / p_sum
            else:
                p_final = p_model  # Fallback nếu n-gram không có dữ liệu
            
            # ---- Sampling ----
            next_token_id = np.random.choice(len(p_final), p=p_final)
            
            # Kiểm tra EOS
            eos_id = self.tokenizer.token_to_id.get(Tokenizer.EOS_TOKEN)
            if next_token_id == eos_id:
                print(f"  [Step {step+1}] Gặp <EOS>, dừng sinh.")
                break
            
            # Decode token và nối vào chuỗi
            next_token = self.tokenizer.id_to_token.get(next_token_id, "?")
            generated_ids.append(next_token_id)
            
            # Word-level: thêm khoảng trắng giữa các từ
            if self.tokenizer.mode == "word":
                current_text += " " + next_token
            else:
                current_text += next_token
            
            print(f"  [Step {step+1}] Sinh: '{next_token}' -> \"{current_text}\"")
        
        print(f"{'='*60}")
        print(f"  Kết quả: \"{current_text}\"")
        print(f"{'='*60}\n")
        
        return current_text


# ==========================================
# CHẠY CHÍNH
# ==========================================
if __name__ == "__main__":
    # Corpus tiếng Việt để xây dựng từ điển và học n-gram
    # Corpus càng lớn -> n-gram càng phong phú -> kết quả sinh càng tự nhiên
    training_texts = [
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
    
    # Khởi tạo, xây dựng vocab (word-level), và học n-gram
    generator = TransformerGenerator(d_model=64, num_heads=4)
    generator.build(training_texts)
    
    # Nạp trọng số đã huấn luyện (nếu có file model_weights.npy)
    generator.load_trained_weights("model_weights.npy")
    
    # ── Demo 1: Forward Pass ──
    print("\n" + "─" * 60)
    print("  DEMO 1: Forward Pass qua Attention Pipeline")
    print("─" * 60)
    logits = generator.forward("Xin chào")
    print(f"  Input:  'Xin chào'")
    print(f"  Output: logits shape = {logits.shape}")
    print(f"  (1 batch × {logits.shape[1]} tokens × {generator.tokenizer.vocab_size} vocab)")
    
    # ── Demo 2: Autoregressive Generation ──
    print("\n" + "─" * 60)
    print("  DEMO 2: Sinh văn bản tự hồi quy (Autoregressive)")
    print("─" * 60)
    
    seeds = ["Xin chào", "Tôi đang", "Chúc các bạn", "Học máy"]
    for seed in seeds:
        generator.generate(seed, max_new_tokens=15, temperature=0.3, blend_ratio=0.95)
    
    # ── Demo 3: Tokenizer ──
    print("─" * 60)
    print("  DEMO 3: Tokenizer (Encode / Decode)")
    print("─" * 60)
    demo_texts = ["Chào các bạn", "Xin chào thế giới", "Tôi là sinh viên"]
    for text in demo_texts:
        token_ids = generator.tokenizer.encode(text)
        decoded = generator.tokenizer.decode(token_ids)
        print(f"  '{text}'")
        print(f"    Encode: {token_ids}")
        print(f"    Decode: '{decoded}'")
        print()
