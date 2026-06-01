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
        alpha = 0.1  # Laplace smoothing
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
    
    def generate(self, seed_text, max_new_tokens=20, temperature=1.0, blend_ratio=0.85):
        """
        Vòng lặp tự hồi quy (Autoregressive Generation).
        
        Quy trình mỗi bước:
        1. Đưa chuỗi hiện tại vào pipeline Attention (forward pass)
        2. Lấy logits tại vị trí cuối cùng -> xác suất từ model (p_model)
        3. Lấy xác suất n-gram dựa trên ngữ cảnh cuối (p_ngram)
        4. Kết hợp: p_final = blend_ratio * p_ngram + (1 - blend_ratio) * p_model
        5. Sampling token tiếp theo từ p_final
        6. Nối token mới vào chuỗi, lặp lại
        
        :param seed_text: Văn bản khởi đầu (ví dụ: "Xin")
        :param max_new_tokens: Số token tối đa cần sinh
        :param temperature: Độ "ngẫu nhiên" khi sampling (cao = đa dạng hơn)
        :param blend_ratio: Tỷ lệ pha trộn n-gram (0.0 = chỉ model, 1.0 = chỉ n-gram)
        :return: Chuỗi văn bản đã sinh
        """
        if temperature <= 0:
            raise ValueError("temperature phải lớn hơn 0")
        blend_ratio = float(np.clip(blend_ratio, 0.0, 1.0))

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
        # Chào hỏi
        "Xin chào các bạn",
        "Xin chào thế giới",
        "Xin chào tất cả mọi người",
        "Xin lỗi tôi đến muộn",
        "Chào buổi sáng các bạn",
        "Chào mừng bạn đến đây",
        # Thời tiết
        "Hôm nay trời đẹp quá",
        "Hôm nay tôi đi học",
        "Hôm nay là một ngày tốt lành",
        # Giới thiệu bản thân
        "Tôi đang học lập trình",
        "Tôi là sinh viên năm ba",
        "Tôi yêu Việt Nam",
        "Tôi thích học máy rất nhiều",
        "Tôi đang làm đồ án môn học",
        # Công nghệ
        "Python là ngôn ngữ lập trình tuyệt vời",
        "Transformer thay đổi thế giới trí tuệ nhân tạo",
        "Attention là cơ chế quan trọng nhất",
        "Học máy là lĩnh vực rất thú vị",
        "Trí tuệ nhân tạo đang phát triển nhanh",
        "Mô hình ngôn ngữ lớn rất mạnh",
        # Học tập
        "Chúng ta cùng học nhé",
        "Chúng ta là bạn tốt",
        "Học lập trình rất thú vị",
        "Sinh viên cần học tốt",
        "Đồ án môn học rất quan trọng",
        # Chúc mừng
        "Chúc bạn một ngày tốt lành",
        "Chúc các bạn học tốt",
        "Chúc mọi người thành công",
        # Mô tả
        "Đây là một ví dụ đơn giản",
        "Đây là đồ án phân tích thuật toán",
        "Mô hình này rất đơn giản nhưng hiệu quả",
        "Kết quả rất tốt và chính xác",
        # Thêm câu cho n-gram coverage
        "Các bạn có thể thấy kết quả",
        "Thế giới đang thay đổi nhanh chóng",
        "Ngôn ngữ lập trình Python rất phổ biến",
        "Một ngày tốt lành cho tất cả",
        "Bạn có thể học lập trình dễ dàng",
        "Rất vui được gặp các bạn",
        "Cảm ơn các bạn rất nhiều",
        "Tất cả mọi người đều có thể học",
    ]
    
    # Khởi tạo, xây dựng vocab (word-level), và học n-gram
    generator = TransformerGenerator(d_model=64, num_heads=4)
    generator.build(training_texts)
    
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
        generator.generate(seed, max_new_tokens=10, temperature=0.5, blend_ratio=0.9)
    
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
