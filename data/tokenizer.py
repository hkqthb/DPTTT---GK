"""
Task 5: Tokenizer & Embedding
===============================
Chuyển đổi văn bản tiếng Việt thành Token IDs và ngược lại.
Tạo Input Embedding từ Token IDs.

Hỗ trợ 2 chế độ tokenize:
- Word-level (mặc định): Tách theo khoảng trắng -> mỗi từ là 1 token
- Character-level: Mỗi ký tự là 1 token

Luồng xử lý:
    Text -> Token IDs -> Embedding Vectors (Input cho model)
    Output Logits -> Token ID -> Text (kết quả sinh từ)
"""

import numpy as np


class Tokenizer:
    """
    Tokenizer đa chế độ cho tiếng Việt.
    
    - Word-level: "Xin chào bạn" -> ["Xin", "chào", "bạn"] -> [5, 6, 7]
    - Char-level: "Xin" -> ["X", "i", "n"] -> [5, 6, 7]
    
    Ánh xạ mỗi token thành một số nguyên (Token ID),
    sau đó tra bảng để biến số thành vector (Input Embedding).
    """
    
    # Các token đặc biệt
    PAD_TOKEN = "<PAD>"   # Padding - đệm cho câu ngắn
    UNK_TOKEN = "<UNK>"   # Unknown - token không biết
    BOS_TOKEN = "<BOS>"   # Beginning of Sequence - bắt đầu câu
    EOS_TOKEN = "<EOS>"   # End of Sequence - kết thúc câu
    
    def __init__(self, embed_dim=64, mode="word"):
        """
        Khởi tạo Tokenizer.
        
        :param embed_dim: Kích thước vector embedding cho mỗi token
        :param mode: Chế độ tokenize - "word" (theo từ) hoặc "char" (theo ký tự)
        """
        if mode not in {"word", "char"}:
            raise ValueError("mode phải là 'word' hoặc 'char'")

        self.embed_dim = embed_dim
        self.mode = mode
        
        # Từ điển ánh xạ: token -> token ID
        self.token_to_id = {}
        # Từ điển ngược: token ID -> token
        self.id_to_token = {}
        
        # Thêm các token đặc biệt trước
        self._add_special_tokens()
        
        # Ma trận embedding sẽ được tạo khi build vocab
        self.embedding_matrix = None
    
    def _add_special_tokens(self):
        """Thêm các token đặc biệt vào từ điển."""
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        for token in special_tokens:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def _tokenize(self, text):
        """
        Tách văn bản thành danh sách token tùy theo chế độ.
        
        :param text: Chuỗi văn bản
        :return: Danh sách các token (string)
        """
        if self.mode == "word":
            return text.split()  # Tách theo khoảng trắng
        else:
            return list(text)    # Tách theo ký tự
    
    def _detokenize(self, tokens):
        """
        Ghép danh sách token thành văn bản.
        
        :param tokens: Danh sách các token (string)
        :return: Chuỗi văn bản
        """
        if self.mode == "word":
            return " ".join(tokens)
        else:
            return "".join(tokens)
    
    def build_vocab(self, texts):
        """
        Xây dựng từ điển từ danh sách văn bản.
        
        Quét qua tất cả các token trong texts để tạo ánh xạ.
        Sau đó khởi tạo ma trận embedding.
        
        :param texts: Danh sách các chuỗi văn bản
        """
        for text in texts:
            for token in self._tokenize(text):
                if token not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token
        
        # Khởi tạo ma trận embedding ngẫu nhiên
        vocab_size = len(self.token_to_id)
        self.embedding_matrix = np.random.randn(vocab_size, self.embed_dim) * 0.01
        
        mode_name = "word-level" if self.mode == "word" else "char-level"
        print(f"[Tokenizer] Đã xây dựng từ điển ({mode_name}): {vocab_size} tokens, "
              f"embed_dim={self.embed_dim}")
    
    @property
    def vocab_size(self):
        """Trả về kích thước từ điển."""
        return len(self.token_to_id)
    
    def encode(self, text, add_special_tokens=True):
        """
        Chuyển đổi văn bản thành danh sách Token IDs.
        
        :param text: Chuỗi văn bản đầu vào
        :param add_special_tokens: Có thêm BOS/EOS hay không
        :return: Danh sách các Token IDs
        """
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.token_to_id[self.BOS_TOKEN])
        
        for token in self._tokenize(text):
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.token_to_id[self.UNK_TOKEN])
        
        if add_special_tokens:
            token_ids.append(self.token_to_id[self.EOS_TOKEN])
        
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Chuyển đổi danh sách Token IDs thành văn bản.
        
        :param token_ids: Danh sách các Token IDs
        :param skip_special_tokens: Bỏ qua các token đặc biệt
        :return: Chuỗi văn bản
        """
        special_ids = set()
        if skip_special_tokens:
            special_ids = {
                self.token_to_id[self.PAD_TOKEN],
                self.token_to_id[self.UNK_TOKEN],
                self.token_to_id[self.BOS_TOKEN],
                self.token_to_id[self.EOS_TOKEN],
            }
        
        tokens = []
        for tid in token_ids:
            if tid in special_ids:
                continue
            if tid in self.id_to_token:
                tokens.append(self.id_to_token[tid])
            else:
                tokens.append("?")
        
        return self._detokenize(tokens)
    
    def text_to_embedding(self, text, add_special_tokens=True):
        """
        Pipeline hoàn chỉnh: Text -> Token IDs -> Embedding Vectors.
        
        :param text: Chuỗi văn bản đầu vào
        :param add_special_tokens: Có thêm BOS/EOS hay không
        :return: Ma trận embedding, shape (1, seq_len, embed_dim)
                 (batch_size=1 cho 1 câu)
        """
        if self.embedding_matrix is None:
            raise RuntimeError("Chưa xây dựng từ điển! Gọi build_vocab() trước.")
        
        # Bước 1: Text -> Token IDs
        token_ids = self.encode(text, add_special_tokens=add_special_tokens)
        
        # Bước 2: Token IDs -> Embedding Vectors (tra bảng embedding)
        embeddings = self.embedding_matrix[token_ids]  # shape: (seq_len, embed_dim)
        
        # Bước 3: Thêm chiều batch_size = 1
        embeddings = np.expand_dims(embeddings, axis=0)  # shape: (1, seq_len, embed_dim)
        
        return embeddings
    
    def batch_encode(self, texts, max_length=None, add_special_tokens=True, return_mask=False):
        """
        Mã hóa nhiều câu thành một batch với padding.
        
        :param texts: Danh sách các chuỗi văn bản
        :param max_length: Chiều dài tối đa (None = tự động theo câu dài nhất)
        :param add_special_tokens: Có thêm BOS/EOS hay không
        :param return_mask: Nếu True, trả thêm padding mask, True = token thật
        :return: embeddings hoặc tuple (embeddings, padding_mask)
        """
        if self.embedding_matrix is None:
            raise RuntimeError("Chưa xây dựng từ điển! Gọi build_vocab() trước.")
        
        # Mã hóa tất cả các câu
        all_ids = [self.encode(text, add_special_tokens) for text in texts]
        
        # Xác định chiều dài tối đa
        if max_length is None:
            max_length = max(len(ids) for ids in all_ids)
        
        # Padding các câu ngắn
        pad_id = self.token_to_id[self.PAD_TOKEN]
        padded_ids = []
        padding_masks = []
        for ids in all_ids:
            if len(ids) < max_length:
                valid_len = len(ids)
                ids = ids + [pad_id] * (max_length - len(ids))
            else:
                ids = ids[:max_length]
                valid_len = max_length
            padded_ids.append(ids)
            padding_masks.append([True] * valid_len + [False] * (max_length - valid_len))
        
        # Chuyển thành numpy array và tra bảng embedding
        padded_ids = np.array(padded_ids)  # shape: (batch_size, max_len)
        embeddings = self.embedding_matrix[padded_ids]  # shape: (batch_size, max_len, embed_dim)
        
        if return_mask:
            return embeddings, np.array(padding_masks, dtype=bool)
        return embeddings
