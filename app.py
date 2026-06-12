import os
import sys
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

# Thêm thư mục gốc vào path để import các package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import TransformerGenerator
from data.tokenizer import Tokenizer
from core.math_utils import stable_softmax
from attention.multi_head import MultiHeadAttention
from experiments.benchmark import naive_attention_core, vectorized_attention_core

app = FastAPI(title="Multi-Head Attention Visualizer API")

# Mảng training texts mặc định giống trong main.py
TRAINING_TEXTS = [
    # Chào hỏi
    "Xin chào các bạn", "Xin chào thế giới", "Xin chào tất cả mọi người",
    "Xin lỗi tôi đến muộn", "Chào buổi sáng các bạn", "Chào mừng bạn đến đây",
    "Chào buổi chiều mọi người", "Xin chào và chào mừng đến lớp học",
    "Rất vui được gặp các bạn", "Rất vui được làm quen với bạn",
    "Cảm ơn các bạn rất nhiều", "Cảm ơn thầy cô đã hướng dẫn",
    "Cảm ơn mọi người đã lắng nghe", "Xin phép được trình bày",
    "Chúc mọi người một ngày tốt lành", "Chúc các bạn học tốt và thành công",
    "Chúc bạn một ngày tốt lành", "Chúc các bạn luôn vui vẻ",
    "Chúc mọi người thành công", "Hẹn gặp lại các bạn vào tuần sau",
    # Bản thân & Sinh viên
    "Tôi đang học lập trình", "Tôi là sinh viên năm ba",
    "Tôi là sinh viên ngành công nghệ thông tin", "Tôi yêu Việt Nam",
    "Tôi thích học máy rất nhiều", "Tôi đang làm đồ án môn học",
    "Tôi đang làm đồ án phân tích thuật toán", "Tôi đang nghiên cứu về trí tuệ nhân tạo",
    "Tôi đang tìm hiểu về mô hình Transformer", "Tôi rất thích lập trình bằng Python",
    "Tôi muốn trở thành kỹ sư phần mềm giỏi", "Tôi đã hoàn thành bài tập về nhà",
    "Tôi cần ôn thi cuối kỳ môn này", "Sinh viên cần học tốt và chăm chỉ",
    "Sinh viên năm ba phải làm đồ án", "Sinh viên công nghệ thông tin rất giỏi",
    "Đời sinh viên rất vui và nhiều kỷ niệm", "Chúng ta cùng học nhé",
    "Chúng ta là bạn tốt", "Chúng ta cần chuẩn bị bài thuyết trình",
    "Chúng ta hãy cùng nhau cố gắng", "Các bạn có thể thấy kết quả rất rõ ràng",
    "Các bạn hãy xem ví dụ sau đây", "Tất cả mọi người đều có thể học lập trình",
    "Bạn có thể học lập trình dễ dàng",
    # Thời tiết
    "Hôm nay trời đẹp quá", "Hôm nay trời nắng đẹp", "Hôm nay tôi đi học",
    "Hôm nay là một ngày tốt lành", "Hôm nay chúng ta học bài mới",
    "Hôm nay tôi rất vui vì được gặp bạn", "Hôm nay lớp học rất sôi nổi",
    "Ngày mai chúng ta sẽ thi giữa kỳ", "Ngày mai tôi sẽ hoàn thành đồ án",
    "Buổi sáng hôm nay rất đẹp", "Buổi chiều chúng ta đi thư viện",
    "Cuối tuần tôi sẽ ôn bài", "Tối nay tôi sẽ viết code",
    "Một ngày tốt lành cho tất cả", "Một ngày mới bắt đầu với năng lượng tích cực",
    # Công nghệ & Lập trình
    "Python là ngôn ngữ lập trình tuyệt vời", "Python là ngôn ngữ lập trình phổ biến nhất hiện nay",
    "Python rất dễ học và rất mạnh mẽ", "Ngôn ngữ lập trình Python rất phổ biến",
    "Lập trình là kỹ năng quan trọng của thế kỷ", "Lập trình giúp giải quyết nhiều vấn đề thực tế",
    "Học lập trình rất thú vị và bổ ích", "Học lập trình cần kiên nhẫn và thực hành",
    "Thuật toán là nền tảng của khoa học máy tính", "Thuật toán sắp xếp và tìm kiếm rất quan trọng",
    "Cấu trúc dữ liệu và thuật toán là môn học cơ sở", "Phân tích độ phức tạp thuật toán là kỹ năng cần thiết",
    "Đồ án phân tích thuật toán rất thú vị", "Đồ án môn học rất quan trọng và cần thiết",
    "Đồ án này giúp hiểu rõ cơ chế Attention", "Đây là một ví dụ đơn giản nhưng hiệu quả",
    "Đây là đồ án phân tích thuật toán của nhóm", "Mã nguồn được viết bằng Python và NumPy",
    "NumPy giúp tính toán ma trận rất nhanh", "Kết quả rất tốt và chính xác",
    "Kết quả thực nghiệm cho thấy thuật toán hoạt động tốt", "Kết quả benchmark chứng minh hiệu quả của vectorization",
    "Mô hình này rất đơn giản nhưng hiệu quả", "Mô hình đã được kiểm thử kỹ lường",
    "Chúng tôi đã chạy thử nghiệm thành công", "Chương trình chạy ổn định và cho kết quả chính xác",
    "Hiệu suất tính toán được cải thiện đáng kể",
    # AI & Deep Learning
    "Transformer thay đổi thế giới trí tuệ nhân tạo",
    "Transformer là kiến trúc nền tảng của các mô hình ngôn ngữ lớn",
    "Transformer sử dụng cơ chế Self Attention để xử lý ngôn ngữ",
    "Attention là cơ chế quan trọng nhất trong Transformer",
    "Attention cho phép mô hình hiểu ngữ cảnh tốt hơn",
    "Self Attention giúp mỗi từ nhìn toàn bộ các từ khác",
    "Self Attention có độ phức tạp bậc hai theo chiều dài chuỗi",
    "Multi Head Attention chia thành nhiều đầu để học các pattern khác nhau",
    "Multi Head Attention là thành phần cốt lõi của Transformer",
    "Trí tuệ nhân tạo đang phát triển rất nhanh", "Trí tuệ nhân tạo thay đổi cuộc sống con người",
    "Trí tuệ nhân tạo được ứng dụng trong nhiều lĩnh vực", "Học máy là lĩnh vực rất thú vị",
    "Học máy là nhánh quan trọng của trí tuệ nhân tạo", "Học máy giúp máy tính học từ dữ liệu",
    "Học sâu là bước tiến lớn của học máy", "Mô hình ngôn ngữ lớn rất mạnh và thông minh",
    "Mô hình ngôn ngữ lớn có thể hiểu và sinh văn bản", "Mô hình ngôn ngữ lớn đang thay đổi thế giới",
    "Xử lý ngôn ngữ tự nhiên là lĩnh vực quan trọng", "Xử lý ngôn ngữ tự nhiên giúp máy hiểu tiếng người",
    "ChatGPT là ứng dụng nổi bật của mô hình ngôn ngữ lớn", "Dữ liệu là nhiên liệu của trí tuệ nhân tạo",
    "Ma trận Attention thể hiện mối quan hệ giữa các từ", "Softmax chuyển điểm số thành phân phối xác suất",
    "Embedding chuyển từ thành vector số để máy hiểu được", "Positional Encoding giúp mô hình biết thứ tự của từ",
    "Query Key Value là ba thành phần của Attention", "Causal Mask đảm bảo mỗi token chỉ nhìn về phía trước",
    "Vectorization giúp tính toán nhanh hơn hàng trăm lần",
    # Đồ án & Thuyết trình
    "Đồ án này trình bày về cơ chế Self Attention", "Đồ án gồm năm thành phần chính",
    "Nhóm chúng em xin trình bày đồ án giữa kỳ", "Nhóm đã hoàn thành tất cả các yêu cầu",
    "Nhóm đã kiểm thử kỹ lưỡng toàn bộ mã nguồn", "Chúng em xin cảm ơn thầy cô đã lắng nghe",
    "Phần tiếp theo là kết quả thực nghiệm", "Phần này trình bày về kiến trúc hệ thống",
    "Biểu đồ cho thấy thời gian tăng theo bậc hai", "Bảng so sánh cho thấy vectorized nhanh hơn nhiều",
    "Demo cho thấy pipeline hoạt động chính xác", "Kết luận là Self Attention có độ phức tạp bậc hai",
    "Hướng phát triển là cài đặt Flash Attention", "Mục tiêu của đồ án là phân tích độ phức tạp",
    "Báo cáo gồm mười slide chính", "Slide này trình bày công thức toán học",
    "Phần demo minh họa pipeline sinh văn bản", "Chúng ta có thể thấy kết quả rất rõ ràng",
    "Thực nghiệm chứng minh lý thuyết là đúng", "Cảm ơn thầy cô và các bạn đã lắng nghe",
    # Câu bổ sung
    "Thế giới đang thay đổi nhanh chóng", "Thế giới công nghệ luôn đổi mới",
    "Công nghệ thông tin là ngành rất có triển vọng", "Công nghệ đang thay đổi cách chúng ta sống",
    "Khoa học máy tính phát triển rất nhanh", "Khoa học và công nghệ là chìa khóa thành công",
    "Việt Nam đang phát triển mạnh về công nghệ", "Việt Nam có nhiều kỹ sư phần mềm giỏi",
    "Tương lai thuộc về trí tuệ nhân tạo", "Tương lai của công nghệ rất tươi sáng",
    "Nghiên cứu khoa học cần sự kiên nhẫn", "Nghiên cứu về Attention đang rất sôi nổi",
    "Dự án này rất có ý nghĩa thực tiễn", "Dự án giúp hiểu rõ hơn về thuật toán",
    "Giáo dục là nền tảng phát triển đất nước", "Sáng tạo và đổi mới là chìa khóa thành công",
    "Làm việc nhóm giúp hoàn thành dự án tốt hơn", "Làm việc chăm chỉ sẽ mang lại kết quả tốt",
    "Thực hành nhiều sẽ giúp bạn giỏi hơn", "Thực hành là cách tốt nhất để học lập trình",
    "Kiến thức là sức mạnh", "Kiến thức nền tảng rất quan trọng",
    "Mỗi ngày một tiến bộ hơn", "Mỗi dự án là một bài học quý giá",
    "Thành công đến từ sự nỗ lực không ngừng", "Thành công cần kiên nhẫn và quyết tâm",
    "Sự kiên nhẫn là chìa khóa của thành công", "Đam mê công nghệ giúp tôi tiến bộ mỗi ngày",
    "Đam mê và nỗ lực sẽ dẫn đến thành công", "Hãy luôn cố gắng và không bao giờ bỏ cuộc",
    "Hãy tin vào bản thân và nỗ lực hết mình", "Nỗ lực hôm nay sẽ tạo nên thành công ngày mai",
    "Mọi thứ đều bắt đầu từ những bước nhỏ", "Mọi người đều có thể thành công nếu cố gắng",
    "Đây là kết quả sau nhiều ngày làm việc", "Đây là thành quả của cả nhóm",
    "Chúng tôi rất tự hào về dự án này", "Chúng tôi hy vọng thầy cô hài lòng",
    "Xin chân thành cảm ơn tất cả mọi người", "Xin cảm ơn và hẹn gặp lại"
]

# State toàn cục
model_state = {
    "generator": None,
    "d_model": 64,
    "num_heads": 4
}

def init_generator(d_model=64, num_heads=4):
    gen = TransformerGenerator(d_model=d_model, num_heads=num_heads)
    gen.build(TRAINING_TEXTS)
    model_state["generator"] = gen
    model_state["d_model"] = d_model
    model_state["num_heads"] = num_heads
    return gen

# Khởi tạo lần đầu
init_generator()

class InitParams(BaseModel):
    d_model: int = 64
    num_heads: int = 4

class TextParams(BaseModel):
    text: str

class GenerateStepParams(BaseModel):
    current_text: str
    temperature: float = 0.3
    blend_ratio: float = 0.95
    repetition_penalty: float = 0.3

class BenchmarkParams(BaseModel):
    seq_lengths: List[int] = [10, 50, 100, 200, 400]

@app.post("/api/init")
def api_init(params: InitParams):
    try:
        init_generator(d_model=params.d_model, num_heads=params.num_heads)
        return {"status": "success", "vocab_size": model_state["generator"].tokenizer.vocab_size}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/tokenize")
def api_tokenize(params: TextParams):
    gen = model_state["generator"]
    if not gen:
        raise HTTPException(status_code=400, detail="Model is not initialized")
    token_ids = gen.tokenizer.encode(params.text, add_special_tokens=False)
    tokens = [gen.tokenizer.id_to_token[tid] for tid in token_ids]
    return {"tokens": tokens, "token_ids": token_ids}

@app.post("/api/attention")
def api_attention(params: TextParams):
    gen = model_state["generator"]
    if not gen:
        raise HTTPException(status_code=400, detail="Model is not initialized")
    
    text = params.text
    if gen.tokenizer.mode == "word" and not text.strip():
        return {"tokens": [], "attention_weights": []}

    # Pipeline logic forward (lấy attention_weights)
    X = gen.tokenizer.text_to_embedding(text, add_special_tokens=False)
    seq_len = X.shape[1]
    
    # Cộng positional encoding
    from core.math_utils import sinusoidal_positional_encoding
    position_encoding = sinusoidal_positional_encoding(seq_len, gen.d_model)
    X = X + position_encoding[np.newaxis, :, :]
    
    # Multi-head forward và lấy weights
    padding_mask = np.ones((X.shape[0], X.shape[1]), dtype=bool)
    _, attention_weights = gen.mha.forward(
        X, mask=padding_mask, causal=True, return_attention=True
    )
    
    # Trả về tokens
    token_ids = gen.tokenizer.encode(text, add_special_tokens=False)
    tokens = [gen.tokenizer.id_to_token[tid] for tid in token_ids]
    
    # attention_weights shape: (batch_size, num_heads, query_len, key_len)
    # Vì batch_size = 1, lấy index 0 và convert sang list
    weights_list = attention_weights[0].tolist() # shape (num_heads, seq_len, seq_len)

    return {
        "tokens": tokens,
        "attention_weights": weights_list
    }

@app.post("/api/generate_step")
def api_generate_step(params: GenerateStepParams):
    gen = model_state["generator"]
    if not gen:
        raise HTTPException(status_code=400, detail="Model is not initialized")
    
    current_text = params.current_text
    temperature = params.temperature
    blend_ratio = params.blend_ratio
    repetition_penalty = params.repetition_penalty

    # Chạy forward pass
    X = gen.tokenizer.text_to_embedding(current_text, add_special_tokens=False)
    seq_len = X.shape[1]
    
    from core.math_utils import sinusoidal_positional_encoding
    position_encoding = sinusoidal_positional_encoding(seq_len, gen.d_model)
    X = X + position_encoding[np.newaxis, :, :]
    
    padding_mask = np.ones((X.shape[0], X.shape[1]), dtype=bool)
    logits, attention_weights = gen.mha.forward(
        X, mask=padding_mask, causal=True, return_attention=True
    )
    logits = gen.output_layer.forward(logits)
    
    next_logits = logits[0, -1, :] / temperature
    for sid in gen._special_token_ids:
        next_logits[sid] = -1e9
        
    p_model = stable_softmax(next_logits)
    
    # N-gram
    if gen.tokenizer.mode == "word":
        words = current_text.split()
    else:
        words = list(current_text)
        
    unk_id = gen.tokenizer.token_to_id[Tokenizer.UNK_TOKEN]
    p_ngram = None
    if len(words) >= 2:
        id_a = gen.tokenizer.token_to_id.get(words[-2], unk_id)
        id_b = gen.tokenizer.token_to_id.get(words[-1], unk_id)
        trigram_key = (id_a, id_b)
        if trigram_key in gen.trigram_probs:
            p_ngram = gen.trigram_probs[trigram_key]
            
    if p_ngram is None and len(words) >= 1:
        last_id = gen.tokenizer.token_to_id.get(words[-1], unk_id)
        p_ngram = gen.bigram_probs[last_id]
        
    if p_ngram is None:
        p_ngram = np.ones_like(p_model) / len(p_model)
        
    p_final = blend_ratio * p_ngram + (1 - blend_ratio) * p_model
    
    # Phạt lặp (ở đây ta tự decode tokens đã sinh ra)
    # Lấy 5 tokens gần nhất từ current_text
    token_ids = gen.tokenizer.encode(current_text, add_special_tokens=False)
    if repetition_penalty > 0 and len(token_ids) > 0:
        window = token_ids[-5:]
        for i, past_id in enumerate(reversed(window)):
            penalty = repetition_penalty / (i + 1)
            p_final[past_id] *= (1.0 - penalty)
            
    p_final = p_final / (p_final.sum() + 1e-12)
    
    # Top 10 tokens ứng viên
    top_indices = np.argsort(p_final)[::-1][:10]
    top_candidates = []
    for idx in top_indices:
        token_str = gen.tokenizer.id_to_token[idx]
        top_candidates.append({
            "token": token_str,
            "prob": float(p_final[idx]),
            "p_model": float(p_model[idx]),
            "p_ngram": float(p_ngram[idx])
        })
        
    # Sample token tiếp theo
    next_token_id = np.random.choice(len(p_final), p=p_final)
    next_token = gen.tokenizer.id_to_token[next_token_id]
    
    # Nối từ
    new_text = current_text
    if gen.tokenizer.mode == "word":
        new_text += " " + next_token
    else:
        new_text += next_token
        
    # Trả về kết quả
    weights_list = attention_weights[0].tolist() # (num_heads, seq_len, seq_len)
    tokens = [gen.tokenizer.id_to_token[tid] for tid in token_ids]
    
    return {
        "next_token": next_token,
        "new_text": new_text,
        "top_candidates": top_candidates,
        "tokens": tokens,
        "attention_weights": weights_list
    }

@app.post("/api/benchmark")
def api_benchmark(params: BenchmarkParams):
    d_model = model_state["d_model"]
    num_heads = model_state["num_heads"]
    d_k = d_model // num_heads
    batch_size = 1
    
    results = []
    
    for seq_len in params.seq_lengths:
        # Generate random Q, K, V
        Q = np.random.randn(batch_size, num_heads, seq_len, d_k)
        K = np.random.randn(batch_size, num_heads, seq_len, d_k)
        V = np.random.randn(batch_size, num_heads, seq_len, d_k)
        
        # Benchmark vectorized
        start_t = time.perf_counter()
        _ = vectorized_attention_core(Q, K, V, mask=None, causal=True)
        time_vectorized = time.perf_counter() - start_t
        
        # Benchmark naive (giới hạn seq_len lớn vì naive cực kỳ chậm)
        time_naive = None
        if seq_len <= 150: # Naive O(L^2) trên Python list quá lớn sẽ bị timeout
            start_t = time.perf_counter()
            _ = naive_attention_core(Q, K, V, mask=None, causal=True)
            time_naive = time.perf_counter() - start_t
            
        results.append({
            "seq_len": seq_len,
            "vectorized_time_ms": round(time_vectorized * 1000, 3),
            "naive_time_ms": round(time_naive * 1000, 3) if time_naive is not None else None
        })
        
    return {"results": results}

# Router phục vụ file tĩnh của frontend
@app.get("/", response_class=HTMLResponse)
def get_index():
    # Chúng ta sẽ đọc trực tiếp từ thư mục ui/index.html
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "index.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h3>UI files are loading... Please refresh.</h3>")

# Mount thư mục tĩnh `ui` cho stylesheet và javascript
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")), name="static")
