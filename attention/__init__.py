# Attention module: Logic cốt lõi của Attention (Task 2 & 3)
# - scaled_dot_product_attention: Tính toán Attention Score & Masking
# - MultiHeadAttention: Tách/gộp Head

from .scaled_dot_product import scaled_dot_product_attention
from .multi_head import MultiHeadAttention
