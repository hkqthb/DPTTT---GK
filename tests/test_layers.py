import numpy as np
from src.math_utils import softmax, LinearLayer

# ==========================================
# TEST KHỐI CODE TASK 1: LAYER BASE & MATH
# ==========================================
if __name__ == "__main__":
    print("=" * 55)
    print("KIỂM TRA TASK 1: LAYER BASE & MATH UTILITIES")
    print("=" * 55)

    # ----------------------------------------------------------
    # Test 1: Softmax cơ bản — tổng mỗi hàng phải bằng 1
    # ----------------------------------------------------------
    print("\n[Test 1] Softmax - Tổng các hàng phải bằng 1")
    x = np.random.randn(4, 6)
    result = softmax(x, axis=-1)
    row_sums = result.sum(axis=-1)
    print(f"  Input shape:      {x.shape}")
    print(f"  Output shape:     {result.shape}")
    print(f"  Tổng từng hàng:   {np.round(row_sums, 6)}")
    assert np.allclose(row_sums, 1.0), "FAILED: Tổng hàng không bằng 1!"
    print("  -> PASSED")

    # ----------------------------------------------------------
    # Test 2: Softmax chống tràn số với giá trị rất lớn
    # Không dùng stable softmax: exp(1000) = inf -> NaN
    # ----------------------------------------------------------
    print("\n[Test 2] Softmax - Chống overflow với giá trị lớn")
    x_large = np.array([[1000.0, 1001.0, 999.0]])
    result_large = softmax(x_large, axis=-1)
    print(f"  Input:  {x_large}")
    print(f"  Output: {np.round(result_large, 4)}")
    assert not np.any(np.isnan(result_large)), "FAILED: Kết quả bị NaN!"
    assert np.allclose(result_large.sum(), 1.0), "FAILED: Tổng không bằng 1!"
    print("  -> PASSED")

    # ----------------------------------------------------------
    # Test 3: LinearLayer — kiểm tra shape đầu ra
    # ----------------------------------------------------------
    print("\n[Test 3] LinearLayer - Kiểm tra shape đầu ra")
    batch_size, seq_len, d_in, d_out = 2, 4, 8, 3
    layer = LinearLayer(in_features=d_in, out_features=d_out)
    x = np.random.randn(batch_size, seq_len, d_in)
    output = layer.forward(x)
    print(f"  Input shape:   {x.shape}")
    print(f"  Weight shape:  {layer.W.shape}")
    print(f"  Output shape:  {output.shape}")
    expected = (batch_size, seq_len, d_out)
    assert output.shape == expected, f"FAILED: Mong đợi {expected}, nhận {output.shape}"
    print("  -> PASSED")

    # ----------------------------------------------------------
    # Test 4: LinearLayer — Xavier initialization
    # Độ lệch chuẩn của W phải xấp xỉ sqrt(2 / (in + out))
    # ----------------------------------------------------------
    print("\n[Test 4] LinearLayer - Xavier initialization")
    d_in, d_out = 512, 64
    layer_large = LinearLayer(in_features=d_in, out_features=d_out)
    expected_scale = np.sqrt(2.0 / (d_in + d_out))
    actual_std = np.std(layer_large.W)
    print(f"  Expected std (Xavier): {expected_scale:.4f}")
    print(f"  Actual std of W:       {actual_std:.4f}")
    # Cho phép sai số 30% do tính ngẫu nhiên
    assert abs(actual_std - expected_scale) < expected_scale * 0.3, \
        "FAILED: Trọng số lệch quá nhiều so với Xavier!"
    print("  -> PASSED")

    print("\n" + "=" * 55)
    print("TẤT CẢ TESTS ĐÃ QUA!")
    print("=" * 55)
