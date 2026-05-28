import numpy as np


def softmax(x, axis=-1):
    """
    Tính Softmax ổn định về mặt số học (Numerically Stable Softmax).

    Công thức chuẩn: softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Vấn đề: Khi x rất lớn (ví dụ 1000), exp(x) tràn số thành inf (overflow).
    Giải pháp: Trừ đi max(x) trước khi tính exp. Kết quả toán học không đổi vì:
        softmax(x - c) = softmax(x) với mọi hằng số c.

    :param x:    Ma trận đầu vào, bất kỳ kích thước nào.
    :param axis: Chiều để chuẩn hóa (mặc định: chiều cuối cùng).
    :return:     Ma trận xác suất cùng kích thước x, tổng theo `axis` bằng 1.
    """
    # Trừ max theo `axis`, giữ nguyên số chiều để broadcasting hoạt động đúng
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


class LinearLayer:
    """
    Lớp chiếu tuyến tính không có bias: output = X @ W

    Khởi tạo W bằng Xavier/Glorot Initialization thay vì random thuần túy.
    Mục tiêu: Giữ cho phương sai của các activation ổn định khi đi qua nhiều
    lớp liên tiếp, tránh gradient bị triệt tiêu (vanishing) hoặc bùng nổ (exploding).

    Công thức Xavier: W ~ N(0, sqrt(2 / (fan_in + fan_out)))
    """

    def __init__(self, in_features, out_features):
        """
        :param in_features:  Số chiều của vector đầu vào (fan_in).
        :param out_features: Số chiều của vector đầu ra (fan_out).
        """
        # Scale theo Xavier: căn bậc hai của 2 chia cho tổng số chiều vào và ra
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features) * scale

    def forward(self, x):
        """
        Thực hiện phép chiếu tuyến tính.

        :param x: Ma trận đầu vào, shape (..., in_features).
                  Hỗ trợ batch và chuỗi: (batch, seq_len, in_features).
        :return:  Ma trận đầu ra,  shape (..., out_features).
        """
        return x @ self.W
