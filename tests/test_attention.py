import unittest

import numpy as np

from attention.multi_head import MultiHeadAttention
from attention.scaled_dot_product import scaled_dot_product_attention
from core.math_utils import sinusoidal_positional_encoding
from data.tokenizer import Tokenizer


class ScaledDotProductAttentionTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_causal_mask_blocks_future_tokens(self):
        Q = np.random.randn(1, 2, 5, 4).astype(np.float32)
        K = np.random.randn(1, 2, 5, 4).astype(np.float32)
        V = np.random.randn(1, 2, 5, 4).astype(np.float32)

        output, weights = scaled_dot_product_attention(Q, K, V, causal=True)

        self.assertEqual(output.shape, (1, 2, 5, 4))
        self.assertEqual(weights.shape, (1, 2, 5, 5))
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.triu(weights[0, 0], k=1), 0.0, atol=1e-7)

    def test_padding_mask_blocks_padded_keys(self):
        Q = np.random.randn(1, 2, 4, 4).astype(np.float32)
        K = np.random.randn(1, 2, 4, 4).astype(np.float32)
        V = np.random.randn(1, 2, 4, 4).astype(np.float32)
        padding_mask = np.array([[True, True, False, False]])

        _, weights = scaled_dot_product_attention(Q, K, V, mask=padding_mask)

        np.testing.assert_allclose(weights[..., 2:], 0.0, atol=1e-7)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)

    def test_all_masked_rows_return_zero_distribution(self):
        Q = np.random.randn(1, 1, 3, 4).astype(np.float32)
        K = np.random.randn(1, 1, 3, 4).astype(np.float32)
        V = np.random.randn(1, 1, 3, 4).astype(np.float32)
        padding_mask = np.array([[False, False, False]])

        output, weights = scaled_dot_product_attention(Q, K, V, mask=padding_mask)

        np.testing.assert_allclose(weights, 0.0, atol=1e-7)
        np.testing.assert_allclose(output, 0.0, atol=1e-7)


class MultiHeadAttentionTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_split_and_concat_are_inverse_operations(self):
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        x = np.random.randn(3, 4, 8).astype(np.float32)

        split = mha.split_heads(x)
        merged = mha.concat_heads(split)

        self.assertEqual(split.shape, (3, 2, 4, 4))
        np.testing.assert_allclose(merged, x)

    def test_forward_self_attention_returns_expected_shapes(self):
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        x = np.random.randn(2, 5, 8).astype(np.float32)
        padding_mask = np.array(
            [
                [True, True, True, True, True],
                [True, True, True, False, False],
            ]
        )

        output, weights = mha.forward(
            x,
            mask=padding_mask,
            causal=True,
            return_attention=True,
        )

        self.assertEqual(output.shape, (2, 5, 8))
        self.assertEqual(weights.shape, (2, 2, 5, 5))
        np.testing.assert_allclose(weights[1, :, :, 3:], 0.0, atol=1e-7)


class UtilityTest(unittest.TestCase):
    def test_positional_encoding_shape_and_determinism(self):
        first = sinusoidal_positional_encoding(seq_len=6, d_model=7)
        second = sinusoidal_positional_encoding(seq_len=6, d_model=7)

        self.assertEqual(first.shape, (6, 7))
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(first[0, 0::2], 0.0, atol=1e-7)

    def test_tokenizer_batch_encode_can_return_padding_mask(self):
        tokenizer = Tokenizer(embed_dim=4)
        tokenizer.build_vocab(["Xin chào", "Xin chào các bạn"])

        embeddings, mask = tokenizer.batch_encode(
            ["Xin chào", "Xin chào các bạn"],
            return_mask=True,
        )

        self.assertEqual(embeddings.shape[:2], mask.shape)
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, -1])
        self.assertTrue(mask[1, -1])


if __name__ == "__main__":
    unittest.main()
