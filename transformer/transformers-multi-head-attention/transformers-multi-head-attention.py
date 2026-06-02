import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Linear projections
    Q_proj = Q @ W_q  # (batch_size, seq_len, d_model)
    K_proj = K @ W_k
    V_proj = V @ W_v

    # Split into heads
    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    # Shapes: (batch_size, num_heads, seq_len, d_k)

    # Scaled dot-product attention
    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    # (batch_size, num_heads, seq_len, seq_len)

    attention_weights = softmax(scores, axis=-1)

    head_outputs = np.matmul(attention_weights, V_heads)
    # (batch_size, num_heads, seq_len, d_k)

    # Concatenate heads
    concat = head_outputs.transpose(0, 2, 1, 3).reshape(
        batch_size, seq_len, d_model
    )

    # Final output projection
    output = concat @ W_o

    return output