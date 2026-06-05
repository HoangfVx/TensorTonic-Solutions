import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    layer_norm = gamma * (x - np.mean(x, axis=-1, keepdims=True)) / np.sqrt(np.var(x, axis=-1, keepdims=True) + eps) + beta
    return layer_norm

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
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

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    z = np.dot(x, W1) + b1
    output = np.dot(np.maximum(0, z), W2) + b2
    return output

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    mha = multi_head_attention(Q=x, K=x, V=x, W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o, num_heads=num_heads)
    x_ = layer_norm(x=(x + mha), gamma=gamma1, beta=beta1)
    output = layer_norm(x=(x_+feed_forward(x_, W1, b1, W2, b2)), gamma=gamma2, beta=beta2)
    return output