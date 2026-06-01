import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_k = Q.size(-1)

    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Normalize scores into attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Apply attention weights to values
    attn_output = torch.matmul(attn_weights, V)
    return attn_output