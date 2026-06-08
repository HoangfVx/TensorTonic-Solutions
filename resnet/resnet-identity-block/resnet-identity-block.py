import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    x = np.asarray(x)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    x_ = x.copy()
    h = np.maximum(0, x @ W1.T)
    y = np.maximum(0, h @ W2.T + x_)
    return y
    
