import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    x = np.asarray(x)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    W3 = np.asarray(W3)

    # Shortcut path
    if Ws is None:
        shortcut = x
    else:
        shortcut = x @ np.asarray(Ws)

    # Main path
    h = np.maximum(0, x @ W1)      
    z1 = np.maximum(0, h @ W2)
    z2 = z1 @ W3

    # Residual addition + ReLU
    y = np.maximum(0, z2 + shortcut)
    return y