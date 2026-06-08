import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns: np.ndarray with sum of main path output and projected shortcut
    """
    x = np.asarray(x)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)

    # Shortcut path
    if Ws is None:
        shortcut = x
    else:
        shortcut = x @ np.asarray(Ws)

    # Main path
    h = np.maximum(0, x @ W1)      
    z = h @ W2

    # Residual addition + ReLU
    y = np.maximum(0, z + shortcut)

    return y
