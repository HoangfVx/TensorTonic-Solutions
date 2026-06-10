import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: np.ndarray of same shape as input with batch-normalized and skip-connected output
    """
    x, W1, W2, gamma1, beta1, gamma2, beta2 = map(np.asarray, [x, W1, W2, gamma1, beta1, gamma2, beta2])
    if mode == "post":
        z = relu(batch_norm(x @ W1, gamma1, beta1))
        return {"output": relu(batch_norm(z @ W2, gamma2, beta2) + x), "mode": mode}
    elif mode == "pre":
        z = relu(batch_norm(x, gamma1, beta1)) @ W1 
        return {"output": relu(batch_norm(z, gamma2, beta2)) @ W2 + x, "mode": mode}

def batch_norm(x, gamma, beta, eps=1e-5):
    x_hat = (x - x.mean(axis=0, keepdims=True)) / (x.var(axis=0, keepdims=True) + eps) ** 0.5

    return gamma * x_hat + beta 

def relu(x):
    return np.maximum(0, x)
