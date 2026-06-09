import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    grad = x.copy()

    for J in gradients_F:
        grad = (J + np.eye(J.shape[0])).T @ grad

    return grad

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    grad = x.copy()

    for J in gradients_F:
        grad = J.T @ grad

    return grad
        