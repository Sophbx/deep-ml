import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    
    return (Q, K, V)

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    scores = Q @ np.transpose(K)
    scaledScores = scores / np.sqrt(Q.shape[-1])
    # Subtract each row's maximum to prevent exponential overflow.
    scaledScores = scaledScores - np.max(scaledScores, axis=-1, keepdims=True)
    
    softmaxA = np.exp(scaledScores) / np.sum(np.exp(scaledScores), axis=-1, keepdims=True)
    output = softmaxA @ V

    return output    

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    # Case check
    seq_len, d_model = Q.shape

    if n_heads <= 0 or d_model % n_heads != 0:
        raise ValueError("n_heads must be positive and evenly divide d_model.")
    
    d_head = d_model // n_heads

    # Divide each query/key/value into n_heads of d_head length vectors
    Q = Q.reshape(seq_len, n_heads, d_head)
    K = K.reshape(seq_len, n_heads, d_head)
    V = V.reshape(seq_len, n_heads, d_head)

    output_h = []

    for i in range (n_heads):
        # We take multiple queries in Q but refering to one head together
        # Think: How can we reuse self_attention?
        Q_h = Q[:, i, :]
        K_h = K[:, i, :]
        V_h = V[:, i, :]

        output_h.append(self_attention(Q_h, K_h, V_h))

    return np.concatenate(output_h, axis=-1)