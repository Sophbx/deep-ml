import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	(seq_len, d_model) = Q.shape
	scores = Q @ K.T
	scores = scores / np.sqrt(d_model)

	# Your code here
	if mask is not None:
		mask = np.broadcast_to(mask, scores.shape)

		if np.any(np.all(mask==-np.inf, axis=-1)):
			raise ValueError("Query needs >= 1 allowed key.")
		
		scores = np.where(mask==-np.inf, -np.inf, scores)

	# stable softmax
	scores = scores - np.max(scores, axis=-1, keepdims=True)
	softmaxA = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

	output = softmaxA @ V

	return output