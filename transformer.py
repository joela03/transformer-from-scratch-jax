"""Implementing a Transformer from scratch in Jax"""

import jax
import jax.numpy as jnp

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Args:
        q: queries (bach, seq_len, d_model)
        k: keys (bach, seq_len, d_model)
        v: values (bach, seq_len, d_model)
        mask: Description
    Returns:
        output (batch, seq_len, d_model)
    """

    # Get dim of the key vectors
    d_k = q.shape[-1]

    # Calculate scaled dot product of keys and queries
    scores = jax.matmul(q, k.transpose(0, 2, 1)) / jnp.sqrt(len(k))

    if mask:
        scores = jnp.where(mask == 0, scores, -1e9)

    # Apply softmax to get attention weights
    attention_weights= jax.nn.softmax(scores, -1)

    # Multiply by values to get weighted sum
    output = jnp.matmul(attention_weights, v)

    return output
    
