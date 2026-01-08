"""Implementing a Transformer from scratch in Jax"""

import jax
import jax.numpy as jnp
import haiku as hk

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

class MultiHeadAttention(hk.module):
    """Multi-head attention module."""

    def __init__(
        self,
        num_heads:int,
        d_model: int,
        name: Optional[str]
    ):
        """
        Args:
            num_heads: Number of attention heads
            d_model: Model dimension (must be divisible by num_heads)
        """
        super().__init__(name=name)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input (batch, sew_len, d_model)

        Returns:
            Output (batch, seq_len, d_model)
        """

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Initialise weight matrices
        Q = hk.Linear(self.d_model)(x)
        K = hk.Linear(self.d_model)(x)
        V = hk.Linear(self.d_model)(x)

        # Split into multiple heads
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k )
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k )
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k )

        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Apply attention
        attn_output = scaled_dot_product_attention(Q, K, V)
        
        # Concatenate heads
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)

        # Multiply by final weight matrix
        output = hk.Linear(self.d_model, name='output')(attn_output)

        return output

class FeedForward(hk.Module):
    def __init__(self, d_model, d_ff, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.d_ff = d_ff
