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
        self.dropout_rate = dropout_rate
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
        x: Input (batch, seq_len, d_model)

        Returns:
            Output (batch, seq_len, d_model)
        """

        # First linear layer
        x1 = hk.Linear(self.d_ff, name='dense1')(x)

        # Apply Relu activation
        x2 = jax.nn.relu(x1)

        # Second linear layer
        output = hk.Linear(self.d_model, name='dense2')(x2)

        return output


def positional_encoding(seq_len: int, d_model: int) -> jnp.ndarray:
    """
    Generate positional encodings.
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
    
    Returns:
        Positional encodings (seq_len, d_model)
    """
    # Create positional indices
    pos = jnp.arange(seq_len)

    # Create dimensional indices for even positional
    i = jnp.arange(0, d_model, 2)

    # Calculate division term
    div_term = jnp.power(10000.0, i/ d_model)

    # Broadcast for matrix computation
    pos = pos[:, None]
    div_term = div_term[None, :]

    # Compute angles
    angle = pos / div_term  

    # Initialise positional encoding matrix
    pe = jnp.zeros((seq_len, d_model))

    # Apply sine to even indices
    pe = pe.at[:, 0::2].set(jnp.sin(angle))

    # Apply cos to odd indices
    pe = pe.at[:, 1::2].set(jnp.cos(angle))

    return pe
class EncoderBlock(hk.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        name: str | None = None,
    ):
        super().__init__(name=name)

        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            d_model=d_model,
            name="mha",
        )

        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            name="ffn",
        )

        self.ln1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.ln2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, seq_len, d_model)
        """

        # Self attention
        attn_out = self.mha(x)

        # Residual connection + normalisation
        x = self.ln1(x + attn_out)

        # Feed forward Neural Network
        ffn_out = self.ffn(x)

        # Residual connection and normalisation
        x = self.ln2(x + ffn_out) 

        return x
