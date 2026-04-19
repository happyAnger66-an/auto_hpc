import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.library import custom_op
from typing import Optional, Tuple


@custom_op("custom_attn::scaled_dot_product_attention", mutates_args=())
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Custom implementation of scaled dot product attention using torch.library.custom_op
    
    Args:
        query: Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
        key: Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim)
        value: Value tensor of shape (batch_size, num_heads, seq_len_v, head_dim)
        attn_mask: Attention mask tensor
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        scale: Scale factor for attention scores
    
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
    """
    # Calculate the scale factor
    if scale is None:
        scale_factor = query.size(-1) ** -0.5
    else:
        scale_factor = scale
    
    # Compute attention scores
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    
    # Apply causal mask if required
    if is_causal:
        L = query.size(-2)
        M = key.size(-2)
        causal_mask = torch.tril(torch.ones((L, M), dtype=torch.bool, device=query.device))
        attn_weights.masked_fill_(~causal_mask, float('-inf'))
    
    # Apply attention mask if provided
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    
    # Apply softmax to get attention probabilities
    # Using softmax with finite values ensures numerical stability
    attn_weights = torch.softmax(attn_weights, dim=-1)
    
    # Apply dropout if specified
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    # Compute output
    output = torch.matmul(attn_weights, value)
    
    return output


# Register the abstract implementation for shape inference
from torch.library import register_fake

@register_fake("custom_attn::scaled_dot_product_attention")
def scaled_dot_product_attention_abstract(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Abstract implementation for shape inference"""
    return torch.empty_like(query)


class CustomAttention(nn.Module):
    """
    Custom Attention module using the custom scaled dot product attention operator
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = dropout
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None, 
                is_causal: bool = False) -> torch.Tensor:
        """
        Forward pass of the custom attention module
        """
        batch_size, seq_len, embed_dim = query.shape
        
        # Project query, key, and value
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply custom attention operator
        attn_output = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=is_causal,
            scale=self.scale
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)
        
        return output


class SimpleLLM(nn.Module):
    """
    Simple LLM model using the custom attention operator
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, max_seq_len: int):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(0.1)
        )
        
        # Stack of attention layers
        self.attention_layers = nn.ModuleList([
            CustomAttention(embed_dim, num_heads, dropout=0.1) 
            for _ in range(num_layers)
        ])
        
        # Final output layer
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the simple LLM model
        """
        batch_size, seq_len = x.shape
        
        # Embed tokens and add positional encoding
        x = self.token_embedding(x)
        pos_enc = self.pos_encoding[:, :seq_len, :]
        x = x + pos_enc
        x = self.dropout(x)
        
        # Pass through attention layers
        for attn_layer in self.attention_layers:
            # Self-attention
            attn_out = attn_layer(x, x, x)
            # Add & Norm
            x = self.norm1(x + attn_out)
            
            # Feed-forward network
            ff_out = self.ffn(x)
            # Add & Norm
            x = self.norm2(x + ff_out)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits