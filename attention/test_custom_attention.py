import torch
import torch.nn.functional as F
import numpy as np
from custom_attention import CustomAttention, SimpleLLM, scaled_dot_product_attention


def test_custom_scaled_dot_product_attention():
    """Test the custom scaled dot product attention against PyTorch's implementation"""
    print("Testing custom scaled dot product attention...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define parameters
    batch_size = 2
    num_heads = 4
    seq_len = 8
    head_dim = 16
    
    # Create random tensors
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)
    
    # Test without mask
    custom_output = scaled_dot_product_attention(query, key, value)
    
    # Compare with PyTorch's implementation
    pytorch_output = F.scaled_dot_product_attention(query, key, value)
    
    # Check if outputs are close
    diff = torch.abs(custom_output - pytorch_output).max().item()
    print(f"Max difference without mask: {diff}")
    assert diff < 1e-5, f"Difference too large: {diff}"
    
    # Test with attention mask using boolean mask (like PyTorch expects)
    # Create a boolean mask where True means "masked out"
    mask_bool = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    mask_bool = mask_bool.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
    
    # Convert to additive mask (negative infinity for masked positions)
    mask_additive = mask_bool.float() * float('-inf')
    
    custom_output_masked = scaled_dot_product_attention(query, key, value, attn_mask=mask_additive)
    pytorch_output_masked = F.scaled_dot_product_attention(query, key, value, attn_mask=mask_additive)
    
    # Check for NaN values
    assert not torch.isnan(custom_output_masked).any(), "Custom implementation produced NaN values"
    assert not torch.isnan(pytorch_output_masked).any(), "PyTorch implementation produced NaN values"
    
    diff_masked = torch.abs(custom_output_masked - pytorch_output_masked).max().item()
    print(f"Max difference with mask: {diff_masked}")
    assert diff_masked < 1e-5, f"Difference too large with mask: {diff_masked}"
    
    # Test causal attention
    custom_output_causal = scaled_dot_product_attention(query, key, value, is_causal=True)
    pytorch_output_causal = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    
    diff_causal = torch.abs(custom_output_causal - pytorch_output_causal).max().item()
    print(f"Max difference with causal: {diff_causal}")
    assert diff_causal < 1e-5, f"Difference too large with causal: {diff_causal}"
    
    # Test gradients
    loss_custom = custom_output.sum()
    loss_custom.backward()
    
    grad_query_custom = query.grad.clone()
    grad_key_custom = key.grad.clone()
    grad_value_custom = value.grad.clone()
    
    # Zero gradients and test PyTorch implementation
    query.grad.zero_()
    key.grad.zero_()
    value.grad.zero_()
    
    loss_pytorch = pytorch_output.sum()
    loss_pytorch.backward()
    
    grad_query_pytorch = query.grad.clone()
    grad_key_pytorch = key.grad.clone()
    grad_value_pytorch = value.grad.clone()
    
    # Compare gradients
    grad_diff_q = torch.abs(grad_query_custom - grad_query_pytorch).max().item()
    grad_diff_k = torch.abs(grad_key_custom - grad_key_pytorch).max().item()
    grad_diff_v = torch.abs(grad_value_custom - grad_value_pytorch).max().item()
    
    print(f"Gradient differences - Query: {grad_diff_q}, Key: {grad_diff_k}, Value: {grad_diff_v}")
    assert grad_diff_q < 1e-5, f"Gradient difference for query too large: {grad_diff_q}"
    assert grad_diff_k < 1e-5, f"Gradient difference for key too large: {grad_diff_k}"
    assert grad_diff_v < 1e-5, f"Gradient difference for value too large: {grad_diff_v}"
    
    print("✅ Custom scaled dot product attention test passed!")


def test_custom_attention_module():
    """Test the custom attention module"""
    print("\nTesting custom attention module...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define parameters
    embed_dim = 64
    num_heads = 4
    batch_size = 2
    seq_len = 8
    
    # Create custom attention module
    attention = CustomAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
    
    # Create random input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass
    output = attention(x, x, x)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} vs {expected_shape}"
    
    print(f"Output shape: {output.shape}")
    print("✅ Custom attention module test passed!")


def test_simple_llm_model():
    """Test the simple LLM model"""
    print("\nTesting simple LLM model...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define parameters
    vocab_size = 1000
    embed_dim = 128
    num_heads = 8
    hidden_dim = 256
    num_layers = 2
    max_seq_len = 32
    batch_size = 2
    seq_len = 16
    
    # Create model
    model = SimpleLLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )
    
    # Create random input tokens
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(input_ids)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape, f"Output shape mismatch: {logits.shape} vs {expected_shape}"
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print("✅ Simple LLM model test passed!")


def test_accuracy_comparison():
    """Compare accuracy between custom attention and PyTorch's built-in attention"""
    print("\nTesting accuracy comparison...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define parameters
    embed_dim = 64
    num_heads = 4
    batch_size = 2
    seq_len = 8
    
    # Create inputs
    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    
    # Custom attention model
    custom_attn = CustomAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
    
    # Create equivalent PyTorch MultiheadAttention
    pytorch_attn = torch.nn.MultiheadAttention(
        embed_dim=embed_dim, 
        num_heads=num_heads, 
        dropout=0.0, 
        batch_first=True
    )
    
    # Initialize both models with the same weights
    with torch.no_grad():
        # Copy weights from custom attention to PyTorch attention
        pytorch_attn.in_proj_weight.copy_(
            torch.cat([custom_attn.q_proj.weight, 
                      custom_attn.k_proj.weight, 
                      custom_attn.v_proj.weight])
        )
        pytorch_attn.in_proj_bias.copy_(
            torch.cat([custom_attn.q_proj.bias, 
                      custom_attn.k_proj.bias, 
                      custom_attn.v_proj.bias])
        )
        pytorch_attn.out_proj.weight.copy_(custom_attn.out_proj.weight)
        pytorch_attn.out_proj.bias.copy_(custom_attn.out_proj.bias)
    
    # Forward pass with custom attention
    x_copy = x.clone().detach().requires_grad_(True)
    custom_output = custom_attn(x_copy, x_copy, x_copy)
    custom_loss = custom_output.sum()
    custom_loss.backward()
    custom_grad = x_copy.grad.clone()
    
    # Forward pass with PyTorch attention
    x_copy2 = x.clone().detach().requires_grad_(True)
    pytorch_output, _ = pytorch_attn(x_copy2, x_copy2, x_copy2, need_weights=False)
    pytorch_loss = pytorch_output.sum()
    pytorch_loss.backward()
    pytorch_grad = x_copy2.grad.clone()
    
    # Compare outputs
    output_diff = torch.abs(custom_output - pytorch_output).max().item()
    grad_diff = torch.abs(custom_grad - pytorch_grad).max().item()
    
    print(f"Output difference: {output_diff}")
    print(f"Gradient difference: {grad_diff}")
    
    assert output_diff < 1e-4, f"Output difference too large: {output_diff}"
    assert grad_diff < 1e-4, f"Gradient difference too large: {grad_diff}"
    
    print("✅ Accuracy comparison test passed!")


def test_numerical_stability():
    """Test numerical stability of the custom attention implementation"""
    print("\nTesting numerical stability...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define parameters
    batch_size = 1
    num_heads = 1
    seq_len = 4
    head_dim = 4
    
    # Create tensors that could potentially cause numerical issues
    query = torch.ones(batch_size, num_heads, seq_len, head_dim) * 10.0
    key = torch.ones(batch_size, num_heads, seq_len, head_dim) * 10.0
    value = torch.ones(batch_size, num_heads, seq_len, head_dim) * 10.0
    
    # Run attention
    output = scaled_dot_product_attention(query, key, value)
    
    # Check for NaN or Inf values
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    
    print("✅ Numerical stability test passed!")


if __name__ == "__main__":
    test_custom_scaled_dot_product_attention()
    test_custom_attention_module()
    test_simple_llm_model()
    test_accuracy_comparison()
    test_numerical_stability()
    print("\n🎉 All tests passed!")