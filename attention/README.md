# Custom Attention Operator with torch.library.custom_op

This project demonstrates the implementation of a custom attention operator using PyTorch's `torch.library.custom_op`. It includes:

1. A custom scaled dot-product attention implementation
2. An attention module that uses the custom operator
3. A simple LLM model that utilizes the custom attention
4. Tests to verify correctness and accuracy

## Files

- `custom_attention.py`: Contains the custom attention operator implementation and the SimpleLLM model
- `test_custom_attention.py`: Tests for verifying the custom attention implementation
- `requirements.txt`: Dependencies needed for the project

## Usage

To run the tests:

```bash
cd /home/zhangxa/codes/auto_hpc/attention
python test_custom_attention.py
```

## Features

- Custom implementation of scaled dot-product attention using `torch.library.custom_op`
- Compatible with PyTorch's autograd system
- Supports attention masks and causal masking
- Includes gradient verification
- Accuracy comparison with PyTorch's built-in attention mechanisms