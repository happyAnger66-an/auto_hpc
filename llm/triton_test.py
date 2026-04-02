import torch
import torch_tensorrt
import triton
import triton.language as tl
import tensorrt.plugin as trtp

# --- 1. 编写 Triton Kernel ---
@triton.jit
def mul_kernel(X, Y, Z, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_vals = tl.load(X + offsets)
    y_vals = tl.load(Y + offsets)
    z_vals = x_vals * y_vals
    tl.store(Z + offsets, z_vals)

# --- 2. 包装成 PyTorch 自定义算子 ---
@torch.library.custom_op("my_lib::mul", mutates_args=())
def torch_mul(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    Z = torch.empty_like(X)
    BLOCK_SIZE = 1024
    grid = lambda meta: (X.numel() // meta["BLOCK_SIZE"],)
    mul_kernel[grid](X, Y, Z, BLOCK_SIZE=BLOCK_SIZE)
    return Z

# 注册一个“元内核”，用于形状推导
@torch.library.register_fake("my_lib::mul")
def _(X, Y):
    return X  # 输出形状与输入相同