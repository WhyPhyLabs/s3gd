import torch

try:
    import triton
    import triton.language as tl
except ImportError as e:
    print('triton is not installed, please install by running `pip install triton>=2.2.0`')
    exit()

# triton cuda kernel

@triton.autotune(configs = [
    triton.Config({'BLOCK_SIZE': 128}, num_warps = 4),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps = 8),
], key = ['n_elements'], restore_value=['p_ptr', 'm_ptr'])
@triton.jit
def smoothed_signsgd_step_kernel(
    p_ptr,
    g_ptr,
    m_ptr,
    lr_p,
    wd_p,
    beta_p,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    T_MATH: tl.constexpr,  # arithmetic dtype (e.g. tl.bfloat16)
    T_STATE: tl.constexpr,  # momentum-storage dtype (e.g. tl.float8e4nv)
):
    pid = tl.program_id(axis = 0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # load
    p = tl.load(p_ptr + offsets, mask = mask, other=T_MATH)
    g = tl.load(g_ptr + offsets, mask = mask, other=T_MATH)
    m = tl.load(m_ptr + offsets, mask = mask, other=T_STATE)
    m = tl.astype(m, T_MATH)

    lr   = tl.full((), lr_p,   T_MATH)
    wd   = tl.full((), wd_p,   T_MATH)
    beta = tl.full((), beta_p, T_MATH)

    # stepweight decay

    p = p * (1 - lr * wd)

    # --- sign(grad) ---
    sgn_g = tl.where(g > 0, 1.0, tl.where(g < 0, -1.0, 0.0))

    # --- Smoothed SignSGD momentum update ---
    m = beta * m + (1.0 - beta) * sgn_g

    # --- parameter update ---
    p = p - lr * m


    # store new params and momentum running average coefficient

    tl.store(p_ptr + offsets, p, mask=mask)
    tl.store(m_ptr + offsets, tl.astype(m, T_STATE), mask=mask)

_DTYPE_MAP = {
    "bf16" : (tl.bfloat16, torch.bfloat16),
    "fp16" : (tl.float16,  torch.float16),
    "fp32" : (tl.float32,  torch.float32),
    "fp8e4": (tl.float8e4nv, torch.float8_e4m3fn),   # PyTorch ≥ 2.3
    "fp8e5": (tl.float8e5nv, torch.float8_e5m2),     # idem
}

def smoothed_signsgd_step(
    p: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    lr: float,
    wd: float,
    beta: float,
    math_dtype: str = "bf16",  # bf16 arithmetic
    state_dtype: str = "fp8e4",  # fp8 momentum
):
    T_MATH, torch_math = _DTYPE_MAP[math_dtype]
    T_STATE, torch_state = _DTYPE_MAP[state_dtype]

    assert all([t.is_cuda for t in (p, grad, momentum)])
    assert grad.dtype == p.dtype == torch_math, \
        "p & grad must be in math_dtype"
    assert momentum.dtype == torch_state, \
        "momentum must be in state_dtype"

    n_elements = p.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)    

    smoothed_signsgd_step_kernel[grid](
        p, grad, momentum,
        lr,
        wd,
        beta,
        n_elements,
        BLOCK_SIZE=None,
        T_MATH=T_MATH,
        T_STATE=T_STATE,
    )
