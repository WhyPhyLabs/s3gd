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
):
    pid = tl.program_id(axis = 0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # load
    p = tl.load(p_ptr + offsets, mask = mask)
    g = tl.load(g_ptr + offsets, mask = mask)
    m = tl.load(m_ptr + offsets, mask = mask)

    lr = tl.full((), lr_p, p.dtype)
    wd = tl.full((), wd_p, p.dtype)
    beta = tl.full((), beta_p, p.dtype)

    # stepweight decay

    p = p * (1 - lr * wd)

    # --- sign(grad) ---
    sgn_g = tl.where(g > 0, 1.0, tl.where(g < 0, -1.0, 0.0))

    # --- Smoothed SignSGD momentum update ---
    m_new = beta * m + (1.0 - beta) * sgn_g

    # --- parameter update ---
    p_new = p - lr * m_new


    # store new params and momentum running average coefficient

    tl.store(p_ptr + offsets, p_new, mask=mask)
    tl.store(m_ptr + offsets, m_new, mask=mask)

def smoothed_signsgd_step(
    p: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    lr: float,
    wd: float,
    beta: float
):
    assert all([t.is_cuda for t in (p, grad, momentum)])
    n_elements = p.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)    

    smoothed_signsgd_step_kernel[grid](
        p, grad, momentum,
        lr,
        wd,
        beta,
        n_elements
    )
