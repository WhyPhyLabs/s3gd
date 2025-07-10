## S3GD

S3GD is a highly-optimized, PyTorch-compatible Triton implementation of the Smoothed SignSGD optimizer, meant for reinforcement learning post-training. You can find more information in our <a href="https://whyphy.ai/blog">blog post</a>.  
  
FP8 instructions are only available on SM 90 or newer. The kernel will compile on earlier generations if you use a state_dtype other than FP8.

## Install 

```bash
$ git clone https://github.com/WhyPhyLabs/smoothed-sign-sgd.git

$ pip install .
```

## Usage Examples

```python
1) bf16 maths + fp8 state on Hopper / H200
p        = torch.zeros_like(w0, dtype=torch.bfloat16, device="cuda")
g        = torch.zeros_like(w0, dtype=torch.bfloat16, device="cuda")
momentum = torch.zeros_like(w0, dtype=torch.float8_e4m3fn, device="cuda")


smoothed_signsgd_step(
    p, g, momentum,
    lr=3e-3, wd=1e-2, beta=0.9,
    math_dtype="bf16", state_dtype="fp8e4",
)


2) all-bf16 fallback (Ampere GPUs don’t support FP8)
smoothed_signsgd_step(
    p_bf16, g_bf16, m_bf16,
    lr=1e-2, wd=0., beta=0.95,
    math_dtype="bf16", state_dtype="bf16",
)
```

To use a fused kernel for updating the parameters, first `pip install triton -U --pre`, then

```python
opt = SmoothedSignSGD(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-2,
    use_triton=True # set this to True to use cuda kernel with Triton lang
)
```

## Citations

```bibtex
@online{kalomaze2025tweet,
	url = {https://x.com/kalomaze/status/1940424032119316813},
	author = {kalomaze},
	title = {On Adam with aggressive gradient clipping causing sparse updates},
	year = {2025}
}
```

```bibtex
@misc{cesista2025adamagressiveclipping,
	url = {http://leloykun.github.io/ponder/adam-aggressive-clipping/},
	author = {Franz Louis Cesista},
	title = {Adam with Agressive Gradient Clipping ≈ Smoothed SignSGD/NormSGD},
	year = {2025}
}
```

```bibtex
@misc{chen2023symbolic,
	url = {https://arxiv.org/abs/2302.06675},
	author = {Chen, Xiangning and Liang, Chen and Huang, Da and Real, Esteban and Wang, Kaiyuan and Liu, Yao and Pham, Hieu and Dong, Xuanyi and Luong, Thang and Hsieh, Cho-Jui and Lu, Yifeng and Le, Quoc V.},
	title = {Symbolic Discovery of Optimization Algorithms},
	publisher = {arXiv},
	year = {2023}
}
```

```bibtex
@misc{lucidrains,
	url = {https://github.com/lucidrains/lion-pytorch},
	author = {Phil Wang},
	title = {lion-pytorch},
	year = {2024}
}
```
