# # rhino/nn/ops/__init__.py
# import os
# import warnings
# import torch
# from torch.utils.cpp_extension import load as _load

# _here = os.path.dirname(__file__)
# _cpp = os.path.join(_here, "cpp")

# cpp_sources = [
#     os.path.join(_cpp, "cudabind.cpp"),
#     os.path.join(_cpp, "bias_act.cpp"),
#     os.path.join(_cpp, "filtered_lrelu.cpp"),
#     os.path.join(_cpp, "fused_bias_leakyrelu.cpp"),
#     os.path.join(_cpp, "upfirdn2d.cpp"),
# ]

# cuda_sources = [
#     os.path.join(_cpp, "bias_act_cuda.cu"),
#     os.path.join(_cpp, "filtered_lrelu.cu"),
#     os.path.join(_cpp, "fused_bias_leakyrelu_cuda.cu"),
#     os.path.join(_cpp, "upfirdn2d_kernel.cu"),
# ]

# sources = list(cpp_sources)
# if torch.cuda.is_available():
#     sources += cuda_sources
# else:
#     warnings.warn("CUDA not available — building CPU-only extension.")

# extra_cflags = ["-O3"]
# extra_cuda_cflags = ["-O3"]

# # IMPORTANT: is_python_module=False
# _load(
#     name="rhino_ext_jit",
#     sources=sources,
#     extra_cflags=extra_cflags,
#     extra_cuda_cflags=extra_cuda_cflags,
#     is_python_module=False,
#     verbose=True,
# )

# # After dlopen, your kernels registered with:
# # TORCH_LIBRARY(_ext, m) { m.def(...); }
# # TORCH_LIBRARY_IMPL(_ext, CUDA/CPU, m) { m.impl(...); }
# # are now available as torch.ops._ext.*
