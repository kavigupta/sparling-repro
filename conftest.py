"""
Loaded implicitly by pytest when running tests in this directory.

Auto-remaps CUDA operations to CPU when no GPU is available.
"""

import torch


if not torch.cuda.is_available():
    # Patch torch.load to default to CPU
    _original_load = torch.load

    def _cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return _original_load(*args, **kwargs)

    torch.load = _cpu_load

    # Patch .cuda() on tensors and modules to be identity (stay on CPU)
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
