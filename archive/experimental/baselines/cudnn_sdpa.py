"""
cuDNN Frontend SDPA (Optional)

Placeholder for cuDNN frontend integration.
This is optional - if not implemented, benchmark will skip it.
"""

from .registry import register

@register("cudnn_frontend_sdpa")
def cudnn_sdpa(*args, **kwargs):
    """cuDNN Frontend SDPA - not yet implemented"""
    raise NotImplementedError(
        "cuDNN Frontend SDPA not yet wired. "
        "See: https://github.com/NVIDIA/cudnn-frontend/tree/main/samples/python"
    )

