"""
Baseline Attention Implementations Registry

Provides drop-in baselines for L4 (Ada, sm_89):
- pytorch_sdpa_{flash,cudnn,efficient,math}
- flashattn2
- cudnn_frontend_sdpa (optional)
"""

from . import registry
from . import pytorch_sdpa
from . import flashattn2
# cudnn_sdpa is optional
try:
    from . import cudnn_sdpa
except ImportError:
    pass

__all__ = ['registry']

