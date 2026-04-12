"""
Self-Flow Modules.
"""

from .model import SelfFlowDiT
from .sampling import denoise_loop

# Backward compatibility alias
SelfFlowPerTokenDiT = SelfFlowDiT

__all__ = [
    "SelfFlowDiT",
    "SelfFlowPerTokenDiT",
    "denoise_loop",
]

