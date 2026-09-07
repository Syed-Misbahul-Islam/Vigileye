"""VigilEye - real-time multimodal driver drowsiness & distraction detection."""

__version__ = "1.0.0"

from .config import load_config, Cfg
from .fusion import DriverState, FusionResult, Signals, FEATURE_NAMES

__all__ = [
    "load_config",
    "Cfg",
    "DriverState",
    "FusionResult",
    "Signals",
    "FEATURE_NAMES",
    "__version__",
]
