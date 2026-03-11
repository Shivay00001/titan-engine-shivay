"""TITAN: Trillion-scale Intelligent Training Architecture for Networks"""
from .core.hms   import HMSStreamingEngine, NVMeBlockStore
from .core.mlme  import MicroHeadAttention, StripeFFN, ErrorAccumulationBank
from .core.asdt  import ASDTOptimizer, ParameterClass
from .core.trd   import TRDLinear, TensorRingMatrix, convert_model_to_trd
from .core.tgss  import TGSSManager, CountMinSketch
from .core.bsps  import BSPSManager, Phase
from .core.hge   import HGEManager, GradientHologram
from .training.trainer import TITANTrainer, TITANConfig, build_titan_trainer

__version__ = "0.1.0"
__all__ = [
    "HMSStreamingEngine", "NVMeBlockStore",
    "MicroHeadAttention", "StripeFFN", "ErrorAccumulationBank",
    "ASDTOptimizer", "ParameterClass",
    "TRDLinear", "TensorRingMatrix", "convert_model_to_trd",
    "TGSSManager", "CountMinSketch",
    "BSPSManager", "Phase",
    "HGEManager", "GradientHologram",
    "TITANTrainer", "TITANConfig", "build_titan_trainer",
]
