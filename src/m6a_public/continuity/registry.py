"""Five registered prototypes; factory never loads data or checkpoints."""
from .alignment_innovation import AdaptiveAlignmentDecoder, InnovationStateDecoder
from .dense_state import DenseMultiscaleDecoder, PersistentStateDecoder
from .predictive import PredictiveContinuityDecoder


MODELS = {
    "dense_multiscale": DenseMultiscaleDecoder,
    "persistent_state": PersistentStateDecoder,
    "predictive": PredictiveContinuityDecoder,
    "adaptive_alignment": AdaptiveAlignmentDecoder,
    "innovation_state": InnovationStateDecoder,
}


def build_model(name, **kwargs):
    if name not in MODELS:
        raise ValueError(f"unknown framework {name!r}; choose from {tuple(MODELS)}")
    return MODELS[name](**kwargs)
