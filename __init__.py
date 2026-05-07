"""Action Genome detector package built around a DINOv3 Faster R-CNN backbone."""

from .ag_dataset import ActionGenomeSceneSayerDetectorDataset, collate_fn
from .detector_eval import evaluate_detection_map, evaluate_detection_map_distributed
from .dinov3_fasterrcnn import create_scenesayer_detector_model

__all__ = [
    "ActionGenomeSceneSayerDetectorDataset",
    "collate_fn",
    "create_scenesayer_detector_model",
    "evaluate_detection_map",
    "evaluate_detection_map_distributed",
]
