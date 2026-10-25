from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EZTwoLayerPrepArtifacts:
    diff_down_up: Optional[np.ndarray] = None
    diff_up_down: Optional[np.ndarray] = None
    hblur_down: Optional[np.ndarray] = None
    hblur_up: Optional[np.ndarray] = None

    diff_up_down_tubed: Optional[np.ndarray] = None
    gradient_stages: list[np.ndarray] = field(default_factory=list)
    input_img: Optional[np.ndarray] = None


@dataclass
class RPEArtifacts:
    store_full: bool = True #Setting true for now

    ez_two_layer_prep: Optional[EZTwoLayerPrepArtifacts] = None


@dataclass
class ILMArtifacts:
    store_full: bool = False