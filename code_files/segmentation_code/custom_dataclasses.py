from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class HyperPreprocessParams:
    ILM_line: Optional[np.ndarray] = None
    grad_weight: float = 0.5
    kernel_size: int = 20
    blur_sigma: float = 4.0
    suppression_sigma: float = 12.0
    peak_prominence: Optional[float] = None


@dataclass
class HyperCombo:
    name: str
    downsample_factor: int
    rigidity: float
    # dp_input_key: str   # which intermediate becomes DP input
    preprocess_params: HyperPreprocessParams
    lambda_step_in_onh_region: Optional[float] = None


@dataclass
class ImageBundle:
    output: np.ndarray
    blurred: Optional[np.ndarray] = None
    grad_raw: Optional[np.ndarray] = None
    grad_shift: Optional[np.ndarray] = None
    grad_peak_suppressed: Optional[np.ndarray] = None
    peak_img: Optional[np.ndarray] = None,
    grad_ilm_suppressed: Optional[np.ndarray] = None
    combined: Optional[np.ndarray] = None
    combined_peak_suppressed: Optional[np.ndarray] = None