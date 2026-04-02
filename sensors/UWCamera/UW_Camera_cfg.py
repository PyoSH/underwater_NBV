from dataclasses import dataclass, field
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from .UW_Camera_parallel import UWCamera

@configclass
class UWCameraCfg(CameraCfg):
    class_type:         type = UWCamera 
    data_types:         list[str] = field(default_factory=lambda: [
        "rgba",
        "distance_to_camera" 
    ])

    backscatter_value:  tuple = (0.0, 0.31, 0.24)
    atten_coeff:        tuple = (0.05, 0.05, 0.2)
    backscatter_coeff:  tuple = (0.05, 0.05, 0.05)

    enable_viewport:    bool = False
    viewport_env_id:    int = 0