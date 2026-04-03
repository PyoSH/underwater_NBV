from dataclasses import dataclass, field
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from .ImagingSonar import ImagingSonar

@configclass
class ImagingSonarCfg(CameraCfg):
    class_type:     type=ImagingSonar
    data_types:     list[str]=field(default_factory=lambda:[
        "distance_to_camera",
        "normals",
        "semantic_segmentation",
    ])
    min_range:      float=0.2
    max_range:      float=3.0
    range_res:      float=0.008
    angular_res:    float=0.5
    hori_res:       float=3000.0
    hori_fov:       float=130.0
    vert_fov:       float=20.0

    attenuation:        float=0.1
    binning_method:     str="sum"
    normalizing_method: str="range"

    intensity_offset:   float=0.0
    intensity_gain:     float=1.0

    gau_noise_param:    float=0.2
    ray_noise_param:    float=0.05
    central_peak:       float=2.0
    central_std:        float=0.001

    semantic_to_reflectivity:   dict=field(default_factory=dict)

    enable_viewport:    bool=False