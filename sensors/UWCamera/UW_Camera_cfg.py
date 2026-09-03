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

    # 수중 렌더를 `update()`에서 매번 돌리지 않고, 환경이 `refresh_uw()`로
    # **결정당 1회** 명시적으로 돌리게 한다.
    #
    # decimation이 큰 환경(step_3: 500)에서만 의미가 있다. `scene.update()`가
    # 물리 서브스텝마다 불리는데 실제 GPU 렌더는 `render_interval`마다 1회뿐이라,
    # 매번 돌리면 같은 프레임을 499번 더 계산한다(2026-09-03 프로파일: 전체
    # wall time의 75.1%, 결정당 8.31초 → 1.71초로 4.87배 단축).
    #
    # **기본값이 False인 이유**: decimation=1인 환경(step_1_NBV)은 매 스텝
    # 프레임이 새로 나므로 기존 동작이 옳고, 그쪽은 `refresh_uw()`를 부르지
    # 않는다. 기본을 True로 두면 step_1의 카메라가 첫 프레임에 얼어붙는다.
    defer_uw_render:    bool = False