"""
BROV2 파라미터 로더 — robots/data/BROV2/brov2_heavy.yaml의 단일 진입점
========================================================================
`step_2_BROV/env.py`에 얹혀있던 걸 여기로 승격(2026-07) — step_2_BROV뿐
아니라 향후 step_3(수중 인지+물리 통합)도 BROV2 파라미터가 필요하면
env.py가 아니라 여기를 가져다 쓴다.

CAD/유체계수가 바뀌면 `brov2_heavy.yaml`만 갱신하면 되고, 코드에 값을
중복 하드코딩하지 않는다 — mass/inertia/collision은 USD가 정본이라 여기 없음.
"""

import os

import yaml

_BROV2_YAML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "BROV2", "brov2_heavy.yaml"
)


def load_brov2_yaml(yaml_path: str = _BROV2_YAML_PATH) -> dict:
    """brov2_heavy.yaml 전체를 읽어서 dict로 반환한다."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def coBM_vector_ned(params: dict) -> tuple[float, float, float]:
    """params['coBM'](Z-up body frame, COM 기준)을 SNAME/NED body frame
    (X=전방,Y=우현,Z=하방)으로 변환해 반환한다.
    """
    v = params["coBM"]
    return (v[0], -v[1], -v[2])  # Z-up -> SNAME/NED (T3, self-inverse)


def thruster_pos_dir_ned(params: dict) -> tuple[list, list]:
    """params['thrusters']['list']의 position/axis(Z-up body frame, USD 정본의 미러)를
    SNAME/NED body frame(X=전방,Y=우현,Z=하방)으로 변환해 (pos, dir) 리스트로 반환한다.

    리스트 순서가 곧 T1~T8 순서 — BROV2ThrusterModel._POS/_DIR과 동일 인덱싱.
    """
    pos, dir_ = [], []
    for t in params["thrusters"]["list"]:
        px, py, pz = t["position"]
        ax, ay, az = t["axis"]
        pos.append([px, -py, -pz])   # Z-up -> SNAME/NED (T3, self-inverse)
        dir_.append([ax, -ay, -az])
    return pos, dir_
