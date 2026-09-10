"""Water Linked DVL A50 모사 — SITL 이 실기 오차 특성을 재현하게 한다. 2026-09-10.

왜 필요한가: 섬도우 추정기를 SITL 에서 검증하려면 SITL 의 DVL 이 A50 처럼 **틀려야**
한다. GT 에서 뽑은 완벽한 속도로는 추정기가 항상 만점을 받아 시험이 무의미하다.

전부 공식 자료 값이다 (waterlinked.com 데이터시트 · docs.waterlinked.com):

    4-beam convex Janus,  빔각 22.5 deg
    최소 고도 5 cm,  최대 50 m,  ping 4~15 Hz
    장기 정확도 ±1.01 % (Standard) / ±0.1 % (Performance)
    DR = DVL 속도 + 내장 IMU/AHRS 칼만 융합
    DR yaw 드리프트 0.1~0.3 deg/min (**자이로 보정 후**)
    FOM = X-Y 평면 위치의 추정 표준편차, 속도 불가 시 상승
    bottom lock 상실 시 가속도만으로 예측 -> 큰 오차

틸트 한계는 **사양에 없다**. 확실한 것과 아닌 것을 갈라 둔다 (2026-09-10 정정):

  확실(기하)  look_at 자세에서 DVL 축(body -Z)은 수직에서 정확히 (90 - phi) 만큼 기운다.
              카메라(body +X)와 DVL(body -Z)이 서로 수직이기 때문이다 — 카메라가 바닥을
              정면으로 볼수록 DVL 은 수평을 향한다.
  확실(기하)  phi < 22.5 deg 에서 4빔 중 **1개**가 수평 위로 넘어간다.
  **불확실**  그래도 3빔이 남고 4-beam Janus 는 3빔이면 3-D 해가 나온다. 따라서
              "바닥 반사가 없어 lock 불가" 는 **성립하지 않는다**(초기 서술 오류).
              실제 한계는 스치는 입사각·사거리인데 **데이터시트에 값이 없다**.
              phi=10 deg 에서 최악 유효빔은 수직이탈 80.8 deg, 고도 1.5 m 에 사거리 9.3 m.

그래서 `max_beam_from_vertical_deg` 는 **측정되지 않은 손잡이**다. 기본값 None 은
기하만 본다(낙관적). 실기에서 재기 전까지 이 값을 스윕해 결과의 민감도를 보인다.
실기 증거는 전부 거의 수평 자세에서 나온 것이라(순항 valid 82~88 %) 기울인 상태의
데이터는 **아예 없다**.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

BEAM_ANGLE_DEG = 22.5
N_BEAMS = 4
MIN_ALTITUDE_M = 0.05
MAX_ALTITUDE_M = 50.0
BEAMS_REQUIRED = 3          # Janus 4빔 중 3빔이면 3-D 속도 해가 나온다


@dataclass
class A50Config:
    """장착·성능 설정. 기본값은 이 프로젝트의 BlueROV2 Heavy."""
    # DVL_frame, robots/data/BROV2/brov2_custom_physics.usda (body FLU)
    lever_arm_m: tuple = (-0.17052628, -0.09799999, -0.20666213)
    # 데이터시트: Standard ±1.01 %, Performance ±0.1 %. 보유 버전 미확인 -> 보수적으로 Standard
    long_term_accuracy: float = 0.0101
    # docs: 자이로 보정 후 0.1~0.3 deg/min
    yaw_drift_deg_per_min: float = 0.2
    # 스치는 각 한계. **데이터시트에 없다 — 측정 전까지는 추정치를 박지 않는다.**
    # None = 기하만 판정(낙관적). 값을 주면 그 각을 넘는 빔을 무효로 센다.
    # 설계 판단은 이 값을 스윕한 민감도로 하고, 실기 측정으로 확정한다.
    max_beam_from_vertical_deg: float | None = None
    velocity_noise_std: float = 0.003


@dataclass
class A50State:
    """DR 누적 상태. `reset()` 이 A50 의 reset_dead_reckoning 에 해당한다."""
    p_dr: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_bias_rad: float = 0.0
    scale_err: float = 0.0
    t_last: float | None = None
    t_reset: float = 0.0

    def reset(self, t: float, rng: np.random.Generator, cfg: A50Config) -> None:
        """정지 상태 자이로 보정 + DR 리셋. 새 오차 실현값을 뽑는다."""
        self.p_dr = np.zeros(3)
        self.yaw_bias_rad = 0.0
        # 장기 정확도는 '이동거리의 몇 %' 이므로 런당 스케일 실현값으로 모사한다
        self.scale_err = float(rng.uniform(-cfg.long_term_accuracy, cfg.long_term_accuracy))
        self.t_last = t
        self.t_reset = t


def beam_directions(R_wb: np.ndarray) -> list[np.ndarray]:
    """world 기준 4빔 방향. body -Z 를 축으로 22.5 deg, 90 deg 간격."""
    axis = R_wb @ np.array([0.0, 0.0, -1.0])
    ref = R_wb @ np.array([1.0, 0.0, 0.0])
    e1 = ref - float(ref @ axis) * axis
    n = np.linalg.norm(e1)
    if n < 1e-9:                       # 축이 ref 와 나란하면 다른 기준축
        ref = R_wb @ np.array([0.0, 1.0, 0.0])
        e1 = ref - float(ref @ axis) * axis
        n = np.linalg.norm(e1)
    e1 /= n
    e2 = np.cross(axis, e1)
    ca, sa = math.cos(math.radians(BEAM_ANGLE_DEG)), math.sin(math.radians(BEAM_ANGLE_DEG))
    return [ca * axis + sa * (math.cos(a) * e1 + math.sin(a) * e2)
            for a in (2 * math.pi * k / N_BEAMS for k in range(N_BEAMS))]


def bottom_lock(p_dvl_w: np.ndarray, R_wb: np.ndarray, floor_z: float,
                cfg: A50Config) -> tuple[bool, int, float]:
    """(velocity_valid, 유효 빔 수, 고도[m]).

    고도는 DVL 에서 바닥까지의 **수직** 거리다. 빔은 바닥을 향하고, 수평 위를 향하는
    빔은 반사가 없다. 스치는 각은 `max_beam_from_vertical_deg` 로 자른다.
    """
    altitude = float(p_dvl_w[2] - floor_z)
    if not (MIN_ALTITUDE_M <= altitude <= MAX_ALTITUDE_M):
        return False, 0, altitude
    good = 0
    for d in beam_directions(R_wb):
        if d[2] >= -1e-6:                       # 수평 이상 -> 바닥 반사 없음
            continue
        ang = math.degrees(math.acos(min(1.0, -d[2])))   # 수직에서 벗어난 각
        if cfg.max_beam_from_vertical_deg is not None and ang > cfg.max_beam_from_vertical_deg:
            continue
        good += 1
    return good >= BEAMS_REQUIRED, good, altitude


def measured_body_velocity(v_body_true: np.ndarray, omega_body: np.ndarray,
                           cfg: A50Config, rng: np.random.Generator) -> np.ndarray:
    """DVL 이 **실제로 보는** body 속도 = 차체 속도 + w x r + 잡음.

    레버암 항은 옵션이 아니다: |r|=0.285 m 이므로 |w|=0.2 rad/s 에서 이미 0.057 m/s,
    순항 0.25 m/s 의 23 % 다.
    """
    r = np.asarray(cfg.lever_arm_m)
    noise = rng.normal(0.0, cfg.velocity_noise_std, size=3)
    return v_body_true + np.cross(omega_body, r) + noise


def dr_step(state: A50State, t: float, v_body_meas: np.ndarray, yaw_true: float,
            R_wb: np.ndarray, valid: bool, cfg: A50Config) -> None:
    """A50 내부 DR 한 스텝. valid=False 면 속도가 없어 **적분을 멈추고** 오차만 큰다.

    실기와 같은 실패 모드를 만든다 — A50 은 가속도로 예측하려 들지만 그 결과는
    "substantial errors" 라고 문서가 직접 말한다. 여기서는 위치를 유지하고 FOM 만
    키운다(가속도 예측을 흉내 내면 SITL 이 실기보다 낙관적/비관적 어느 쪽으로도 틀릴 수 있다).
    """
    if state.t_last is None:
        state.t_last = t
        return
    dt = t - state.t_last
    state.t_last = t
    if dt <= 0.0:
        return
    state.yaw_bias_rad += math.radians(cfg.yaw_drift_deg_per_min / 60.0) * dt
    if not valid:
        return
    # DR 은 자기 yaw(드리프트 포함)로 body 속도를 회전시킨다
    yaw_dr = yaw_true + state.yaw_bias_rad
    c, s = math.cos(yaw_dr), math.sin(yaw_dr)
    # roll/pitch 는 A50 AHRS 가 중력으로 잡아 드리프트가 없다고 본다 -> 참값 사용
    v_lvl = R_wb @ v_body_meas
    yaw_only = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    yaw_true_inv = np.array([[math.cos(-yaw_true), -math.sin(-yaw_true), 0.0],
                             [math.sin(-yaw_true), math.cos(-yaw_true), 0.0],
                             [0.0, 0.0, 1.0]])
    v_dr = yaw_only @ (yaw_true_inv @ v_lvl)
    state.p_dr = state.p_dr + v_dr * dt * (1.0 + state.scale_err)


def figure_of_merit(state: A50State, t: float, valid: bool, last_valid_t: float,
                    cfg: A50Config) -> float:
    """FOM = X-Y 위치 표준편차 추정 [m].

    유효할 때는 이동거리의 장기정확도 비율, 무효 구간에서는 가속도 예측의 2차 발산을
    쓴다(bias 0.01 m/s^2 가정 -> 0.5*a*t^2). 문서의 "속도 불가 시 FOM 상승" 을 잰다.
    """
    travelled = float(np.linalg.norm(state.p_dr[:2]))
    fom = cfg.long_term_accuracy * travelled + 0.01
    if not valid:
        gap = max(0.0, t - last_valid_t)
        fom += 0.5 * 0.01 * gap * gap
    return fom
