"""16차원 BROV observation을 위한 명시적 model-based PI/PD 제어기.

입력 순서는 학습 환경/``ObservationBuilder``와 동일하다::

    q_e(4), v_e_b(3), omega_b(3), z_v(3), z_q(3)

오차는 모두 current - desired로 정의되어 있으므로 안정화 wrench에는 음의
피드백을 적용한다. Observation의 body frame은 Isaac/정책 frame(X forward,
Y left, Z up)이고 thruster allocation matrix는 SNAME(X forward, Y right,
Z down)이므로 wrench를 allocation 전에 명시적으로 좌표 변환한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from deploy.vendor.thruster import BROV2ThrusterModel, build_allocation_matrix


_WRENCH_ZUP_TO_SNAME = torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])
_F_MAX = torch.tensor([85.0, 85.0, 120.0, 26.0, 14.0, 22.0])


@dataclass(frozen=True)
class ControllerOutput:
    wrench_zup: torch.Tensor
    wrench_sname: torch.Tensor
    estimated_wrench_zup: torch.Tensor
    normalized_action_zup: torch.Tensor
    thruster_force: torch.Tensor
    pwm: torch.Tensor


def _vec3(value, name: str, *, positive: bool = False) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32)
    if result.shape != (3,) or not torch.isfinite(result).all():
        raise ValueError(f"{name}은 유한한 3개 값이어야 함")
    if positive and bool((result <= 0.0).any()):
        raise ValueError(f"{name}은 모두 양수여야 함")
    return result


def quaternion_error_rotation_vector(q_error: torch.Tensor) -> torch.Tensor:
    """[w,x,y,z] error quaternion을 shortest rotation vector(rad)로 변환한다."""
    q = torch.as_tensor(q_error, dtype=torch.float32)
    if q.shape != (4,) or not torch.isfinite(q).all():
        raise ValueError("q_error는 유한한 quaternion 4개 값이어야 함")
    norm = q.norm()
    if float(norm) < 1e-6:
        raise ValueError("q_error norm이 0에 가까움")
    q = q / norm
    # q와 -q는 같은 자세. w>=0 hemisphere를 써서 항상 최단 회전을 선택한다.
    if float(q[0]) < 0.0:
        q = -q
    vector_norm = q[1:4].norm()
    if float(vector_norm) < 1e-7:
        return 2.0 * q[1:4]
    angle = 2.0 * torch.atan2(vector_norm, q[0].clamp_min(0.0))
    return q[1:4] * (angle / vector_norm)


class ModelBasedController:
    """Body-frame velocity PI + attitude/rate PI-D wrench controller."""

    def __init__(
        self,
        thruster_pos,
        thruster_dir,
        *,
        linear_kp=(25.0, 25.0, 35.0),
        linear_ki=(0.0, 0.0, 0.0),
        attitude_kp=(3.0, 3.0, 3.0),
        attitude_ki=(0.0, 0.0, 0.0),
        angular_kd=(1.5, 1.5, 1.0),
        force_limit=(15.0, 15.0, 20.0),
        torque_limit=(3.0, 3.0, 3.0),
        minimum_active_pwm=0.10,
        thruster_force_activation=0.25,
        device="cpu",
    ):
        self.device = device
        self.linear_kp = _vec3(linear_kp, "linear_kp").to(device)
        self.linear_ki = _vec3(linear_ki, "linear_ki").to(device)
        self.attitude_kp = _vec3(attitude_kp, "attitude_kp").to(device)
        self.attitude_ki = _vec3(attitude_ki, "attitude_ki").to(device)
        self.angular_kd = _vec3(angular_kd, "angular_kd").to(device)
        self.force_limit = _vec3(force_limit, "force_limit", positive=True).to(device)
        self.torque_limit = _vec3(torque_limit, "torque_limit", positive=True).to(device)
        self.minimum_active_pwm = float(minimum_active_pwm)
        self.thruster_force_activation = float(thruster_force_activation)
        if not 0.075 < self.minimum_active_pwm <= 1.0:
            raise ValueError("minimum_active_pwm은 T200 deadband(0.075)보다 크고 1 이하여야 함")
        if self.thruster_force_activation < 0.0:
            raise ValueError("thruster_force_activation은 0 이상이어야 함")

        self.thruster = BROV2ThrusterModel(
            num_envs=1, dt=0.04, device=device, pos=thruster_pos, dir=thruster_dir
        )
        self.B = build_allocation_matrix(self.thruster._pos, self.thruster._dir)
        self.B_pinv = torch.linalg.pinv(self.B)
        self._wrench_transform = _WRENCH_ZUP_TO_SNAME.to(device)
        self._f_max = _F_MAX.to(device)

    def compute(self, observation: torch.Tensor) -> ControllerOutput:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.shape != (16,) or not torch.isfinite(obs).all():
            raise ValueError("observation은 유한한 16차원 벡터여야 함")

        q_e = obs[0:4]
        v_e = obs[4:7]
        omega = obs[7:10]
        z_v = obs[10:13]
        z_q = obs[13:16]
        rotation_error = quaternion_error_rotation_vector(q_e).to(self.device)

        force_zup = -self.linear_kp * v_e - self.linear_ki * z_v
        torque_zup = (
            -self.attitude_kp * rotation_error
            -self.angular_kd * omega
            -self.attitude_ki * z_q
        )
        force_zup = torch.maximum(torch.minimum(force_zup, self.force_limit), -self.force_limit)
        torque_zup = torch.maximum(torch.minimum(torque_zup, self.torque_limit), -self.torque_limit)
        wrench_zup = torch.cat((force_zup, torque_zup))

        # Allocation matrix는 SNAME/FRD, observation/controller는 FLU/Z-up이다.
        wrench_sname = wrench_zup * self._wrench_transform
        desired_force = self.B_pinv @ wrench_sname
        # inverse_thrust() 내부와 같은 물리 한계. 여기서 먼저 clamp해야 진단값도
        # 실제 PWM으로 실현 가능한 추력을 나타낸다. 하드코딩 (-51.5, 64.1)은
        # BlueRobotics 공개 데이터의 20V 값이었고, 4S 팩이 부하에서 내는 14V에서는
        # 실측 -34.5/+44.4 N이다. 이제 thruster 모델에 설정된 공급 전압의 실측
        # 테이블에서 가져온다.
        desired_force = self.thruster.clamp_thrust(desired_force)
        pwm = self.thruster.inverse_thrust(desired_force.unsqueeze(0)).squeeze(0)
        # T200 모델은 |pwm|<=0.075에서 추력이 정확히 0이다. 역함수는 작은
        # non-zero force를 deadband 내부 PWM으로 돌려줄 수 있으므로 그대로 보내면
        # 원하는 wrench가 전혀 만들어지지 않는다. 작은 수치 잡음은 0으로 끄고,
        # 실제로 활성화할 채널은 검증된 ±40us(정규화 0.10) 이상으로 보상한다.
        active = desired_force.abs() >= self.thruster_force_activation
        pwm_sign = torch.sign(desired_force)
        pwm = torch.where(
            active,
            pwm_sign * torch.maximum(pwm.abs(), torch.tensor(self.minimum_active_pwm, device=self.device)),
            torch.zeros_like(pwm),
        ).clamp(-1.0, 1.0)
        estimated_force, estimated_torque = self.thruster.compute(pwm.unsqueeze(0))
        estimated_wrench_zup = torch.cat(
            (estimated_force.squeeze(0), estimated_torque.squeeze(0))
        )

        return ControllerOutput(
            wrench_zup=wrench_zup,
            wrench_sname=wrench_sname,
            estimated_wrench_zup=estimated_wrench_zup,
            normalized_action_zup=(wrench_zup / self._f_max).clamp(-1.0, 1.0),
            thruster_force=desired_force,
            pwm=pwm,
        )
