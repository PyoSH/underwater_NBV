import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hydrodynamics import BROV2ThrusterModel

_PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")

# BROV2_Heavy 질량/관성텐서 (Z-up body frame, brov2_custom_physics.usda에 실제 적용된
# 값과 동일 — brov2_spec.md §2, principalAxes=identity라 비대각 성분 없음).
# 에너지 보존 체크(운동에너지)에 사용.
_MASS = 14.635
_INERTIA_BODY = torch.diag(torch.tensor([0.289, 0.329, 0.337]))


def _quat_to_euler_zyx_deg(quat: torch.Tensor) -> torch.Tensor:
    """quat: (...,4) order (w,x,y,z) -> [roll,pitch,yaw] in degrees (body 3-2-1 Euler)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = torch.clamp(2 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1) * (180.0 / torch.pi)


class TrajectoryLogger:
    """
    공통 시계열 로거 — 6-DOF 동역학 진단 플롯 설계에 따라 매 step
    위치/자세/속도/각속도/에너지/입력을 기록하고, 시간을 공유 x축으로 하는
    6-panel figure를 저장한다.
    """

    def __init__(self):
        self.t = []
        self.pos = []
        self.euler_deg = []
        self.lin_vel_b = []
        self.ang_vel_b = []
        self.energy = []
        self.action_mag = []

    def record(self, env, t: float, action: torch.Tensor) -> None:
        pos     = env._robot.data.root_pos_w[0].detach().cpu()
        quat    = env._robot.data.root_quat_w[0].detach().cpu()
        lin_vel = env._robot.data.root_lin_vel_b[0].detach().cpu()
        ang_vel = env._robot.data.root_ang_vel_b[0].detach().cpu()

        ke_trans = 0.5 * _MASS * float(lin_vel @ lin_vel)
        ke_rot   = 0.5 * float(ang_vel @ _INERTIA_BODY @ ang_vel)

        self.t.append(t)
        self.pos.append(pos.tolist())
        self.euler_deg.append(_quat_to_euler_zyx_deg(quat).tolist())
        self.lin_vel_b.append(lin_vel.tolist())
        self.ang_vel_b.append(ang_vel.tolist())
        self.energy.append(ke_trans + ke_rot)
        self.action_mag.append(float(action[0].abs().mean()))

    def plot(self, title: str, filename: str, highlight_axis: int | None = None) -> str:
        os.makedirs(_PLOT_DIR, exist_ok=True)
        save_path = os.path.join(_PLOT_DIR, filename)

        t        = np.array(self.t)
        pos      = np.array(self.pos)
        euler    = np.array(self.euler_deg)
        lin_vel  = np.array(self.lin_vel_b)
        ang_vel  = np.array(self.ang_vel_b)
        energy   = np.array(self.energy)
        act_mag  = np.array(self.action_mag)

        fig, axes = plt.subplots(6, 1, figsize=(10, 16), sharex=True)

        def plot_group(ax, data, labels, ylabel):
            for i, lbl in enumerate(labels):
                is_main = highlight_axis is not None and i == highlight_axis
                ax.plot(t, data[:, i], label=lbl,
                        linewidth=2.5 if is_main else 1.0,
                        alpha=1.0 if (highlight_axis is None or is_main) else 0.45)
            ax.set_ylabel(ylabel)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        plot_group(axes[0], pos,     ["X", "Y", "Z"],                              "Position [m]")
        plot_group(axes[1], euler,   ["roll", "pitch", "yaw"],                     "Orientation [deg]")
        plot_group(axes[2], lin_vel, ["u(surge)", "v(sway)", "w(heave)"],          "Body lin.vel [m/s]")
        plot_group(axes[3], ang_vel, ["p(roll rate)", "q(pitch rate)", "r(yaw rate)"], "Body ang.vel [rad/s]")

        axes[4].plot(t, energy, color="darkred", linewidth=1.5)
        axes[4].axhline(energy[0], color="gray", linestyle="--", linewidth=1, alpha=0.6, label="initial")
        axes[4].set_ylabel("KE trans+rot [J]")
        axes[4].legend(loc="upper right", fontsize=8)
        axes[4].grid(True, alpha=0.3)

        axes[5].fill_between(t, act_mag, step="pre", alpha=0.3, color="steelblue")
        axes[5].set_ylabel("mean |PWM|")
        axes[5].set_xlabel("time [s]")
        axes[5].grid(True, alpha=0.3)

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"     [plot] saved to {save_path}")
        return save_path


# ── 직선 이동 검증을 위한 상수 ────────────────────────────────────────────────
# backward/left/down 은 forward/right/up 명령의 부호 반전.
_DIRECTION_CMDS = {
    "forward" : torch.tensor([-1., -1.,  1.,  1.,  0.,  0.,  0.,  0.]),
    "backward": torch.tensor([ 1.,  1., -1., -1.,  0.,  0.,  0.,  0.]),
    "right"   : torch.tensor([ 1., -1.,  1., -1.,  0.,  0.,  0.,  0.]),
    "left"    : torch.tensor([-1.,  1., -1.,  1.,  0.,  0.,  0.,  0.]),
    "up"      : torch.tensor([ 0.,  0.,  0.,  0., -1., -1., -1., -1.]),
    "down"    : torch.tensor([ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.]),
}
_DIRECTION_EXPECTED = {
    "forward" : "X 증가 (전진)",
    "backward": "X 감소 (후진)",
    "right"   : "Y 감소 (우측)",
    "left"    : "Y 증가 (좌측)",
    "up"      : "Z 증가 (상승)",
    "down"    : "Z 감소 (하강)",
}
_DIRECTION_AXIS = {
    "forward" : 0,   # X
    "backward": 0,
    "right"   : 1,   # Y
    "left"    : 1,
    "up"      : 2,   # Z
    "down"    : 2,
}
# 주축 변위의 기대 부호 (+1: 증가, -1: 감소)
_DIRECTION_SIGN = {
    "forward" : +1,
    "backward": -1,
    "right"   : -1,
    "left"    : +1,
    "up"      : +1,
    "down"    : -1,
}

# ── 검증 함수 1: 중성부력 ────────────────────────────────────────────────────
def test_neutral_buoyancy(env, duration_s: float = 5.0) -> None:
    print("\n" + "=" * 60)
    print("검증 1: 중성부력 확인  (추력 = 0)")
    print("=" * 60)
    print(f"  시뮬레이션 시간  : {duration_s:.1f} s")
    print(f"  volume           : {env._volume:.6f} m³")
    print(f"  water_density    : {env._water_density:.1f} kg/m³")
    print(f"  예상 부력        : {env._water_density * 9.81 * env._volume:.2f} N\n")

    obs, _ = env.reset()
    print(f"  num_bodies  : {env._robot.num_bodies}")
    body_pos = env._robot.data.body_pos_w[0]  # [num_bodies, 3]
    print(f"  {'body':<40} {'X':>8} {'Y':>8} {'Z':>8}")
    for name, pos in zip(env._robot.body_names, body_pos):
        print(f"  {name:<40} {pos[0].item():>8.3f} {pos[1].item():>8.3f} {pos[2].item():>8.3f}")
    action = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)

    policy_dt = env.cfg.sim.dt * env.cfg.decimation
    n_steps   = int(duration_s / policy_dt)

    z_init = env._robot.data.root_pos_w[0, 2].item()
    print(f"  초기 Z 위치 : {z_init:.4f} m")
    print(f"  {'step':>6} | {'시간(s)':>7} | {'Z 위치(m)':>10} | {'ΔZ(m)':>8} | {'수직 속도(m/s)':>14}")
    print("  " + "-" * 56)

    logger = TrajectoryLogger()
    logger.record(env, 0.0, action)

    for step in range(n_steps):
        obs, _, terminated, truncated, _ = env.step(action)
        t = (step + 1) * policy_dt
        logger.record(env, t, action)

        if (step + 1) % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            z_now  = env._robot.data.root_pos_w[0, 2].item()
            vz     = env._robot.data.root_lin_vel_b[0, 2].item()
            dz     = z_now - z_init
            print(f"  {step+1:>6} | {t:>7.2f} | {z_now:>10.4f} | {dz:>+8.4f} | {vz:>14.4f}")

        if terminated.any() or truncated.any():
            print("  [경고] 에피소드 종료 (경계 이탈 또는 시간 초과)")
            break

    z_final = env._robot.data.root_pos_w[0, 2].item()
    dz_total = z_final - z_init
    print(f"\n  총 Z 변화 : {dz_total:+.4f} m  →  ", end="")
    if abs(dz_total) < 0.1:
        print("✓ 중성부력 양호")
    elif dz_total < 0:
        print("✗ 로봇 하강 → brov2_heavy.yaml의 volume 증가 필요")
    else:
        print("✗ 로봇 상승 → brov2_heavy.yaml의 volume 감소 필요")

    logger.plot("Neutral Buoyancy (thrust=0)", "neutral_buoyancy.png", highlight_axis=2)


# ── 검증 함수 2: 직선 이동 ────────────────────────────────────────────────────
def test_straight_line(env, thrust: float = 0.5, duration_s: float = 3.0,
                        directions: list[str] | None = None) -> None:
    """directions 기본값은 ["forward"]만 — 화살표 등 디버그 시각화를 빠르게 확인할 때는
    한 방향만 도는 게 편해서. 6방향 전부 돌리려면 directions=list(_DIRECTION_CMDS)로 호출."""
    if directions is None:
        directions = ["forward"]

    print("\n" + "=" * 60)
    print("검증 2: 직선 이동 확인")
    print("=" * 60)
    print(f"  PWM 크기 : {thrust:.2f}  /  각 방향 시뮬 시간 : {duration_s:.1f} s  /  방향: {directions}")

    policy_dt = env.cfg.sim.dt * env.cfg.decimation
    n_steps   = int(duration_s / policy_dt)

    for direction in directions:
        cmd_template = _DIRECTION_CMDS[direction]
        print(f"\n  ── {direction.upper()}  (예상: {_DIRECTION_EXPECTED[direction]}) ──")
        obs, _ = env.reset()
        pos_init = env._robot.data.root_pos_w[0].clone()
        print(f"     초기 위치 : X={pos_init[0]:.3f}  Y={pos_init[1]:.3f}  Z={pos_init[2]:.3f}")

        action = (cmd_template * thrust).unsqueeze(0).expand(env.num_envs, -1).to(env.device)

        logger = TrajectoryLogger()
        logger.record(env, 0.0, action)

        for step in range(n_steps):
            env.step(action)
            logger.record(env, (step + 1) * policy_dt, action)

        pos_final = env._robot.data.root_pos_w[0]
        disp      = pos_final - pos_init
        axis      = _DIRECTION_AXIS[direction]

        print(f"     최종 위치 : X={pos_final[0]:.3f}  Y={pos_final[1]:.3f}  Z={pos_final[2]:.3f}")
        print(f"     변위      : ΔX={disp[0]:+.3f}  ΔY={disp[1]:+.3f}  ΔZ={disp[2]:+.3f}")

        main_disp  = disp[axis].item()
        drift_axes = [i for i in range(3) if i != axis]
        drift      = (disp[drift_axes[0]]**2 + disp[drift_axes[1]]**2).sqrt().item()

        print(f"     주축 변위 : {main_disp:+.3f} m  /  횡방향 표류 : {drift:.3f} m", end="  →  ")

        ok = main_disp * _DIRECTION_SIGN[direction] > 0.05
        if ok and drift < abs(main_disp) * 0.5:
            print("✓ 방향 정상")
        elif not ok:
            print("✗ 방향 반대 또는 무반응")
        else:
            print("△ 방향은 맞으나 표류 과다")

        logger.plot(f"Straight Line: {direction}", f"straight_line_{direction}.png", highlight_axis=axis)


# ── 검증 함수 3: 회전 (yaw/roll/pitch) ────────────────────────────────────────
# 명령 조합은 hydrodynamics.py의 실제 _POS/_DIR(SNAME frame)로 thruster allocation을
# 풀어서 구한 값 (최소자승, 교차축 성분 ~1e-16 수준으로 확인됨). T1~T4는 yaw만,
# T5~T8은 roll/pitch만 담당하도록 그룹이 이미 분리되어 있어 두 그룹을 독립적으로 풀었음.
_ROTATION_CMDS = {
    "yaw"  : torch.tensor([ 1., -1., -1.,  1.,  0.,  0.,  0.,  0.]),
    "roll" : torch.tensor([ 0.,  0.,  0.,  0.,  1., -1.,  1., -1.]),
    "pitch": torch.tensor([ 0.,  0.,  0.,  0., -1., -1.,  1.,  1.]),
}
# IsaacLab Z-up body frame 기준 (root_ang_vel_b 인덱스): X=roll, Y=pitch, Z=yaw
_ROTATION_AXIS = {
    "yaw"  : 2,
    "roll" : 0,
    "pitch": 1,
}


def test_rotation(env, thrust: float = 0.3, duration_s: float = 3.0) -> None:
    print("\n" + "=" * 60)
    print("검증 3: 회전 (yaw / roll / pitch) 확인")
    print("=" * 60)
    print(f"  PWM 크기 : {thrust:.2f}  /  각 축 시뮬 시간 : {duration_s:.1f} s")

    policy_dt = env.cfg.sim.dt * env.cfg.decimation
    n_steps   = int(duration_s / policy_dt)

    for axis_name, cmd_template in _ROTATION_CMDS.items():
        print(f"\n  ── {axis_name.upper()} ──")
        obs, _ = env.reset()

        action = (cmd_template * thrust).unsqueeze(0).expand(env.num_envs, -1).to(env.device)

        logger = TrajectoryLogger()
        logger.record(env, 0.0, action)

        for step in range(n_steps):
            env.step(action)
            logger.record(env, (step + 1) * policy_dt, action)

        ang_vel = env._robot.data.root_ang_vel_b[0]  # (3,) body frame [roll, pitch, yaw]
        print(f"     최종 각속도 (body, rad/s) : X(roll)={ang_vel[0]:+.4f}  "
              f"Y(pitch)={ang_vel[1]:+.4f}  Z(yaw)={ang_vel[2]:+.4f}")

        axis = _ROTATION_AXIS[axis_name]
        main_rate  = ang_vel[axis].item()
        cross_axes = [i for i in range(3) if i != axis]
        cross      = (ang_vel[cross_axes[0]] ** 2 + ang_vel[cross_axes[1]] ** 2).sqrt().item()

        print(f"     주축 각속도 : {main_rate:+.4f} rad/s  /  교차축 각속도 : {cross:.4f} rad/s", end="  →  ")
        if abs(main_rate) < 0.02:
            print("✗ 반응 없음")
        elif cross < abs(main_rate) * 0.5:
            print("✓ 회전축 정상")
        else:
            print("△ 축은 맞으나 교차축 과다")

        logger.plot(f"Rotation: {axis_name}", f"rotation_{axis_name}.png", highlight_axis=axis)


# ── 검증 함수 3.5: 6자유도 종합 (직선 6방향 + 회전 3축) ────────────────────────
def test_six_dof(env, thrust: float = 0.5, rotation_thrust: float = 0.3,
                  duration_s: float = 3.0) -> None:
    """surge/sway/heave(직선 6방향)와 roll/pitch/yaw(회전 3축)를 순서대로 전부 검증한다.
    개별 방향/축 판정은 test_straight_line / test_rotation과 동일한 기준을 그대로 쓴다."""
    test_straight_line(
        env, thrust=thrust, duration_s=duration_s,
        directions=list(_DIRECTION_CMDS),   # forward/backward/right/left/up/down 전부
    )
    test_rotation(env, thrust=rotation_thrust, duration_s=duration_s)


# ── 검증 함수 4: 추진기 모델 ──────────────────────────────────────────────────
def test_thruster_model(env, duration_s: float = 2.0) -> None:
    print("\n" + "=" * 60)
    print("검증 4: 추진기 모델 입출력 확인")
    print("=" * 60)

    # [Part A] 정적 변환표
    print("\n  [Part A] 정적 PWM → RPM → Thrust 변환표\n")
    print(f"  {'PWM':>6} | {'RPM (approx)':>14} | {'Thrust [N]':>12}")
    print("  " + "-" * 38)

    model = BROV2ThrusterModel(num_envs=1, dt=1.0, device="cpu")
    db = model._DEADBAND

    for pwm_val in [-1.0, -0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        model._pwm_state.zero_()
        cmd = torch.zeros((1, 8))
        cmd[0, 0] = pwm_val   # T1 채널 하나만 활성화 (8개 전부 켜면 T1~T4 수평 성분이 서로 상쇄됨)
        for _ in range(50): model.compute(cmd)
        f, _ = model.compute(cmd)
        thrust_mag = f[0].norm().item()   # 벡터 크기 사용 — 축/부호 관례에 무관하게 |thrust|와 정확히 일치

        pwm_s = model._pwm_state[0, 0].item()
        rpm_est = (3659.9 * pwm_s + 345.21) if pwm_s > db else (3494.4 * pwm_s - 433.50) if pwm_s < -db else 0.0
        print(f"  {pwm_val:>6.2f} | {rpm_est:>14.1f} | {thrust_mag:>12.4f}")

    # [Part B] 시뮬레이션 램프
    print(f"\n  [Part B] 시뮬레이션 T1 PWM 선형 램프 (0 → 1)")
    print(f"  {'시간(s)':>7} | {'PWM':>5} | {'X 속도(m/s)':>12} | {'X 가속도 추정':>14}")
    print("  " + "-" * 46)

    env.reset()
    policy_dt = env.cfg.sim.dt * env.cfg.decimation
    n_steps   = int(duration_s / policy_dt)
    prev_vx   = env._robot.data.root_lin_vel_b[0, 0].item()

    for step in range(n_steps):
        t = (step + 1) * policy_dt
        pwm = min(1.0, step / max(n_steps - 1, 1))
        action = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
        action[:, 0] = pwm
        env.step(action)

        vx = env._robot.data.root_lin_vel_b[0, 0].item()
        ax_est = (vx - prev_vx) / policy_dt
        prev_vx = vx

        if (step + 1) % max(1, n_steps // 8) == 0:
            print(f"  {t:>7.2f} | {pwm:>5.2f} | {vx:>12.4f} | {ax_est:>14.4f}")
