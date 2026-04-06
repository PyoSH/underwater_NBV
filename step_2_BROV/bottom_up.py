import torch
from hydrodynamics import BROV2ThrusterModel

# ── 직선 이동 검증을 위한 상수 ────────────────────────────────────────────────
_DIRECTION_CMDS = {
    "forward": torch.tensor([ 1.,  1., -1., -1.,  0.,  0.,  0.,  0.]),
    "right"  : torch.tensor([ 1., -1.,  1., -1.,  0.,  0.,  0.,  0.]),
    "up"     : torch.tensor([ 0.,  0.,  0.,  0., -1., -1., -1., -1.]),
}
_DIRECTION_EXPECTED = {
    "forward": "X 증가 (전진)",
    "right"  : "Y 감소 (우측)",
    "up"     : "Z 증가 (상승)",
}
_DIRECTION_AXIS = {
    "forward": 0,   # X
    "right"  : 1,   # Y (음수 방향)
    "up"     : 2,   # Z
}

# ── 검증 함수 1: 중성부력 ────────────────────────────────────────────────────
def test_neutral_buoyancy(env, duration_s: float = 5.0) -> None:
    print("\n" + "=" * 60)
    print("검증 1: 중성부력 확인  (추력 = 0)")
    print("=" * 60)
    print(f"  시뮬레이션 시간  : {duration_s:.1f} s")
    print(f"  volume           : {env.cfg.volume:.6f} m³")
    print(f"  water_density    : {env.cfg.water_density:.1f} kg/m³")
    print(f"  예상 부력        : {env.cfg.water_density * 9.81 * env.cfg.volume:.2f} N\n")

    obs, _ = env.reset()
    action = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)

    policy_dt = env.cfg.sim.dt * env.cfg.decimation
    n_steps   = int(duration_s / policy_dt)

    z_init = env._robot.data.root_pos_w[0, 2].item()
    print(f"  초기 Z 위치 : {z_init:.4f} m")
    print(f"  {'step':>6} | {'시간(s)':>7} | {'Z 위치(m)':>10} | {'ΔZ(m)':>8} | {'수직 속도(m/s)':>14}")
    print("  " + "-" * 56)

    for step in range(n_steps):
        obs, _, terminated, truncated, _ = env.step(action)
        t = (step + 1) * policy_dt

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
        print("✗ 로봇 하강 → envCfg.volume 증가 필요")
    else:
        print("✗ 로봇 상승 → envCfg.volume 감소 필요")


# ── 검증 함수 2: 직선 이동 ────────────────────────────────────────────────────
def test_straight_line(env, thrust: float = 0.5, duration_s: float = 3.0) -> None:
    print("\n" + "=" * 60)
    print("검증 2: 직선 이동 확인")
    print("=" * 60)
    print(f"  PWM 크기 : {thrust:.2f}  /  각 방향 시뮬 시간 : {duration_s:.1f} s")

    policy_dt = env.cfg.sim.dt * env.cfg.decimation
    n_steps   = int(duration_s / policy_dt)

    for direction, cmd_template in _DIRECTION_CMDS.items():
        print(f"\n  ── {direction.upper()}  (예상: {_DIRECTION_EXPECTED[direction]}) ──")
        obs, _ = env.reset()
        pos_init = env._robot.data.root_pos_w[0].clone()
        print(f"     초기 위치 : X={pos_init[0]:.3f}  Y={pos_init[1]:.3f}  Z={pos_init[2]:.3f}")

        action = (cmd_template * thrust).unsqueeze(0).expand(env.num_envs, -1).to(env.device)

        for step in range(n_steps):
            env.step(action)

        pos_final = env._robot.data.root_pos_w[0]
        disp      = pos_final - pos_init
        axis      = _DIRECTION_AXIS[direction]

        print(f"     최종 위치 : X={pos_final[0]:.3f}  Y={pos_final[1]:.3f}  Z={pos_final[2]:.3f}")
        print(f"     변위      : ΔX={disp[0]:+.3f}  ΔY={disp[1]:+.3f}  ΔZ={disp[2]:+.3f}")

        main_disp  = disp[axis].item()
        drift_axes = [i for i in range(3) if i != axis]
        drift      = (disp[drift_axes[0]]**2 + disp[drift_axes[1]]**2).sqrt().item()

        print(f"     주축 변위 : {main_disp:+.3f} m  /  횡방향 표류 : {drift:.3f} m", end="  →  ")

        ok = main_disp < -0.05 if direction == "right" else main_disp > 0.05
        if ok and drift < abs(main_disp) * 0.5:
            print("✓ 방향 정상")
        elif not ok:
            print("✗ 방향 반대 또는 무반응")
        else:
            print("△ 방향은 맞으나 표류 과다")


# ── 검증 함수 3: 추진기 모델 ──────────────────────────────────────────────────
def test_thruster_model(env, duration_s: float = 2.0) -> None:
    print("\n" + "=" * 60)
    print("검증 3: 추진기 모델 입출력 확인")
    print("=" * 60)

    # [Part A] 정적 변환표
    print("\n  [Part A] 정적 PWM → RPM → Thrust 변환표\n")
    print(f"  {'PWM':>6} | {'RPM (approx)':>14} | {'Thrust [N]':>12}")
    print("  " + "-" * 38)

    model = BROV2ThrusterModel(num_envs=1, dt=1.0, device="cpu")
    db = model._DEADBAND

    for pwm_val in [-1.0, -0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        model._pwm_state.zero_()
        cmd = torch.full((1, 8), pwm_val)
        for _ in range(50): model.compute(cmd)
        f, _ = model.compute(cmd)
        thrust_mag = f[0, 0].item() / 0.7071

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