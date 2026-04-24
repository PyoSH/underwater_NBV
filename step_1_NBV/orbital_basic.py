"""
orbital_policy.py — env.py _apply_action() 기반 궤도 순회
===========================================================

액션 구조 (action_space=9, one-hot argmax 방식):
    [0] θ+  [1] θ-  [2] φ+  [3] φ-  [4] ψ+  [5] ψ-
    [6] light-  [7] light 유지  [8] light+

좌표계 (env.py 실제 구현 기준):
    x = ψ * sin(φ) * cos(θ)
    y = ψ * sin(φ) * sin(θ)
    z = ψ * cos(φ)

    φ=0   → 정수직 위 (z축 최대)
    φ=90° → xy평면 (수평)
    
궤도 설계:
    4개 φ 레벨에서 θ를 0→2π 균등 순회
    각 레벨 내 ψ와 φ는 고정, θ만 delta_theta씩 증가
"""

import math
import torch
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 액션 인덱스 상수 (가독성)
# ─────────────────────────────────────────────────────────────────────────────
# ACT_THETA_POS = 0   # θ 증가
# ACT_THETA_NEG = 1   # θ 감소
# ACT_PHI_POS   = 2   # φ 증가 (수평에 가까워짐)
# ACT_PHI_NEG   = 3   # φ 감소 (수직에 가까워짐)
# ACT_PSI_POS   = 4   # ψ 증가 (멀어짐)
# ACT_PSI_NEG   = 5   # ψ 감소 (가까워짐)
# ACT_LIGHT_DEC = 6   # 조명 감소
# ACT_LIGHT_HLD = 7   # 조명 유지
# ACT_LIGHT_INC = 8   # 조명 증가


def _one_hot_action(idx: int, action_space: int, num_envs: int,
                    device: str) -> torch.Tensor:
    """
    argmax로 선택될 인덱스를 가장 큰 값(1.0)으로 설정한 액션 텐서 반환.
    나머지는 0.0으로, argmax가 항상 idx를 가리킨다.
    """
    a = torch.zeros(num_envs, action_space, device=device)
    a[:, idx] = 1.0
    return a


# ─────────────────────────────────────────────────────────────────────────────
# 궤도 플래너
# ─────────────────────────────────────────────────────────────────────────────

def plan_orbital_trajectory(
    phi_levels_deg: list[float],    # 4개 φ 값 (polar angle, 도 단위)
    steps_per_level: int,           # 레벨당 θ 스텝 수
    target_psi: float,              # 목표 거리
    delta_theta_deg: float,         # env cfg의 delta_theta (도 단위)
    delta_phi_deg: float,           # env cfg의 delta_phi (도 단위)
    delta_psi: float,               # env cfg의 delta_psi
    init_theta_deg: float = 0.0,    # reset 후 초기 θ (env._reset_idx 참고: 0.0)
    init_phi_deg: float = 89.0,     # reset 후 초기 φ (env._reset_idx 참고: 89°)
    init_psi: float = 1.0,          # reset 후 초기 ψ (env._reset_idx 참고: 1.0)
) -> list[dict]:
    """
    각 웨이포인트를 "몇 번 어떤 액션을 눌러야 하는가"의 시퀀스로 반환한다.

    env._apply_action()은 매 스텝 delta만큼 이동하므로,
    목표 각도까지 가려면 (목표 - 현재) / delta 스텝이 필요하다.

    Returns:
        sequence: [{"action_idx": int, "repeat": int, "label": str}, ...]
        최종 도달 상태: (theta, phi, psi) for each level
    """
    sequence = []

    # 현재 상태 추적 (reset 기준값)
    curr_theta = math.radians(init_theta_deg)
    curr_phi   = math.radians(init_phi_deg)
    curr_psi   = init_psi

    dt = math.radians(delta_theta_deg)
    dp = math.radians(delta_phi_deg)

    def steps_needed(curr, target, delta):
        diff = target - curr
        if abs(diff) < 1e-9:
            return 0, 0   # (positive_steps, negative_steps)
        if diff > 0:
            return math.ceil(diff / delta), 0
        else:
            return 0, math.ceil(-diff / delta)

    # θ 한 바퀴 스텝 수 (steps_per_level에 맞게 theta_step_deg 결정)
    theta_step_deg = 360.0 / steps_per_level     # 예: 12 스텝이면 30°/스텝
    theta_step_rad = math.radians(theta_step_deg)
    # delta_theta로 몇 번 눌러야 theta_step_rad 이동하는지
    clicks_per_theta_step = max(1, round(theta_step_rad / dt))

    for level_idx, phi_deg in enumerate(phi_levels_deg):
        target_phi = math.radians(phi_deg)
        target_psi_val = target_psi

        # ── 1단계: ψ 조정 (현재 psi → target_psi) ──────────────────────────
        diff_psi = target_psi_val - curr_psi
        if abs(diff_psi) > 1e-6:
            psi_clicks = abs(round(diff_psi / delta_psi))
            act_idx    = ACT_PSI_POS if diff_psi > 0 else ACT_PSI_NEG
            sequence.append({
                "action_idx": act_idx,
                "repeat"    : psi_clicks,
                "label"     : f"L{level_idx+1} ψ 조정 → {target_psi_val:.2f}",
            })
            curr_psi = target_psi_val

        # ── 2단계: φ 조정 (현재 phi → target_phi) ──────────────────────────
        diff_phi = target_phi - curr_phi
        if abs(diff_phi) > 1e-9:
            phi_clicks = max(1, abs(round(diff_phi / dp)))
            act_idx    = ACT_PHI_POS if diff_phi > 0 else ACT_PHI_NEG
            sequence.append({
                "action_idx": act_idx,
                "repeat"    : phi_clicks,
                "label"     : f"L{level_idx+1} φ 조정 {math.degrees(curr_phi):.1f}°→{phi_deg:.1f}°",
            })
            curr_phi = target_phi

        # ── 3단계: θ 순회 (0→360°, steps_per_level 스텝) ───────────────────
        for step in range(steps_per_level):
            sequence.append({
                "action_idx": ACT_THETA_POS,
                "repeat"    : clicks_per_theta_step,
                "label"     : (f"L{level_idx+1} orbit step {step+1}/{steps_per_level} "
                               f"θ≈{math.degrees(curr_theta):.1f}°"),
            })
            curr_theta = (curr_theta + theta_step_rad) % (2 * math.pi)

    return sequence


# ─────────────────────────────────────────────────────────────────────────────
# 메인 실행 루프
# ─────────────────────────────────────────────────────────────────────────────

def run_orbital_policy(
    env,
    # ── 궤도 설정 ──────────────────────────────────────────────────────────
    phi_levels_deg: list[float] = [75.0, 55.0, 35.0, 15.0],
    #   φ는 polar angle: 75°=수평에 가까움(낮은 고도), 15°=거의 수직 위(높은 고도)
    #   위에서 아래로 내려오는 순서
    steps_per_level: int = 12,       # 레벨당 viewpoint 수 (30° 간격)
    target_psi: float = 3.0,         # 타겟까지 거리 (m) — envCfg.psi_min/max 범위 내
    # ── env cfg delta 값 (envCfg와 일치해야 함) ────────────────────────────
    delta_theta_deg: float = 10.0,   # envCfg.delta_theta를 도로 입력
    delta_phi_deg: float = 5.0,      # envCfg.delta_phi를 도로 입력
    delta_psi: float = 0.1,          # envCfg.delta_psi
    # ── 촬영 설정 ──────────────────────────────────────────────────────────
    settle_steps: int = 1,           # 각 이동 후 안정화 스텝 수
    capture_only_at_waypoint: bool = True,  # True: 순회 스텝에서만 캡처
    on_capture: callable = None,     # (step_info) → None 콜백
) -> list[dict]:
    """
    env.py의 _apply_action() one-hot argmax 방식에 맞춰
    4개 φ 레벨에서 θ를 순회하며 이미지를 수집한다.

    Reset 후 초기 상태: θ=0, φ=89°, ψ=1.0 (env._reset_idx 참고)
    """

    # 시퀀스 계획
    sequence = plan_orbital_trajectory(
        phi_levels_deg   = phi_levels_deg,
        steps_per_level  = steps_per_level,
        target_psi       = target_psi,
        delta_theta_deg  = delta_theta_deg,
        delta_phi_deg    = delta_phi_deg,
        delta_psi        = delta_psi,
        init_theta_deg   = 0.0,
        init_phi_deg     = 89.0,
        init_psi         = 1.0,
    )

    total_waypoints = len(phi_levels_deg) * steps_per_level
    print(f"[orbital] 레벨={len(phi_levels_deg)}, 스텝/레벨={steps_per_level} "
          f"→ 총 {total_waypoints}개 viewpoint", flush=True)

    obs, _ = env.reset()
    captures = []
    global_step = 0

    def _step_action(action_idx: int, repeat: int, is_capture_step: bool):
        nonlocal obs, global_step
        a = _one_hot_action(action_idx, env.cfg.action_space,
                            env.num_envs, env.device)
        for _ in range(repeat):
            obs, reward, terminated, truncated, info = env.step(a)
            global_step += 1
            if terminated.any() or truncated.any():
                obs, _ = env.reset()

        # settle
        hold = _one_hot_action(ACT_LIGHT_HLD, env.cfg.action_space,
                               env.num_envs, env.device)
        for _ in range(settle_steps):
            obs, reward, terminated, truncated, info = env.step(hold)

        if is_capture_step:
            _capture(reward)

    def _capture(reward):
        cam_out = env._camera.data.output
        uw_rgb  = cam_out.get("uw_rgb", None)
        uw_mean = (uw_rgb[0, :, :, :3].float().mean().item()
                   if uw_rgb is not None else float("nan"))

        sph = {
            "theta_deg": math.degrees(env._sph_theta[0].item()),
            "phi_deg"  : math.degrees(env._sph_phi[0].item()),
            "psi"      : env._sph_psi[0].item(),
        }
        cam_pos = env.cam_pos[0].cpu().numpy()
        step_info = {
            "global_step": global_step,
            "sph"        : sph,
            "cam_pos"    : cam_pos.tolist(),
            "uw_mean"    : uw_mean,
            "reward"     : reward[0].item(),
            "uw_rgb"     : uw_rgb[0].cpu() if uw_rgb is not None else None,
        }
        captures.append(step_info)
        if on_capture:
            on_capture(step_info)

        print(
            f"[capture {len(captures):3d}/{total_waypoints}] "
            f"θ={sph['theta_deg']:6.1f}° φ={sph['phi_deg']:5.1f}° ψ={sph['psi']:.2f} "
            f"cam=({cam_pos[0]:.2f},{cam_pos[1]:.2f},{cam_pos[2]:.2f}) "
            f"uw={uw_mean:.3f}",
            flush=True,
        )

    # 시퀀스 실행
    for seg in sequence:
        is_orbit_step = "orbit step" in seg["label"]
        is_capture    = capture_only_at_waypoint and is_orbit_step
        _step_action(seg["action_idx"], seg["repeat"], is_capture_step=is_capture)

    print(f"[orbital] 완료: {len(captures)}개 viewpoint 수집", flush=True)
    return captures
    
def _deg(v):
    if isinstance(v, torch.Tensor): v = v.item()
    return round(math.degrees(float(v)), 2)

def _r(v, d=4):
    if isinstance(v, torch.Tensor): v = v.item()
    return round(float(v), d)

def _sph_to_cart(psi, phi, theta):
    x = psi * math.sin(phi) * math.cos(theta)
    y = psi * math.sin(phi) * math.sin(theta)
    z = psi * math.cos(phi)
    return x, y, z

ACT_NAMES = {
    0:"θ+", 1:"θ−", 2:"φ+", 3:"φ−",
    4:"ψ+", 5:"ψ−", 6:"L−", 7:"L±0", 8:"L+",
}


# ─────────────────────────────────────────────────────────────────────────────
# 현재 env 상태 스냅샷
# ─────────────────────────────────────────────────────────────────────────────

def print_env_state(env, tag="") -> dict:
    e = 0
    theta = env._sph_theta[e].item()
    phi   = env._sph_phi[e].item()
    psi   = env._sph_psi[e].item()
    rock  = env.rock_pos[e].cpu().numpy()

    tx, ty, tz   = _sph_to_cart(psi, phi, theta)
    theory_pos   = np.array([rock[0]+tx, rock[1]+ty, rock[2]+tz])
    actual_pos   = env.cam_pos[e].cpu().numpy()
    actual_quat  = env.cam_orient[e].cpu().numpy()
    err          = theory_pos - actual_pos
    err_norm     = np.linalg.norm(err)
    actual_dist  = np.linalg.norm(actual_pos - rock)
    fwd          = rock - actual_pos
    fwd_norm     = fwd / (np.linalg.norm(fwd) + 1e-8)
    light        = env._light_level[e].item()
    coverage     = env.curr_coverage[e].item() if hasattr(env, 'curr_coverage') else float('nan')

    phi_clamped = (phi > env.cfg.phi_max + 1e-4) or (phi < env.cfg.phi_min - 1e-4)

    print(f"\n{SEPARATOR}")
    print(f"[STATE{' | '+tag if tag else ''}]")
    print(f"{SEPARATOR}")
    print(f"  구면좌표 (env 내부)")
    print(f"    θ = {_deg(theta):8.2f}°   ({_r(theta):.4f} rad)")
    print(f"    φ = {_deg(phi):8.2f}°   ({_r(phi):.4f} rad)  "
          f"[허용: {_deg(env.cfg.phi_min):.0f}°~{_deg(env.cfg.phi_max):.0f}°]"
          f"{'  !! clamp 밖' if phi_clamped else ''}")
    print(f"    ψ = {_r(psi):8.4f} m              "
          f"[허용: {env.cfg.psi_min:.1f}~{env.cfg.psi_max:.1f}]")
    print(f"    light_level = {light}  coverage = {coverage:.4f}  "
          f"[terminal >= {env.cfg.coverage_terminal}]"
          f"{'  !! TERMINAL' if coverage >= env.cfg.coverage_terminal else ''}")
    print(f"")
    print(f"  이론 cam_pos  [{theory_pos[0]:8.4f}, {theory_pos[1]:8.4f}, {theory_pos[2]:8.4f}]")
    print(f"  실제 cam_pos  [{actual_pos[0]:8.4f}, {actual_pos[1]:8.4f}, {actual_pos[2]:8.4f}]")
    print(f"  오차          norm={err_norm:.5f} m")
    print(f"  실제거리      {actual_dist:.4f} m  (ψ={_r(psi):.4f})")
    print(f"  forward       [{fwd_norm[0]:.4f}, {fwd_norm[1]:.4f}, {fwd_norm[2]:.4f}]")
    print(f"  quat[w,x,y,z] [{actual_quat[0]:.4f}, {actual_quat[1]:.4f}, "
          f"{actual_quat[2]:.4f}, {actual_quat[3]:.4f}]")
    print(f"{SEPARATOR}\n")

    return dict(theta=theta, phi=phi, psi=psi,
                theory_pos=theory_pos, actual_pos=actual_pos,
                err_norm=err_norm, coverage=coverage, tag=tag)


# ─────────────────────────────────────────────────────────────────────────────
# action 전후 비교  (수정: reset 감지 + 판정 버그 수정)
# ─────────────────────────────────────────────────────────────────────────────

def print_action_effect(env, action_idx: int, repeat: int = 1) -> dict:
    e = 0

    # step 전 상태 저장
    before = dict(
        theta = env._sph_theta[e].item(),
        phi   = env._sph_phi[e].item(),
        psi   = env._sph_psi[e].item(),
        pos   = env.cam_pos[e].cpu().numpy().copy(),
        light = env._light_level[e].item(),
        cov   = env.curr_coverage[e].item() if hasattr(env, 'curr_coverage') else 0.0,
    )

    a = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)
    a[:, action_idx] = 1.0

    reset_triggered = False
    for i in range(repeat):
        obs, reward, terminated, truncated, _ = env.step(a)
        if terminated.any() or truncated.any():
            reset_triggered = True
            print(f"  !! env.step {i+1}회차에서 terminated/truncated 발생 → reset 트리거됨")
            print(f"     coverage={env.curr_coverage[e].item():.4f}  "
                  f"reward={reward[e].item():.4f}")
            break

    after = dict(
        theta = env._sph_theta[e].item(),
        phi   = env._sph_phi[e].item(),
        psi   = env._sph_psi[e].item(),
        pos   = env.cam_pos[e].cpu().numpy().copy(),
        light = env._light_level[e].item(),
        cov   = env.curr_coverage[e].item() if hasattr(env, 'curr_coverage') else 0.0,
    )

    dtheta = after["theta"] - before["theta"]
    dphi   = after["phi"]   - before["phi"]
    dpsi   = after["psi"]   - before["psi"]
    dpos   = after["pos"]   - before["pos"]
    dlight = after["light"] - before["light"]

    # decimation=1 확인됨
    dec = env.cfg.decimation
    # 기대 delta (decimation × repeat 적용)
    exp = {
        0: ("θ", +env.cfg.delta_theta * dec * repeat),
        1: ("θ", -env.cfg.delta_theta * dec * repeat),
        2: ("φ", +env.cfg.delta_phi   * dec * repeat),
        3: ("φ", -env.cfg.delta_phi   * dec * repeat),
        4: ("ψ", +env.cfg.delta_psi   * dec * repeat),
        5: ("ψ", -env.cfg.delta_psi   * dec * repeat),
        6: ("L", -1 * repeat),
        7: ("L",  0),
        8: ("L", +1 * repeat),
    }
    exp_axis, exp_val = exp[action_idx]

    # ── 수정된 판정: 각 축에 대해 독립적으로 체크 ──────────────────────────
    tol_ang = math.degrees(env.cfg.delta_theta) * 0.1  # 허용 오차 = delta의 10%
    tol_psi = env.cfg.delta_psi * 0.1

    def judge(axis_name, actual_d, expected_if_target):
        if reset_triggered:
            return "RESET"
        diff = abs(actual_d - expected_if_target)
        tol = tol_ang if axis_name in ("θ", "φ") else tol_psi
        return "OK" if diff < tol else "NG"

    exp_theta = math.degrees(exp_val) if exp_axis == "θ" else 0.0
    exp_phi   = math.degrees(exp_val) if exp_axis == "φ" else 0.0
    exp_psi   = exp_val               if exp_axis == "ψ" else 0.0
    exp_light = int(exp_val)          if exp_axis == "L" else 0

    print(f"\n{SEPARATOR}")
    print(f"[ACTION EFFECT]  act[{action_idx}]={ACT_NAMES[action_idx]}  x{repeat}회  "
          f"(decimation={dec})"
          + ("  !! RESET 발생 → 아래 값은 reset 후 초기값" if reset_triggered else ""))
    print(f"{SEPARATOR}")
    print(f"  {'':10s}  {'before':>10s}  {'after':>10s}  {'actual Δ':>10s}  "
          f"{'expect Δ':>10s}  {'판정':>6s}")
    print(f"  {'θ (deg)':10s}  {_deg(before['theta']):>10.2f}  {_deg(after['theta']):>10.2f}  "
          f"{math.degrees(dtheta):>+10.2f}  {exp_theta:>+10.2f}  "
          f"{judge('θ', math.degrees(dtheta), exp_theta):>6s}")
    print(f"  {'φ (deg)':10s}  {_deg(before['phi']):>10.2f}  {_deg(after['phi']):>10.2f}  "
          f"{math.degrees(dphi):>+10.2f}  {exp_phi:>+10.2f}  "
          f"{judge('φ', math.degrees(dphi), exp_phi):>6s}")
    print(f"  {'ψ (m)':10s}  {_r(before['psi']):>10.4f}  {_r(after['psi']):>10.4f}  "
          f"{dpsi:>+10.4f}  {exp_psi:>+10.4f}  "
          f"{judge('ψ', dpsi, exp_psi):>6s}")
    print(f"  {'light':10s}  {before['light']:>10d}  {after['light']:>10d}  "
          f"{dlight:>+10d}  {exp_light:>+10d}  "
          f"{'RESET' if reset_triggered else ('OK' if dlight==exp_light else 'NG'):>6s}")
    print(f"  {'coverage':10s}  {before['cov']:>10.4f}  {after['cov']:>10.4f}  "
          f"{after['cov']-before['cov']:>+10.4f}")
    print(f"")
    print(f"  cam_pos Δ  [{dpos[0]:+.4f}, {dpos[1]:+.4f}, {dpos[2]:+.4f}]  "
          f"norm={np.linalg.norm(dpos):.4f} m")
    if reset_triggered:
        print(f"\n  !! 근본 원인: reset 직후 coverage가 이미 terminal({env.cfg.coverage_terminal}) 이상")
        print(f"     → _voxelize_gt_mesh 또는 TSDF 초기화 확인 필요")
    print(f"{SEPARATOR}\n")

    return dict(before=before, after=after, reset_triggered=reset_triggered,
                dtheta=dtheta, dphi=dphi, dpsi=dpsi)