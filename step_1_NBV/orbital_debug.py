"""
orbital_debug.py — 궤도 순회 디버깅 모듈
==========================================
주요 기능:
  1. print_env_state()     : 현재 env 내부 상태 전체 스냅샷
  2. print_action_effect() : action 전후 상태 비교 (실제 delta 확인)
  3. print_orbital_diff()  : 목표 궤도 vs 실제 cam_pos 오차 분석
  4. run_orbital_debug()   : 궤도 순회 + 매 스텝 전체 비교 출력
"""

import math
import torch
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _r(val, decimals=4):
    """간결한 소수 출력."""
    if isinstance(val, (torch.Tensor,)):
        val = val.item()
    return round(float(val), decimals)

def _deg(rad_val):
    if isinstance(rad_val, torch.Tensor):
        rad_val = rad_val.item()
    return round(math.degrees(float(rad_val)), 2)

def _sph_to_cart(psi, phi, theta):
    """env._apply_action()과 동일한 구면→직교 변환."""
    x = psi * math.sin(phi) * math.cos(theta)
    y = psi * math.sin(phi) * math.sin(theta)
    z = psi * math.cos(phi)
    return x, y, z

SEPARATOR = "─" * 72


# ─────────────────────────────────────────────────────────────────────────────
# 1. 현재 env 상태 스냅샷
# ─────────────────────────────────────────────────────────────────────────────

def print_env_state(env, tag: str = "") -> dict:
    """
    env의 현재 구면좌표, 실제 cam_pos, 이론 cam_pos, 오차를 모두 출력.
    반환값: 스냅샷 dict (비교 저장용)
    """
    e = 0  # env index 0 기준

    # ── 구면좌표 (env 내부 상태) ──────────────────────────────────────────
    theta = env._sph_theta[e].item()
    phi   = env._sph_phi[e].item()
    psi   = env._sph_psi[e].item()

    # ── 이론 카메라 위치 (env._apply_action과 동일 계산) ─────────────────
    rock = env.rock_pos[e].cpu().numpy()
    tx, ty, tz = _sph_to_cart(psi, phi, theta)
    theory_pos = np.array([rock[0]+tx, rock[1]+ty, rock[2]+tz])

    # ── 실제 센서리그 위치 (시뮬레이터 반영값) ───────────────────────────
    actual_pos = env.cam_pos[e].cpu().numpy()
    actual_quat = env.cam_orient[e].cpu().numpy()  # [w,x,y,z]

    # ── 오차 ─────────────────────────────────────────────────────────────
    err = theory_pos - actual_pos
    err_norm = np.linalg.norm(err)

    # ── 타겟까지 실제 거리 ────────────────────────────────────────────────
    actual_dist = np.linalg.norm(actual_pos - rock)

    # ── forward 벡터 (rock 방향) ─────────────────────────────────────────
    fwd = rock - actual_pos
    fwd_norm = fwd / (np.linalg.norm(fwd) + 1e-8)

    # ── 조명 ─────────────────────────────────────────────────────────────
    light = env._light_level[e].item()

    header = f"[STATE{' | '+tag if tag else ''}]"
    print(f"\n{SEPARATOR}")
    print(f"{header}")
    print(f"{SEPARATOR}")
    print(f"  구면좌표 (env 내부)")
    print(f"    θ = {_deg(theta):8.2f}°   ({_r(theta):.4f} rad)")
    print(f"    φ = {_deg(phi):8.2f}°   ({_r(phi):.4f} rad)  "
          f"[허용: {_deg(env.cfg.phi_min):.0f}°~{_deg(env.cfg.phi_max):.0f}°]")
    print(f"    ψ = {_r(psi):8.4f} m              "
          f"[허용: {env.cfg.psi_min:.1f}~{env.cfg.psi_max:.1f}]")
    print(f"    light_level = {light}  "
          f"(intensity = {light * env.cfg.light_intensity_per_level:.0f})")
    print(f"")
    print(f"  이론 cam_pos  (구면→직교 계산값)")
    print(f"    [{theory_pos[0]:8.4f}, {theory_pos[1]:8.4f}, {theory_pos[2]:8.4f}]")
    print(f"  실제 cam_pos  (simulator 반영값)")
    print(f"    [{actual_pos[0]:8.4f}, {actual_pos[1]:8.4f}, {actual_pos[2]:8.4f}]")
    print(f"  오차 벡터     theory - actual")
    print(f"    [{err[0]:8.4f}, {err[1]:8.4f}, {err[2]:8.4f}]  norm={err_norm:.5f} m")
    print(f"")
    print(f"  타겟(rock) 위치   [{rock[0]:.4f}, {rock[1]:.4f}, {rock[2]:.4f}]")
    print(f"  타겟까지 실제거리 {actual_dist:.4f} m  (ψ={_r(psi):.4f})")
    print(f"  forward 방향      [{fwd_norm[0]:.4f}, {fwd_norm[1]:.4f}, {fwd_norm[2]:.4f}]")
    print(f"  cam orient [w,x,y,z]  "
          f"[{actual_quat[0]:.4f}, {actual_quat[1]:.4f}, "
          f"{actual_quat[2]:.4f}, {actual_quat[3]:.4f}]")
    print(f"{SEPARATOR}\n")

    return {
        "theta": theta, "phi": phi, "psi": psi,
        "theory_pos": theory_pos, "actual_pos": actual_pos,
        "err_norm": err_norm, "actual_dist": actual_dist,
        "light": light, "tag": tag,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. action 전후 비교 (단일 액션 효과 검증)
# ─────────────────────────────────────────────────────────────────────────────

ACT_NAMES = {
    0: "θ+", 1: "θ−", 2: "φ+", 3: "φ−",
    4: "ψ+", 5: "ψ−", 6: "L−", 7: "L±0", 8: "L+",
}

def print_action_effect(env, action_idx: int, repeat: int = 1) -> dict:
    """
    action_idx를 repeat번 실행하고, 전후 상태 변화를 출력.
    decimation=2이므로 env.step() 1회 → _apply_action() 2회 → delta 2배 주의.
    """
    e = 0
    before = {
        "theta": env._sph_theta[e].item(),
        "phi"  : env._sph_phi[e].item(),
        "psi"  : env._sph_psi[e].item(),
        "pos"  : env.cam_pos[e].cpu().numpy().copy(),
        "light": env._light_level[e].item(),
    }

    a = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)
    a[:, action_idx] = 1.0

    for _ in range(repeat):
        env.step(a)

    after = {
        "theta": env._sph_theta[e].item(),
        "phi"  : env._sph_phi[e].item(),
        "psi"  : env._sph_psi[e].item(),
        "pos"  : env.cam_pos[e].cpu().numpy().copy(),
        "light": env._light_level[e].item(),
    }

    dtheta = after["theta"] - before["theta"]
    dphi   = after["phi"]   - before["phi"]
    dpsi   = after["psi"]   - before["psi"]
    dpos   = after["pos"]   - before["pos"]
    # decimation=2: env.step 1회 = _apply_action 2회
    expected_delta = {
        0: ("θ", +env.cfg.delta_theta * 2 * repeat),
        1: ("θ", -env.cfg.delta_theta * 2 * repeat),
        2: ("φ", +env.cfg.delta_phi   * 2 * repeat),
        3: ("φ", -env.cfg.delta_phi   * 2 * repeat),
        4: ("ψ", +env.cfg.delta_psi   * 2 * repeat),
        5: ("ψ", -env.cfg.delta_psi   * 2 * repeat),
        6: ("L", -1 * repeat), 7: ("L", 0), 8: ("L", +1 * repeat),
    }
    exp_axis, exp_val = expected_delta[action_idx]

    print(f"\n{SEPARATOR}")
    print(f"[ACTION EFFECT]  act[{action_idx}]={ACT_NAMES[action_idx]}  x{repeat}회  "
          f"(decimation={env.cfg.decimation} → _apply_action x{env.cfg.decimation*repeat}회)")
    print(f"{SEPARATOR}")
    print(f"  {'':12s}  {'before':>12s}  {'after':>12s}  {'actual Δ':>12s}  {'expect Δ':>12s}  {'일치':>4s}")
    print(f"  {'θ (deg)':12s}  {_deg(before['theta']):>12.2f}  {_deg(after['theta']):>12.2f}  "
          f"{_deg(dtheta):>+12.2f}  "
          f"{(_deg(exp_val) if exp_axis=='θ' else 0.0):>+12.2f}  "
          f"{'OK' if exp_axis!='θ' or abs(_deg(dtheta)-_deg(exp_val if exp_axis=='θ' else 0))<0.5 else 'NG':>4s}")
    print(f"  {'φ (deg)':12s}  {_deg(before['phi']):>12.2f}  {_deg(after['phi']):>12.2f}  "
          f"{_deg(dphi):>+12.2f}  "
          f"{(_deg(exp_val) if exp_axis=='φ' else 0.0):>+12.2f}  "
          f"{'OK' if exp_axis!='φ' or abs(_deg(dphi)-_deg(exp_val if exp_axis=='φ' else 0))<0.5 else 'NG':>4s}")
    print(f"  {'ψ (m)':12s}  {_r(before['psi']):>12.4f}  {_r(after['psi']):>12.4f}  "
          f"{_r(dpsi):>+12.4f}  "
          f"{(exp_val if exp_axis=='ψ' else 0.0):>+12.4f}  "
          f"{'OK' if exp_axis!='ψ' or abs(dpsi-(exp_val if exp_axis=='ψ' else 0))<0.01 else 'NG':>4s}")
    print(f"  {'light':12s}  {before['light']:>12d}  {after['light']:>12d}  "
          f"{after['light']-before['light']:>+12d}  "
          f"{(int(exp_val) if exp_axis=='L' else 0):>+12d}  "
          f"{'OK' if exp_axis!='L' or (after['light']-before['light'])==int(exp_val) else 'NG':>4s}")
    print(f"")
    print(f"  cam_pos Δ  [{dpos[0]:+.4f}, {dpos[1]:+.4f}, {dpos[2]:+.4f}]  "
          f"norm={np.linalg.norm(dpos):.4f} m")
    print(f"{SEPARATOR}\n")

    return {"before": before, "after": after,
            "dtheta": dtheta, "dphi": dphi, "dpsi": dpsi}


# ─────────────────────────────────────────────────────────────────────────────
# 3. 목표 궤도 vs 실제 상태 오차 분석
# ─────────────────────────────────────────────────────────────────────────────

def print_orbital_diff(env, target_phi_deg: float, target_psi: float,
                       target_theta_deg: float, label: str = "") -> dict:
    """
    목표 구면좌표 (target) vs env 실제 상태 차이를 출력.
    """
    e = 0
    rock = env.rock_pos[e].cpu().numpy()

    # 목표
    t_theta = math.radians(target_theta_deg)
    t_phi   = math.radians(target_phi_deg)
    t_psi   = target_psi
    tx, ty, tz = _sph_to_cart(t_psi, t_phi, t_theta)
    target_pos = np.array([rock[0]+tx, rock[1]+ty, rock[2]+tz])

    # 실제
    a_theta = env._sph_theta[e].item()
    a_phi   = env._sph_phi[e].item()
    a_psi   = env._sph_psi[e].item()
    actual_pos = env.cam_pos[e].cpu().numpy()

    d_theta = math.degrees(a_theta) - target_theta_deg
    d_phi   = math.degrees(a_phi)   - target_phi_deg
    d_psi   = a_psi - t_psi
    pos_err = np.linalg.norm(actual_pos - target_pos)

    ok_theta = abs(d_theta) < math.degrees(env.cfg.delta_theta) + 0.1
    ok_phi   = abs(d_phi)   < math.degrees(env.cfg.delta_phi)   + 0.1
    ok_psi   = abs(d_psi)   < env.cfg.delta_psi + 0.01

    print(f"\n{SEPARATOR}")
    print(f"[ORBITAL DIFF{' | '+label if label else ''}]")
    print(f"{SEPARATOR}")
    print(f"  {'':8s}  {'target':>10s}  {'actual':>10s}  {'diff':>10s}  {'OK?':>5s}")
    print(f"  {'θ (deg)':8s}  {target_theta_deg:>10.2f}  "
          f"{math.degrees(a_theta):>10.2f}  {d_theta:>+10.2f}  "
          f"{'OK' if ok_theta else '!! NG':>5s}")
    print(f"  {'φ (deg)':8s}  {target_phi_deg:>10.2f}  "
          f"{math.degrees(a_phi):>10.2f}  {d_phi:>+10.2f}  "
          f"{'OK' if ok_phi else '!! NG':>5s}")
    print(f"  {'ψ (m)':8s}  {t_psi:>10.4f}  {a_psi:>10.4f}  {d_psi:>+10.4f}  "
          f"{'OK' if ok_psi else '!! NG':>5s}")
    print(f"")
    print(f"  target_pos  [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
    print(f"  actual_pos  [{actual_pos[0]:.4f}, {actual_pos[1]:.4f}, {actual_pos[2]:.4f}]")
    print(f"  pos error   {pos_err:.5f} m")
    print(f"{SEPARATOR}\n")

    return {
        "d_theta_deg": d_theta, "d_phi_deg": d_phi,
        "d_psi": d_psi, "pos_err": pos_err,
        "ok": ok_theta and ok_phi and ok_psi,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. 궤도 순회 디버그 실행 루프
# ─────────────────────────────────────────────────────────────────────────────

def run_orbital_debug(
    env,
    phi_levels_deg: list[float] = [20.0, 40.0, 60.0, 75.0],
    steps_per_level: int = 24,
    target_psi: float = 3.0,
    settle_steps: int = 1,
    verbose: bool = True,         # 매 스텝 full state 출력
    check_action_effect: bool = True,  # 첫 스텝에서 action 효과 검증
) -> list[dict]:
    """
    궤도 순회를 실행하면서 매 viewpoint마다:
    - 목표 vs 실제 구면좌표 비교
    - 이론 cam_pos vs 실제 cam_pos 오차
    - 누적 오차 통계
    를 출력한다.
    """
    import math
    DELTA_THETA = env.cfg.delta_theta
    DELTA_PHI   = env.cfg.delta_phi
    DELTA_PSI   = env.cfg.delta_psi
    PHI_MAX     = env.cfg.phi_max
    DECIMATION  = env.cfg.decimation  # _apply_action 호출 배수

    def one_hot(idx):
        a = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)
        a[:, idx] = 1.0
        return a

    def clicks(diff, delta):
        return max(0, round(abs(diff) / delta))

    # ── reset ──────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"[ORBITAL DEBUG START]")
    print(f"  phi_levels={phi_levels_deg}°  steps/level={steps_per_level}")
    print(f"  target_psi={target_psi}  settle={settle_steps}")
    print(f"  decimation={DECIMATION}  (env.step 1회 = _apply_action {DECIMATION}회)")
    print(f"  delta_theta={math.degrees(DELTA_THETA):.1f}°  "
          f"delta_phi={math.degrees(DELTA_PHI):.1f}°  delta_psi={DELTA_PSI}")
    print(f"{'='*72}\n")

    obs, _ = env.reset()

    # reset 직후 상태 출력
    snap = print_env_state(env, "AFTER RESET")
    print(f"  [주의] _reset_idx: phi=radians(89°) 설정 → "
          f"clamp 후 실제={math.degrees(env._sph_phi[0].item()):.2f}°\n")

    # ── action 효과 검증 (옵션) ────────────────────────────────────────────
    if check_action_effect:
        print(f"[ACTION EFFECT 검증] 각 축 1회씩 테스트 후 reset\n")
        for act_idx in range(9):
            print_action_effect(env, act_idx, repeat=1)
        obs, _ = env.reset()
        print(f"[검증 완료 → reset]\n")

    # ── ψ 이동 ────────────────────────────────────────────────────────────
    curr_psi = env._sph_psi[0].item()
    curr_phi = env._sph_phi[0].item()

    diff_psi = target_psi - curr_psi
    n = clicks(diff_psi, DELTA_PSI * DECIMATION)  # decimation 보정
    if n > 0:
        act = 4 if diff_psi > 0 else 5
        print(f"[PSI 이동] {curr_psi:.2f} → {target_psi:.2f}  ({n} steps)")
        for _ in range(n):
            obs, *_ = env.step(one_hot(act))
        print_env_state(env, f"PSI 이동 후 (목표={target_psi:.2f})")
        diff = print_orbital_diff(env, math.degrees(curr_phi),
                                  target_psi, 0.0, "ψ 이동 후 체크")
        if not diff["ok"]:
            print(f"  !! ψ 오차 큼: delta_psi={DELTA_PSI}×decimation={DELTA_PSI*DECIMATION}로 계산 재확인 필요")

    # ── 레벨별 궤도 순회 ──────────────────────────────────────────────────
    all_diffs = []
    theta_step_rad = (2 * math.pi) / steps_per_level
    clicks_per_theta = max(1, round(theta_step_rad / (DELTA_THETA * DECIMATION)))

    for level_idx, phi_deg in enumerate(phi_levels_deg):
        target_phi_rad = math.radians(phi_deg)

        # φ 이동
        curr_phi = env._sph_phi[0].item()
        diff_phi = target_phi_rad - curr_phi
        n_phi = clicks(diff_phi, DELTA_PHI * DECIMATION)

        if n_phi > 0:
            act = 2 if diff_phi > 0 else 3
            print(f"\n[L{level_idx+1}] φ 이동: {math.degrees(curr_phi):.1f}°→{phi_deg:.1f}°  "
                  f"({n_phi} steps)")
            for _ in range(n_phi):
                obs, *_ = env.step(one_hot(act))
            if verbose:
                print_env_state(env, f"L{level_idx+1} φ 이동 후")
                print_orbital_diff(env, phi_deg, target_psi, 0.0,
                                   f"L{level_idx+1} φ 체크")
        else:
            print(f"\n[L{level_idx+1}] φ={phi_deg}° 이미 도달 (이동 없음)")

        # θ 순회
        print(f"[L{level_idx+1}] θ 순회 시작  (steps={steps_per_level}, "
              f"{math.degrees(theta_step_rad):.1f}°/step, "
              f"clicks/step={clicks_per_theta})")

        for step in range(steps_per_level):
            target_theta_deg = math.degrees(theta_step_rad) * step

            for _ in range(clicks_per_theta):
                obs, reward, terminated, truncated, _ = env.step(one_hot(0))  # θ+
                if terminated.any() or truncated.any():
                    obs, _ = env.reset()
                    print(f"  !! RESET at L{level_idx+1} step {step+1}")

            # settle
            for _ in range(settle_steps):
                obs, reward, *_ = env.step(one_hot(7))  # light hold

            diff = print_orbital_diff(
                env,
                target_phi_deg   = phi_deg,
                target_psi       = target_psi,
                target_theta_deg = (target_theta_deg + math.degrees(theta_step_rad)) % 360,
                label = f"L{level_idx+1} step {step+1:02d}/{steps_per_level}",
            )
            all_diffs.append({**diff,
                              "level": level_idx+1, "step": step+1,
                              "reward": reward[0].item()})

            if verbose:
                print_env_state(env, f"L{level_idx+1}-{step+1:02d}")

    # ── 전체 오차 요약 ────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"[ORBITAL DEBUG SUMMARY]  총 {len(all_diffs)} viewpoints")
    print(f"{'='*72}")
    for lvl in sorted(set(d["level"] for d in all_diffs)):
        sub = [d for d in all_diffs if d["level"] == lvl]
        pos_errs   = [d["pos_err"]    for d in sub]
        theta_errs = [abs(d["d_theta_deg"]) for d in sub]
        phi_errs   = [abs(d["d_phi_deg"])   for d in sub]
        psi_errs   = [abs(d["d_psi"])       for d in sub]
        rewards    = [d["reward"]           for d in sub]
        ng_count   = sum(1 for d in sub if not d["ok"])
        print(f"  L{lvl}  pos_err avg={sum(pos_errs)/len(pos_errs):.4f}m "
              f"max={max(pos_errs):.4f}m  "
              f"θ_err={sum(theta_errs)/len(theta_errs):.2f}°  "
              f"φ_err={sum(phi_errs)/len(phi_errs):.2f}°  "
              f"ψ_err={sum(psi_errs)/len(psi_errs):.4f}  "
              f"NG={ng_count}/{len(sub)}  "
              f"reward_avg={sum(rewards)/len(rewards):.3f}")
    print(f"{'='*72}\n")

    return all_diffs