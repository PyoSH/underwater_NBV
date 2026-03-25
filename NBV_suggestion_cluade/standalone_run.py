"""
standalone_run.py
-----------------
Isaac Sim headless 모드 또는 순수 NumPy 모드로 NBUV 루프를 실행한다.

실행 방법:
  # Isaac Sim headless
  $ ./python.sh standalone_run.py --scene shipwreck.usd --steps 20

  # 순수 NumPy (Isaac Sim 없이 개발/테스트)
  $ python standalone_run.py --synthetic --steps 10

NBUV 루프 (매 스텝):
  ① OceanSim → RGB, Depth
  ② patch_visibility → visible_mask, l_SC, l_LS
  ③ nbuv_lighting → E_s, B_s (Eq.2~4)
  ④ nbuv_estimator → Q^ML 업데이트 (Eq.19~20)
  ⑤ candidate_generator → 후보 pose 집합
  ⑥ nbuv_optimizer → argmax I_{t+1}(O) (Eq.22~25)
  ⑦ arm_controller → UR5e 이동 → USD 동기화
"""

import argparse
import json
import os
import sys
import numpy as np

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.mesh_builder         import build_patches_from_numpy, make_test_plane
from models.patch_visibility     import compute_visibility
from models.oceansim_bridge      import create_bridge
from models.nbuv_lighting        import compute_lighting
from models.nbuv_estimator       import NBUVEstimator
from planner.candidate_generator import generate_candidates
from planner.nbuv_optimizer      import find_next_best_view
from controller.arm_controller   import ArmController


# ─────────────────────────────────────────────────────────────────────────────
# 설정 로더
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith('_')}


def load_configs(config_dir: str) -> tuple:
    water_params  = _load_json(os.path.join(config_dir, 'water_params.json'))
    nbuv_params   = _load_json(os.path.join(config_dir, 'nbuv_params.json'))
    ws_constraints = _load_json(os.path.join(config_dir, 'workspace_constraints.json'))

    # water_params에 nbuv_params 병합 (sigma_RN 등 공유)
    params = {**water_params, **nbuv_params}
    return params, ws_constraints


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0: 초기화
# ─────────────────────────────────────────────────────────────────────────────

def phase0_init(params: dict, ws_constraints: dict,
                synthetic: bool, scene_path: str = None):
    """
    환경 구성, mesh 생성, 추정기 초기화.

    Returns:
        patches, estimator, bridge, controller
    """
    print("[Phase 0] 초기화 시작")

    # ── Mesh 구성 ─────────────────────────────────────────────────────────────
    if synthetic or scene_path is None:
        print("  [Mesh] 합성 평면 mesh 사용")
        patches = make_test_plane(nx=10, ny=10, width=2.0, height=2.0)
    else:
        print(f"  [Mesh] USD scene에서 geometry 추출: {scene_path}")
        from models.mesh_builder import build_patches_from_usd
        patches = build_patches_from_usd("/World/Environment/scene",
                                          voxel_size=params['voxel_size'])

    N = patches['N']
    print(f"  [Mesh] 패치 수: {N}")

    # ── OceanSim Bridge ───────────────────────────────────────────────────────
    config_path = os.path.join(PROJECT_ROOT, 'config', 'water_params.json')
    bridge      = create_bridge(config_path, synthetic=synthetic)
    print(f"  [Bridge] {'Synthetic' if synthetic else 'Isaac Sim'} 모드")

    # ── 컨트롤러 ──────────────────────────────────────────────────────────────
    controller = ArmController(bridge=bridge)
    print(f"  [Controller] ArmController 초기화 완료")

    # ── 추정기 초기화 ──────────────────────────────────────────────────────────
    sigma_max = params.get('sigma_max', 0.5)
    sigma_RN  = params.get('sigma_RN', 0.01)
    estimator = NBUVEstimator(N=N, sigma_max=sigma_max, sigma_RN=sigma_RN)
    print(f"  [Estimator] Q^ML 초기화: {estimator.Q_ML[0]:.4f} (모든 패치)")

    print("[Phase 0] 초기화 완료\n")
    return patches, estimator, bridge, controller


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: NBUV 루프
# ─────────────────────────────────────────────────────────────────────────────

def phase1_nbuv_loop(patches: dict,
                      estimator: NBUVEstimator,
                      bridge,
                      controller: ArmController,
                      params: dict,
                      ws_constraints: dict,
                      n_steps: int,
                      target_center: np.ndarray) -> dict:
    """
    NBUV Algorithm 1 메인 루프.

    Returns:
        결과 dict (trajectory, info_gain_history, final_rho_ML)
    """
    print(f"[Phase 1] NBUV 루프 시작 ({n_steps} 스텝)")

    trajectory       = []   # [(cam_pos, light_pos), ...]
    info_gain_history = []  # [float, ...]

    fov_deg = ws_constraints.get('fov_deg', 60.0)

    for step in range(n_steps):
        print(f"\n  [Step {step+1}/{n_steps}]")

        cam_pos   = controller.get_cam_pos()
        light_pos = controller.get_light_pos()

        # ── ① 센서 취득 ───────────────────────────────────────────────────────
        rgb   = bridge.get_rgb()
        depth = bridge.get_depth()

        # ── ② 가시성 계산 ─────────────────────────────────────────────────────
        vis = compute_visibility(patches, cam_pos, light_pos, fov_deg)
        n_visible = vis['visible_mask'].sum()

        # ── ③ 조명 계산 (Eq.2~4) ─────────────────────────────────────────────
        lighting = compute_lighting(vis, params)

        # ── ④ 추정기 업데이트 (Eq.12, 17, 19, 20) ────────────────────────────
        est_result = estimator.update(lighting, vis)

        print(f"    가시 패치:     {n_visible}/{patches['N']}")
        print(f"    관측 비율:     {estimator.get_observed_fraction()*100:.1f}%")
        print(f"    rho_ML 평균:   {est_result['rho_ML'].mean():.4f}")

        # ── ⑤ 후보 pose 생성 ──────────────────────────────────────────────────
        candidates = generate_candidates(target_center, ws_constraints)
        print(f"    후보 pose 수:  {len(candidates)}")

        if len(candidates) == 0:
            print("    경고: 유효한 후보가 없습니다. 현재 pose 유지.")
            trajectory.append((cam_pos.copy(), light_pos.copy()))
            info_gain_history.append(0.0)
            continue

        # ── ⑥ 정보이득 최대화 (Eq.22~25) ────────────────────────────────────
        opt_result = find_next_best_view(candidates, patches, estimator,
                                          params, fov_deg)
        info_gain_history.append(opt_result['info_gain'])
        print(f"    최적 info gain: {opt_result['info_gain']:.4f}")

        # ── ⑦ 팔 이동 ────────────────────────────────────────────────────────
        success = controller.move_to(opt_result['cam_pos'],
                                      opt_result['light_pos'])
        trajectory.append((opt_result['cam_pos'].copy(),
                            opt_result['light_pos'].copy()))

        print(f"    baseline:      {controller.get_baseline()*100:.1f} cm")

        if not success:
            print("    경고: 팔 이동 실패 (IK)")

    print(f"\n[Phase 1] 완료")
    print(f"  총 관측 패치 비율: {estimator.get_observed_fraction()*100:.1f}%")
    print(f"  최종 rho_ML 평균:  {estimator.rho_ML.mean():.4f}")

    return {
        'trajectory':        trajectory,
        'info_gain_history': info_gain_history,
        'final_rho_ML':      estimator.rho_ML.copy(),
        'final_Q_ML':        estimator.Q_ML.copy(),
        'obs_count':         estimator.obs_count.copy(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 결과 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, output_dir: str):
    """NumPy 배열로 결과를 저장한다."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'rho_ML.npy'),
            results['final_rho_ML'])
    np.save(os.path.join(output_dir, 'Q_ML.npy'),
            results['final_Q_ML'])
    np.save(os.path.join(output_dir, 'info_gain_history.npy'),
            np.array(results['info_gain_history']))
    np.save(os.path.join(output_dir, 'obs_count.npy'),
            results['obs_count'])

    # 궤적 저장
    cam_traj   = np.array([p[0] for p in results['trajectory']])
    light_traj = np.array([p[1] for p in results['trajectory']])
    np.save(os.path.join(output_dir, 'cam_trajectory.npy'),   cam_traj)
    np.save(os.path.join(output_dir, 'light_trajectory.npy'), light_traj)

    print(f"\n결과 저장 완료: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='NBUV in OceanSim')
    parser.add_argument('--synthetic', action='store_true',
                        help='Isaac Sim 없이 합성 데이터로 실행')
    parser.add_argument('--scene', type=str, default=None,
                        help='USD 씬 파일 경로')
    parser.add_argument('--steps', type=int, default=10,
                        help='NBUV 루프 스텝 수')
    parser.add_argument('--output', type=str, default='./results',
                        help='결과 저장 디렉토리')
    args = parser.parse_args()

    # ── 설정 로드 ──────────────────────────────────────────────────────────────
    config_dir   = os.path.join(PROJECT_ROOT, 'config')
    params, ws_constraints = load_configs(config_dir)

    n_steps = args.steps or params.get('n_steps', 10)

    # ── Phase 0: 초기화 ────────────────────────────────────────────────────────
    patches, estimator, bridge, controller = phase0_init(
        params, ws_constraints,
        synthetic=(args.synthetic or args.scene is None),
        scene_path=args.scene
    )

    # 타겟 중심 (장면 중심)
    target_center = patches['centers'].mean(axis=0)

    # ── Phase 1: NBUV 루프 ────────────────────────────────────────────────────
    results = phase1_nbuv_loop(
        patches, estimator, bridge, controller,
        params, ws_constraints,
        n_steps=n_steps,
        target_center=target_center
    )

    # ── 결과 저장 ──────────────────────────────────────────────────────────────
    save_results(results, args.output)


if __name__ == "__main__":
    main()
