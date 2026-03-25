"""
extension.py
------------
Isaac Sim GUI Extension 진입점.

Isaac Sim의 Extension 시스템에 등록되어
GUI에서 활성화/비활성화할 수 있다.

on_startup():  Phase 0 초기화 (Extension 활성화 시 1회)
on_update():   Phase 1 NBUV 루프 (매 Isaac Sim 프레임)
on_shutdown(): 정리 (Extension 비활성화 시)

standalone_run.py와 동일한 내부 모듈을 사용한다.
"""

import numpy as np
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


# ── Isaac Sim Extension 기반 클래스 ──────────────────────────────────────────
try:
    import omni.ext
    EXTENSION_BASE = omni.ext.IExt
except ImportError:
    # Isaac Sim 없는 환경에서도 파일이 import 가능하도록
    class EXTENSION_BASE:
        pass


class NBUVExtension(EXTENSION_BASE):
    """
    NBUV Isaac Sim Extension.

    GUI 패널에서:
      - 파라미터 조정 (beta, sigma_max, 후보 수 등)
      - NBUV 루프 시작/일시정지/리셋
      - albedo 맵 실시간 시각화
    """

    def on_startup(self, ext_id: str):
        """Extension 활성화 시 호출. Phase 0 초기화."""
        print("[NBUVExtension] 시작")

        self._step   = 0
        self._active = False
        self._results_history = []

        # ── 설정 로드 ──────────────────────────────────────────────────────
        from standalone_run import load_configs
        config_dir = os.path.join(PROJECT_ROOT, 'config')
        self._params, self._ws_constraints = load_configs(config_dir)

        # ── 모듈 임포트 ────────────────────────────────────────────────────
        from models.mesh_builder         import build_patches_from_usd, make_test_plane
        from models.oceansim_bridge      import create_bridge
        from models.nbuv_estimator       import NBUVEstimator
        from controller.arm_controller   import ArmController

        # ── Mesh 구성 ──────────────────────────────────────────────────────
        try:
            self._patches = build_patches_from_usd(
                "/World/Environment/scene",
                voxel_size=self._params['voxel_size']
            )
        except Exception:
            print("[NBUVExtension] USD geometry 없음 → 합성 평면 사용")
            self._patches = make_test_plane(nx=10, ny=10)

        N = self._patches['N']
        print(f"[NBUVExtension] 패치 수: {N}")

        # ── Bridge & Controller ────────────────────────────────────────────
        config_path = os.path.join(PROJECT_ROOT, 'config', 'water_params.json')
        self._bridge = create_bridge(config_path, synthetic=False)

        self._controller = ArmController(bridge=self._bridge)

        # ── 추정기 ─────────────────────────────────────────────────────────
        self._estimator = NBUVEstimator(
            N=N,
            sigma_max=self._params.get('sigma_max', 0.5),
            sigma_RN=self._params.get('sigma_RN', 0.01)
        )

        # 타겟 중심
        self._target_center = self._patches['centers'].mean(axis=0)

        # ── UI 구성 ────────────────────────────────────────────────────────
        self._build_ui()

        print("[NBUVExtension] 초기화 완료")

    def on_update(self, dt: float):
        """
        매 Isaac Sim 프레임 호출.
        _active가 True일 때만 NBUV 스텝을 실행한다.
        """
        if not self._active:
            return

        max_steps = self._params.get('n_steps', 20)
        if self._step >= max_steps:
            self._active = False
            print(f"[NBUVExtension] {max_steps}스텝 완료. 루프 종료.")
            return

        self._run_nbuv_step()
        self._step += 1

    def on_shutdown(self):
        """Extension 비활성화 시 정리."""
        print("[NBUVExtension] 종료")
        self._active = False

    # ── NBUV 단일 스텝 ────────────────────────────────────────────────────────

    def _run_nbuv_step(self):
        """Phase 1 NBUV 루프의 단일 스텝을 실행한다."""
        from models.patch_visibility     import compute_visibility
        from models.nbuv_lighting        import compute_lighting
        from planner.candidate_generator import generate_candidates
        from planner.nbuv_optimizer      import find_next_best_view

        cam_pos   = self._controller.get_cam_pos()
        light_pos = self._controller.get_light_pos()
        fov_deg   = self._ws_constraints.get('fov_deg', 60.0)

        # ① 센서
        _ = self._bridge.get_rgb()
        _ = self._bridge.get_depth()

        # ② 가시성
        vis = compute_visibility(self._patches, cam_pos, light_pos, fov_deg)

        # ③ 조명
        lighting = compute_lighting(vis, self._params)

        # ④ 추정 업데이트
        est_result = self._estimator.update(lighting, vis)

        # ⑤ 후보 생성
        candidates = generate_candidates(self._target_center,
                                          self._ws_constraints)

        if len(candidates) == 0:
            return

        # ⑥ 최적화
        opt_result = find_next_best_view(
            candidates, self._patches, self._estimator,
            self._params, fov_deg)

        # ⑦ 이동
        self._controller.move_to(opt_result['cam_pos'],
                                   opt_result['light_pos'])

        self._results_history.append({
            'step':      self._step,
            'info_gain': opt_result['info_gain'],
            'rho_mean':  est_result['rho_ML'].mean(),
        })

        # UI 업데이트
        self._update_ui(opt_result['info_gain'])

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        """Isaac Sim UI 패널을 구성한다."""
        try:
            import omni.ui as ui

            self._window = ui.Window("NBUV Controller", width=300, height=400)
            with self._window.frame:
                with ui.VStack():
                    ui.Label("NBUV in OceanSim", height=30)
                    ui.Separator()

                    with ui.HStack(height=30):
                        ui.Label("스텝:")
                        self._step_label = ui.Label("0")

                    with ui.HStack(height=30):
                        ui.Label("Info Gain:")
                        self._gain_label = ui.Label("-")

                    with ui.HStack(height=30):
                        ui.Label("관측 비율:")
                        self._obs_label = ui.Label("-")

                    ui.Separator()

                    ui.Button("시작/재개", clicked_fn=self._on_start)
                    ui.Button("일시정지", clicked_fn=self._on_pause)
                    ui.Button("리셋",     clicked_fn=self._on_reset)

        except ImportError:
            print("[NBUVExtension] UI 없는 환경 (omni.ui 미설치)")
            self._window = None

    def _update_ui(self, info_gain: float):
        """UI 레이블을 갱신한다."""
        if self._window is None:
            return
        self._step_label.text = str(self._step)
        self._gain_label.text  = f"{info_gain:.4f}"
        self._obs_label.text   = (
            f"{self._estimator.get_observed_fraction()*100:.1f}%")

    # ── UI 버튼 콜백 ──────────────────────────────────────────────────────────

    def _on_start(self):
        print("[NBUVExtension] 루프 시작")
        self._active = True

    def _on_pause(self):
        print("[NBUVExtension] 루프 일시정지")
        self._active = False

    def _on_reset(self):
        print("[NBUVExtension] 리셋")
        self._active = False
        self._step   = 0
        self._estimator.reset()
        self._results_history = []
