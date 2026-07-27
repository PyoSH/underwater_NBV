"""
BROV2 수중 동역학/유체역학 물리 검증 런처
==========================================
train.py(RL 학습 전용, 향후 RSL-RL 연동 예정)와 분리된 검증 전용 스크립트.
bottom_up.py의 검증 함수(neutral_buoyancy/straight_line/rotation/thruster_model/six_dof)를
--test 플래그로 실행한다.

--record_video 사용 시 IsaacLab 내장 카메라 추적 기능(ViewerCfg origin_type="asset_root")으로
카메라가 로봇 root를 매 렌더 스텝마다 자동으로 따라가며 동작을 mp4로 기록한다.
헤드리스에서 녹화하려면 오프스크린 렌더링이 필요한데, --record_video 사용 시 자동으로 켜진다.

주의: gym.wrappers.RecordVideo는 쓰지 않는다 — gymnasium 1.2 기준 Wrapper가 env.cfg /
env._robot / env.device 같은 속성을 내부 env로 포워딩하지 않아서, bottom_up.py의 테스트
함수(env 속성을 직접 참조)가 AttributeError로 깨진다. 대신 env.step/env.reset을 직접
감싸 매 호출 후 env.render() 프레임을 리스트에 모았다가 종료 시 imageio로 mp4를 쓴다.
"""

import argparse
import os
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BROV2 물리 검증 (train.py와 분리)")
parser.add_argument(
    "--test",
    choices=["neutral_buoyancy", "straight_line", "rotation", "thruster_model", "six_dof"],
    default="neutral_buoyancy",
)
parser.add_argument("--thrust", type=float, default=0.5)
parser.add_argument("--rotation_thrust", type=float, default=0.3)
parser.add_argument("--duration", type=float, default=5.0)
parser.add_argument("--record_video", action="store_true", help="카메라가 로봇을 따라가며 mp4로 기록")
parser.add_argument("--video_path", type=str, default=None, help="기본값: step_2_BROV/videos/<test>.mp4")
parser.add_argument("--cam_eye", type=float, nargs=3, default=[-2.5, -2.5, 1.5],
                     help="로봇 root 기준 카메라 위치 오프셋 [m] (world frame)")
parser.add_argument("--cam_lookat", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                     help="로봇 root 기준 카메라 응시점 오프셋 [m]")
AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()
if args.record_video:
    args.enable_cameras = True   # 헤드리스에서도 오프스크린 렌더링 필요

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.dirname(__file__))
from envs.traj_env_cfg import BROVTrajEnvCfg
from envs.traj_env import BROVTrajEnv
from physics_tests import bottom_up


def _enable_video_recording(env) -> list:
    """env.step/env.reset을 감싸서 매 호출 직후 env.render() 프레임을 수집한다.
    env는 래핑하지 않고 원본 BROVTrajEnv 인스턴스를 그대로 반환하므로 bottom_up.py의
    env.cfg/env._robot 등 직접 속성 접근이 그대로 동작한다."""
    frames: list = []
    orig_step = env.step
    orig_reset = env.reset

    def step_and_capture(action):
        result = orig_step(action)
        frames.append(env.render())
        return result

    def reset_and_capture(*a, **kw):
        result = orig_reset(*a, **kw)
        frames.append(env.render())
        return result

    env.step = step_and_capture
    env.reset = reset_and_capture
    return frames


def _save_video(frames: list, path: str, fps: float) -> None:
    import imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps)
    print(f"[INFO] 영상 저장: {path} ({len(frames)} frames, {fps:.1f} fps)")


if __name__ == "__main__":
    cfg = BROVTrajEnvCfg()
    cfg.scene.num_envs = 1
    cfg.max_bound_x = cfg.max_bound_y = cfg.max_bound_z = 50.0

    render_mode = "rgb_array" if args.record_video else None
    if args.record_video:
        # IsaacLab 내장 뷰포트 카메라 추적(ViewportCameraController) — 매 렌더 스텝마다
        # scene.articulations["robot"]의 root 위치를 읽어 eye/lookat을 자동 갱신한다.
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name  = "robot"
        cfg.viewer.eye    = tuple(args.cam_eye)
        cfg.viewer.lookat = tuple(args.cam_lookat)

    env = BROVTrajEnv(cfg, render_mode=render_mode)

    frames = _enable_video_recording(env) if args.record_video else None

    try:
        if args.test == "neutral_buoyancy":
            bottom_up.test_neutral_buoyancy(env, duration_s=args.duration)
        elif args.test == "straight_line":
            bottom_up.test_straight_line(env, thrust=args.thrust, duration_s=args.duration)
        elif args.test == "rotation":
            bottom_up.test_rotation(env, thrust=args.thrust, duration_s=args.duration)
        elif args.test == "thruster_model":
            bottom_up.test_thruster_model(env, duration_s=args.duration)
        elif args.test == "six_dof":
            bottom_up.test_six_dof(env, thrust=args.thrust,
                                    rotation_thrust=args.rotation_thrust,
                                    duration_s=args.duration)

        print("\n[INFO] 모든 테스트가 완료되었습니다.")

        if args.record_video:
            video_path = args.video_path or os.path.join(
                os.path.dirname(__file__), "videos", f"{args.test}.mp4"
            )
            fps = 1.0 / (cfg.sim.dt * cfg.decimation)
            _save_video(frames, video_path, fps)
        else:
            print("[INFO] WebRTC 접속 유지를 위해 시뮬레이션을 계속 구동합니다. (종료: Ctrl+C)")
            import torch
            while simulation_app.is_running():
                empty_action = torch.zeros(env.action_space.shape, device=env.device)
                env.step(empty_action)
    finally:
        env.close()
        simulation_app.close()
