"""
학습된 BROVVelEnv 정책 검증 — LOS 유도 통합 테스트
======================================================
Sim2Swim(arXiv:2512.08656) 논문의 실물 검증 방법론을 시뮬레이션에서 재현한다.

논문은 시뮬레이션 단독 검증 구간이 아예 없이(학습 reward 곡선 → 바로 실물
테스트) Fig.4의 3가지 실물 시나리오만 보고하는데, 전부 "3D LOS 유도가 명령을
만들고 저수준 정책이 그걸 따라간다"는 구조다. 이 스크립트는 그 셋을 전부
시뮬레이션으로 재현한다 (우리는 실물이 없으니 이 구간을 시뮬레이션으로 채움):

  (a) straight_line        : 직선 왕복, heading/pitch는 LOS가 진행방향으로 계산
                              ("desired heading and pitch equal to desired
                              course and elevation angles calculated by LOS
                              guidance law")
  (b) square_ballast        : 사각 웨이포인트 4개, 자세는 항상 수평(0,0,0) 유지,
                              600g 밸러스트로 순부력을 양성→음성 전환
                              ("changes vehicle's buoyancy from positive to
                              negative"). **근사**: 실제 PhysX 질량은 안 바꾸고
                              (mass 랜덤화는 2단계 미구현) 동일한 순부력 결손을
                              내는 volume 감소로 대체 — 부력 역전 효과는 같지만
                              밸러스트의 관성/무게중심 이동까지 정확히 재현하진
                              않음.
  (c) square_random_attitude: 사각 웨이포인트 4개, 도달할 때마다 자세 목표를
                              roll,pitch~U(-π/2,π/2), yaw~U(-π,π)에서 새로 샘플
                              ("setpoints randomly generated and changed at
                              each waypoint").

`bottom_up.py`의 물리검증과는 목적이 다르다: 그건 "물리모델이 맞는가"였고,
이건 "학습된 정책+LOS 유도 파이프라인이 실제로 경로를 잘 따라가는가"다.

사용법
------
python test_policy.py --checkpoint logs/brov_vel/model_299.pt --test straight_line [--headless]
python test_policy.py --checkpoint logs/brov_vel/model_299.pt --test square_ballast --duration 60 --headless
python test_policy.py --checkpoint logs/brov_vel/model_299.pt --test square_random_attitude --record_video --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BROV2 속도 컨트롤러 + LOS 유도 통합 검증")
parser.add_argument("--checkpoint", type=str, required=True, help="RSL-RL model_*.pt 체크포인트 경로")
parser.add_argument(
    "--test",
    choices=["straight_line", "square_ballast", "square_random_attitude"],
    default="straight_line",
    help="Sim2Swim 논문 Fig.4 (a)/(b)/(c) 재현",
)
parser.add_argument("--duration", type=float, default=60.0,
                     help="[s] — square 경로는 한 바퀴 도는 데 시간이 걸려 straight_line보다 넉넉히")
parser.add_argument("--cruise_speed", type=float, default=0.5,
                     help="[m/s] LOS 순항속도 — 학습 시 Vd(=0.5)와 맞추는 게 기본")
parser.add_argument("--record_video", action="store_true",
                     help="경로 전체가 보이는 고정 조망 카메라로 mp4 기록 (거리는 경로 크기에서 자동 계산)")
parser.add_argument("--video_path", type=str, default=None)
parser.add_argument("--cam_eye", type=float, nargs=3, default=None,
                     help="미지정 시 웨이포인트 경로 크기로부터 자동 계산 (env-local 절대 좌표)")
parser.add_argument("--cam_lookat", type=float, nargs=3, default=None,
                     help="미지정 시 웨이포인트 경로의 중심으로 자동 계산 (env-local 절대 좌표)")
AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()
if args.record_video:
    args.enable_cameras = True   # 헤드리스에서도 오프스크린 렌더링 필요

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── AppLauncher 기동 이후에만 import 가능 ──
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from importlib.metadata import version as _pkg_version

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from envs.vel_env_cfg import BROVVelEnvCfg
from envs.vel_env import BROVVelEnv
from guidance.los_guidance import LOSGuidance
from agents.rsl_rl_ppo_cfg import BROVVelPPORunnerCfg
from robots.dynamics.brov2.params import load_brov2_yaml   # nominal volume 조회용 — 값 재사용, 하드코딩 안 함

_PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")

# 논문 Fig.4 (a)/(b)/(c) → LOSGuidance heading_mode 매핑
_HEADING_MODE = {
    "straight_line"        : "align",               # Trial(a)
    "square_ballast"        : "upright",              # Trial(b), 자세 항상 수평
    "square_random_attitude": "random_at_waypoint",   # Trial(c)
}

_BALLAST_MASS_KG        = 0.6    # 600g, 논문 Trial(b)
_BALLAST_LATERAL_OFFSET = 0.05   # [m] port(−Y, SNAME) 방향 근사 오프셋 — 실측 아님, 예시값

_FORWARD_ARROW_LEN = 0.2   # [m] 3D 플롯의 forward direction 화살표 길이 — 경로 크기와 무관하게 작게 고정

_STARTING_DEPTH = 5.0   # [m] world Z — 웨이포인트 Z와 반드시 맞춰야 함 (아래 _build_waypoints 참조)


def _quat_to_euler_zyx_deg(quat: torch.Tensor) -> torch.Tensor:
    """bottom_up.py._quat_to_euler_zyx_deg와 동일 구현 (quat order [w,x,y,z])."""
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


def _build_waypoints(test: str, device) -> torch.Tensor:
    """(1, num_wp, 3) env-local 웨이포인트.

    Z는 전부 _STARTING_DEPTH로 맞춘다 — 시작 높이와 웨이포인트 높이가 어긋나면
    첫 순간부터 큰 수직 이동을 해야 해서 텀블링이 발생함(실제로 겪은 버그,
    plots/policy_eval_square_ballast.png 참조) — X/Y 평면상의 경로만 테스트하려는
    의도이므로 Z는 전 구간 동일하게 고정.

    straight_line: 2점 왕복 — LOSGuidance가 인덱스를 (idx+1)%num_wp로 순환하므로
    2개 waypoint만 주면 자동으로 "왕복"이 된다 (논문 Trial(a) 재현).
    square_*     : 4점 사각 경로(한 변 5m), 마찬가지로 순환해서 계속 돎 (Trial(b)/(c)).
    """
    z = _STARTING_DEPTH
    if test == "straight_line":
        wps = torch.tensor([[0.0, 0.0, z], [5.0, 0.0, z]], device=device)
    elif test in ("square_ballast", "square_random_attitude"):
        wps = torch.tensor([
            [0.0, 0.0, z],
            [5.0, 0.0, z],
            [5.0, 5.0, z],
            [0.0, 5.0, z],
        ], device=device)
    else:
        raise ValueError(f"Unknown test: {test}")
    return wps.unsqueeze(0)   # (1, N, 3)


def _compute_overview_camera(waypoints_np: np.ndarray) -> tuple[tuple, tuple]:
    """웨이포인트 경로 전체가 한 프레임에 들어오는 고정 조망 카메라(eye, lookat) 계산.

    체이스캠(로봇에 딱 붙어 따라가는 카메라)은 로봇 자체는 잘 보이지만 5m 직선/
    사각형 같은 경로 전체 맥락이 화면에 안 담긴다 — "궤적을 그리며 움직이는 걸
    확인"하려는 목적엔 안 맞음. 대신 경로의 바운딩박스 중심을 바라보는 고정
    카메라를 45도 사선 위에서 잡는다 — 논문 Fig.4 하단 3D 궤적 스냅샷과 비슷한 구도.
    거리는 경로 크기에 비례하되, straight_line처럼 작은 경로에서도 너무 바짝
    붙지 않게 최소 거리(6m)를 둔다.
    """
    center = waypoints_np.mean(axis=0)
    extent = waypoints_np[:, :2].max(axis=0) - waypoints_np[:, :2].min(axis=0)
    path_size = max(float(extent.max()), 1.0)
    dist = max(path_size * 1.6, 6.0)

    d = dist / np.sqrt(2)
    eye = (center[0] - d, center[1] - d, center[2] + dist * 0.5)
    lookat = tuple(center.tolist())
    return eye, lookat


def _apply_physics_scenario(env: "BROVVelEnv", test: str) -> None:
    """평가는 재현성이 중요 — env.reset()이 매번 넣는 랜덤 도메인 랜덤화 값을
    시나리오별 고정값으로 덮어쓴다 (Project_BROV의 legacy 평가 스크립트도
    `use_custom_randomization=False`로 랜덤화를 꺼두고 재현 가능한 조건에서
    테스트했음 — 같은 이유).

    straight_line/square_random_attitude: 전부 nominal(YAML 실측값)로 되돌림.
    square_ballast: 600g 밸러스트와 동일한 순부력 결손을 volume 감소로 근사
    + port 방향 CoB 오프셋(비대칭 밸러스트로 인한 롤 유발 근사).
    """
    env_ids = torch.zeros(1, dtype=torch.long, device=env.device)
    nominal_volume = load_brov2_yaml()["volume"]

    if test == "square_ballast":
        deficit_volume = _BALLAST_MASS_KG / env._hydro._water_density
        volume = torch.full((1,), nominal_volume - deficit_volume, device=env.device)
        cob_offset = torch.tensor([[0.0, -_BALLAST_LATERAL_OFFSET, 0.0]], device=env.device)
    else:
        volume = torch.full((1,), nominal_volume, device=env.device)
        cob_offset = torch.zeros(1, 3, device=env.device)

    env._hydro.randomize(
        env_ids, volume=volume, cob_offset=cob_offset,
        added_mass_rot=env._nominal_added_mass_rot.unsqueeze(0),
    )


def _forward_dir_from_quat(quat: np.ndarray) -> np.ndarray:
    """quat(...,4)[w,x,y,z] → world-frame body-X(전방) 단위벡터(...,3).

    회전행렬의 첫 번째 열 공식(순수 numpy — 플로팅 단계는 torch/math_utils에
    의존하지 않게 분리): R@[1,0,0] = [1-2(y²+z²), 2(xy+wz), 2(xz-wy)].
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.stack([
        1 - 2 * (y**2 + z**2),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
    ], axis=-1)


def _plot_results(log: dict, test_name: str, waypoints: np.ndarray, reach_threshold: float) -> None:
    """Sim2Swim 논문 Fig.4 레이아웃 재현: 위 4단(u/v/w/자세, 실선=실제·점선=목표,
    surge-roll=빨강/sway-pitch=초록/heave-yaw=파랑 색 규칙까지 동일) + 아래 3D
    궤적(Position/Waypoints/Radius of acceptance/Forward direction, 시작=초록·끝=빨강).
    """
    os.makedirs(_PLOT_DIR, exist_ok=True)
    t     = np.array(log["t"])
    pos   = np.array(log["pos"])
    quat  = np.array(log["quat"])
    euler = np.array(log["euler_deg"])
    v_act = np.array(log["v_actual"])
    v_des = np.array(log["v_desired"])
    q_des_euler = np.array(log["q_desired_euler"])

    fig = plt.figure(figsize=(7, 13))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1.3, 2.2], hspace=0.35)

    colors = ["red", "green", "blue"]   # surge/roll, sway/pitch, heave/yaw — 논문과 동일 색 규칙
    vel_labels = ["u", "v", "w"]
    vel_axes = []
    for i in range(3):
        ax = fig.add_subplot(gs[i, 0], sharex=vel_axes[0] if vel_axes else None)
        ax.plot(t, v_des[:, i], ":", color=colors[i], label=f"{vel_labels[i]}$_d$")
        ax.plot(t, v_act[:, i], "-", color=colors[i], label=vel_labels[i])
        ax.set_ylabel("Velocities [m/s]" if i == 0 else f"{vel_labels[i]} [m/s]")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)
        plt.setp(ax.get_xticklabels(), visible=False)
        vel_axes.append(ax)

    # 자세(φ,θ,ψ) — 논문처럼 세 축을 한 패널에 같이
    ax_att = fig.add_subplot(gs[3, 0], sharex=vel_axes[0])
    att_labels = ["$\\phi$(roll)", "$\\theta$(pitch)", "$\\psi$(yaw)"]
    for i in range(3):
        ax_att.plot(t, q_des_euler[:, i], ":", color=colors[i], label=f"{att_labels[i]}$_d$")
        ax_att.plot(t, euler[:, i], "-", color=colors[i], label=att_labels[i])
    ax_att.set_ylabel("Attitude [deg]")
    ax_att.set_xlabel("Time [s]")
    ax_att.legend(loc="upper right", fontsize=7, ncol=2)
    ax_att.grid(alpha=0.3)

    # ── 3D 궤적 (Position/Waypoints/Radius of acceptance/Forward direction) ──
    ax3d = fig.add_subplot(gs[4, 0], projection="3d")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="tab:blue", linewidth=1.5, label="Position")
    ax3d.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                 color="orange", s=25, label="Waypoints")

    theta = np.linspace(0, 2 * np.pi, 60)
    for i, wp in enumerate(waypoints):
        circ = np.stack([
            wp[0] + reach_threshold * np.cos(theta),
            wp[1] + reach_threshold * np.sin(theta),
            np.full_like(theta, wp[2]),
        ], axis=-1)
        ax3d.plot(circ[:, 0], circ[:, 1], circ[:, 2], ":", color="orange",
                  linewidth=1.0, label="Radius of acceptance" if i == 0 else None)

    # 전방 방향 화살표 — 궤적을 따라 몇 지점 샘플링 (경로 규모와 무관하게 작게 고정)
    n_arrows = 8
    idxs = np.linspace(0, len(pos) - 1, n_arrows).astype(int)
    fwd = _forward_dir_from_quat(quat[idxs])
    ax3d.quiver(pos[idxs, 0], pos[idxs, 1], pos[idxs, 2],
                fwd[:, 0], fwd[:, 1], fwd[:, 2],
                length=_FORWARD_ARROW_LEN, color="purple", label="Forward direction")

    ax3d.scatter(*pos[0], color="green", marker="X", s=90, label="start")
    ax3d.scatter(*pos[-1], color="red", marker="X", s=90, label="end")
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.xaxis.set_major_locator(MultipleLocator(1))   # grid 한 칸 = 1m
    ax3d.yaxis.set_major_locator(MultipleLocator(1))
    ax3d.zaxis.set_major_locator(MultipleLocator(1))
    ax3d.legend(loc="upper left", fontsize=7)

    fig.suptitle(f"BROVVelEnv + LOSGuidance — {test_name}")
    save_path = os.path.join(_PLOT_DIR, f"policy_eval_{test_name}.png")
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[INFO] 결과 플롯 저장: {save_path}")

    vel_err = np.linalg.norm(v_act - v_des, axis=-1)
    tail_mask = t > (t[-1] - 2.0)
    print(f"\n[요약] {test_name}")
    print(f"  평균 속도 오차 노름         : {vel_err.mean():.4f} m/s")
    print(f"  마지막 2초 평균 속도 오차   : {vel_err[tail_mask].mean():.4f} m/s (정상상태 오차 근사)")


def _save_video(frames: list, cfg) -> None:
    import imageio
    path = args.video_path or os.path.join(
        os.path.dirname(__file__), "videos", f"policy_eval_{args.test}.mp4"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fps = 1.0 / (cfg.sim.dt * cfg.decimation)
    imageio.mimwrite(path, frames, fps=fps)
    print(f"[INFO] 영상 저장: {path} ({len(frames)} frames, {fps:.1f} fps)")


def main() -> None:
    env_cfg = BROVVelEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = args.duration + 5.0   # 테스트 도중 timeout으로 리셋되지 않게 여유
    env_cfg.max_bound = 30.0                           # 직선 경로가 학습용 경계보다 넓으니 완화
    # 웨이포인트 Z와 시작 높이를 맞춰야 함 — 안 맞으면 첫 순간부터 큰 수직 이동을 하게 됨
    # (실제로 이 불일치로 극초반 텀블링 발생을 확인함 — plots/policy_eval_square_ballast.png).
    # _build_waypoints()도 동일하게 Z=_STARTING_DEPTH를 쓰도록 맞춤.
    env_cfg.starting_depth = _STARTING_DEPTH

    render_mode = "rgb_array" if args.record_video else None
    if args.record_video:
        # validate_physics.py의 체이스캠(로봇에 딱 붙어 따라감)은 로봇 자체는 잘
        # 보이지만 5m 직선/사각형 같은 경로 전체 맥락이 화면에 안 담긴다 — 여기서는
        # 경로(웨이포인트) 전체를 내려다보는 고정 조망 카메라를 씀 (origin_type="env",
        # eye/lookat이 env-local 절대좌표 — _build_waypoints와 동일 기준이라 그대로 사용 가능).
        wp_cpu = _build_waypoints(args.test, "cpu")[0].numpy()
        auto_eye, auto_lookat = _compute_overview_camera(wp_cpu)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index   = 0
        env_cfg.viewer.eye    = tuple(args.cam_eye)    if args.cam_eye    is not None else auto_eye
        env_cfg.viewer.lookat = tuple(args.cam_lookat) if args.cam_lookat is not None else auto_lookat

    env = BROVVelEnv(cfg=env_cfg, render_mode=render_mode)

    waypoints = _build_waypoints(args.test, env.device)
    los = LOSGuidance(
        waypoints, env.device,
        cruise_speed=args.cruise_speed, heading_mode=_HEADING_MODE[args.test],
    )
    env.attach_guidance(los)

    # ── 정책 로드 (train.py와 동일 API, 이미 검증됨) ──
    agent_cfg = BROVVelPPORunnerCfg()
    # rsl-rl-lib>=5.0.0: RslRlMLPModelCfg의 폐기 필드(stochastic 등)가 기본값이어도
    # to_dict()에 직렬화돼 MLPModel.__init__()이 거부함 — train.py와 동일하게 마이그레이션.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env)   # OnPolicyRunner 구성/로드 전용 — 이후 루프는 env(unwrapped)로 직접
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)
    print(f"[INFO] 체크포인트 로드 완료: {args.checkpoint}")

    frames: list = []
    log = {k: [] for k in (
        "t", "pos", "quat", "euler_deg", "v_actual", "v_desired", "q_desired_euler",
        "action_mag", "reset", "wp_idx",
    )}

    obs_dict, _ = env.reset()
    _apply_physics_scenario(env, args.test)   # reset()이 넣은 랜덤 DR 값을 재현 가능한 고정값으로 교체

    num_steps = int(args.duration / env._policy_dt)
    reset_count = 0
    with torch.inference_mode():
        for i in range(num_steps):
            # rsl-rl-lib>=5.0.0: obs_groups 스키마라 정책이 관측 딕셔너리 전체를 받아
            # 내부에서 obs["policy"]를 꺼내 씀 — 텐서만 미리 꺼내서 넘기면 안 됨.
            actions = policy(obs_dict)
            obs_dict, reward, terminated, truncated, info = env.step(actions)
            did_reset = bool((terminated | truncated)[0])
            reset_count += int(did_reset)

            log["t"].append(i * env._policy_dt)
            log["pos"].append(env._robot.data.root_pos_w[0].cpu().tolist())
            log["quat"].append(env._robot.data.root_quat_w[0].cpu().tolist())
            log["euler_deg"].append(_quat_to_euler_zyx_deg(env._robot.data.root_quat_w[0]).cpu().tolist())
            log["v_actual"].append(env._robot.data.root_lin_vel_b[0].cpu().tolist())
            log["v_desired"].append(env._v_d_b[0].cpu().tolist())
            log["q_desired_euler"].append(_quat_to_euler_zyx_deg(env._q_d[0]).cpu().tolist())
            log["action_mag"].append(float(env._actions[0].abs().mean()))
            log["reset"].append(did_reset)
            log["wp_idx"].append(int(los._wp_idx[0]))

            if args.record_video:
                frames.append(env.render())

    # waypoints는 env-local 좌표라 world 좌표(= pos 로그와 같은 기준)로 변환 — env.close() 전에 읽어야 함
    wp_world = (waypoints[0] + env.scene.env_origins[0]).cpu().numpy()
    reach_threshold = los._reach

    # DirectRLEnv.close()는 scene 삭제 후에 콜백을 정리해서, 그 사이에 렌더 이벤트가
    # 한 번 더 들어오면 _debug_vis_callback이 이미 없는 scene에 접근해 죽는다
    # (record_video 사용 시 실제로 발생 확인) — close() 전에 명시적으로 구독 해제.
    env.set_debug_vis(False)
    env.close()

    pos_arr = np.array(log["pos"])
    act_arr = np.array(log["action_mag"])
    t_arr   = np.array(log["t"])
    wp_arr  = np.array(log["wp_idx"])
    switch_steps = np.nonzero(np.diff(wp_arr) != 0)[0] + 1   # wp_idx가 바뀐 스텝 인덱스

    print(f"\n[진단]")
    print(f"  위치 변화 범위 (max-min) X/Y/Z [m]: {(pos_arr.max(0) - pos_arr.min(0)).round(4).tolist()}")
    print(f"  평균 |action|                    : {act_arr.mean():.4f}  (0에 가까우면 정책이 사실상 안 움직임)")
    print(f"  최대 |action|                    : {act_arr.max():.4f}")
    print(f"  중간 리셋 발생 횟수(out_of_bounds/timeout): {reset_count} / {num_steps} step")
    print(f"  waypoint 전환 횟수: {len(switch_steps)}, 전환 시각[s]: {t_arr[switch_steps].round(2).tolist()}")

    _plot_results(log, args.test, wp_world, reach_threshold)
    if args.record_video:
        _save_video(frames, env_cfg)

    simulation_app.close()


if __name__ == "__main__":
    main()
