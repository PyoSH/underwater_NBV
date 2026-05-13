"""
evaluate_RL.py
==============
학습된 PPO 체크포인트를 로드해 OceanEnv에서 자율 동작을 확인하는 독립 평가 스크립트.

목적
----
1. [정성적] 에이전트가 환경에서 자율적으로 움직이는 것을 눈으로 확인
2. [정량적] 여러 에피소드에 걸쳐 return / coverage / success_rate 집계

사용법
------
python evaluate_RL.py \
    --checkpoint ./checkpoints/step_0000500000.pt \
    --num_envs   1 \
    --num_episodes 20 \
    --render          # 렌더링 켜기 (기본 off)
    --step_log        # 스텝마다 상태 출력
"""

from __future__ import annotations
import argparse, os, sys, time, math
from pathlib import Path

# ── AppLauncher는 가장 먼저 ────────────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OceanNBV PPO — Evaluation")

# ── 필수 ──────────────────────────────────────────────────────────────────────
parser.add_argument("--checkpoint",    type=str, required=True,
                    help="평가할 .pt 체크포인트 경로")

# ── 환경 ──────────────────────────────────────────────────────────────────────
parser.add_argument("--num_envs",      type=int, default=1,
                    help="병렬 환경 수 (정성 확인은 1 권장)")

# ── 평가 설정 ──────────────────────────────────────────────────────────────────
parser.add_argument("--num_episodes",  type=int, default=10,
                    help="각 env에서 완료할 에피소드 수")
parser.add_argument("--max_steps",     type=int, default=0,
                    help="스텝 상한 (0 = max_episode_length * num_episodes * 2)")

# ── 출력 설정 ──────────────────────────────────────────────────────────────────
parser.add_argument("--render",        action="store_true",
                    help="Isaac Sim 뷰어 렌더링 활성화")
parser.add_argument("--step_log",      action="store_true",
                    help="매 스텝 상태(theta/phi/coverage 등) 출력")
parser.add_argument("--save_csv",      type=str, default=None,
                    help="결과를 저장할 CSV 경로 (미지정 시 저장 안 함)")

AppLauncher.add_app_launcher_args(parser)

# 카메라는 항상 필요
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── AppLauncher 이후 import ───────────────────────────────────────────────────
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from envCfg    import OceanEnvCfg
from env       import OceanEnv
from algorithm2 import Actor, Critic, make_env_action


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼: 체크포인트 로드
# ─────────────────────────────────────────────────────────────────────────────
def load_actor(checkpoint_path: str, env_cfg: OceanEnvCfg, device: torch.device) -> Actor:
    K_img = env_cfg.visual.num_seq_actor
    ckpt  = torch.load(checkpoint_path, map_location=device)

    actor = Actor(img_ch=K_img, scalar_dim=3).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    step = ckpt.get("global_step", "?")
    print(f"[ckpt] loaded  →  {checkpoint_path}  (global_step={step})")
    return actor


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼: 에피소드 완료 후 결과 기록
# ─────────────────────────────────────────────────────────────────────────────
def _record_episode(results, ep_idx, eid, ep_return, ep_len, env, terminated):
    results["ep_idx"]    .append(ep_idx)
    results["env_id"]    .append(eid)
    results["return"]    .append(ep_return[eid].item())
    results["length"]    .append(ep_len[eid].item())
    results["coverage"]  .append(env.curr_coverage[eid].item())
    results["success"]   .append(float(terminated[eid].item()))
    results["surf_voxels"].append(env._total_surf_voxels[eid].item())


# ─────────────────────────────────────────────────────────────────────────────
# 메인 평가 루프
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(env: OceanEnv, actor: Actor, device: torch.device,
             n_episodes: int, max_steps: int, step_log: bool) -> dict:
    """
    모든 env에서 각 n_episodes 에피소드를 완료할 때까지 rollout.
    에피소드 완료(terminated | truncated) 시마다 결과를 기록한다.
    """
    E = env.num_envs
    if max_steps == 0:
        max_steps = env.max_episode_length * n_episodes * 2

    results = dict(
        ep_idx=[], env_id=[], return_=[], length=[],
        coverage=[], success=[], surf_voxels=[],
    )
    # 키 이름 통일 (위 dict key와 _record_episode 내부 key를 맞춤)
    results = dict(
        ep_idx=[], env_id=[],
        **{k: [] for k in ("return", "length", "coverage", "success", "surf_voxels")}
    )

    obs, _     = env.reset()
    obs_img    = obs["policy"]
    obs_scalar = obs["extra_info"]

    ep_return  = torch.zeros(E, device=device)
    ep_len     = torch.zeros(E, device=device, dtype=torch.long)
    completed  = [0] * E          # env별 완료 에피소드 수
    ep_counter = [0] * E          # env별 에피소드 인덱스 (출력용)

    t_start = time.time()
    step_global = 0

    with torch.no_grad():
        for step in range(max_steps):

            # ── 모든 env가 목표 에피소드 수 달성 시 종료 ──────────────────────
            if min(completed) >= n_episodes:
                break

            # ── 정책 행동 ─────────────────────────────────────────────────────
            pose_act   = actor.greedy(obs_img, obs_scalar)
            env_action = make_env_action(pose_act, E, device)

            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done_any = terminated | truncated

            ep_return += reward
            ep_len    += 1
            step_global += E

            # ── 상세 출력 ─────────────────────────────────────────────────────
            if step_log:
                theta_deg = math.degrees(env._sph_theta[0].item())
                phi_deg   = math.degrees(env._sph_phi[0].item())
                psi       = env._sph_psi[0].item()
                cov       = env.curr_coverage[0].item()
                rew       = reward[0].item()
                print(
                    f"[step {step:5d}]"
                    f"  θ={theta_deg:+7.2f}°  φ={phi_deg:+7.2f}°  ψ={psi:+.3f}"
                    f"  cov={cov:.4f}  rew={rew:+.4f}",
                    flush=True,
                )

            # ── 에피소드 완료 처리 ────────────────────────────────────────────
            for eid in done_any.nonzero(as_tuple=True)[0].tolist():
                if completed[eid] < n_episodes:
                    _record_episode(results, ep_counter[eid], eid,
                                    ep_return, ep_len, env, terminated)

                    status = "SUCCESS " if terminated[eid].item() else "timeout"
                    print(
                        f"  [ep done] env={eid}  ep={ep_counter[eid]:3d}  {status}"
                        f"  return={ep_return[eid].item():+.3f}"
                        f"  len={ep_len[eid].item():4.0f}"
                        f"  coverage={env.curr_coverage[eid].item():.4f}",
                        flush=True,
                    )
                    completed[eid]  += 1
                    ep_counter[eid] += 1

                ep_return[eid] = 0.0
                ep_len[eid]    = 0

            obs_img    = next_obs["policy"]
            obs_scalar = next_obs["extra_info"]

    elapsed = time.time() - t_start
    fps     = step_global / (elapsed + 1e-8)

    # ── 집계 ──────────────────────────────────────────────────────────────────
    def _m(key):
        vals = results[key]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "total_episodes":    len(results["return"]),
        "mean_return":       _m("return"),
        "std_return":        float(np.std(results["return"]))   if results["return"]   else 0.0,
        "mean_length":       _m("length"),
        "mean_coverage":     _m("coverage"),
        "std_coverage":      float(np.std(results["coverage"])) if results["coverage"] else 0.0,
        "success_rate":      _m("success"),
        "timeout_rate":      1.0 - _m("success"),
        "mean_surf_voxels":  _m("surf_voxels"),
        "elapsed_sec":       round(elapsed, 1),
        "fps":               round(fps, 1),
    }
    return summary, results


# ─────────────────────────────────────────────────────────────────────────────
# 결과 출력 및 CSV 저장
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(summary: dict, checkpoint: str):
    ckpt_name = Path(checkpoint).stem
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  Evaluation Summary  ·  {ckpt_name}")
    print(sep)
    print(f"  episodes       : {summary['total_episodes']}")
    print(f"  mean return    : {summary['mean_return']:+.4f}  ± {summary['std_return']:.4f}")
    print(f"  mean length    : {summary['mean_length']:.1f} steps")
    print(f"  mean coverage  : {summary['mean_coverage']:.4f}  ± {summary['std_coverage']:.4f}")
    print(f"  success rate   : {summary['success_rate']*100:.1f} %")
    print(f"  timeout rate   : {summary['timeout_rate']*100:.1f} %")
    print(f"  surf voxels    : {summary['mean_surf_voxels']:.0f}")
    print(f"  elapsed        : {summary['elapsed_sec']} s  ({summary['fps']:.0f} fps)")
    print(sep)


def save_csv(results: dict, path: str):
    import csv
    keys = ["ep_idx", "env_id", "return", "length", "coverage", "success", "surf_voxels"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for i in range(len(results["return"])):
            writer.writerow({k: results[k][i] for k in keys})
    print(f"[csv] saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── 환경 ──────────────────────────────────────────────────────────────────
    env_cfg = OceanEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    render_mode = "rgb_array" if args.render else None
    env    = OceanEnv(cfg=env_cfg, render_mode=render_mode)
    device = env.device

    # ── Actor 로드 ────────────────────────────────────────────────────────────
    actor = load_actor(args.checkpoint, env_cfg, device)

    # ── 평가 실행 ─────────────────────────────────────────────────────────────
    print(f"\n[eval] start  num_envs={args.num_envs}  "
          f"num_episodes={args.num_episodes}  step_log={args.step_log}")

    summary, results = evaluate(
        env, actor, device,
        n_episodes=args.num_episodes,
        max_steps=args.max_steps,
        step_log=args.step_log,
    )

    # ── 출력 ──────────────────────────────────────────────────────────────────
    print_summary(summary, args.checkpoint)

    if args.save_csv:
        save_csv(results, args.save_csv)

    # ── 정리 ──────────────────────────────────────────────────────────────────
    try:
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()