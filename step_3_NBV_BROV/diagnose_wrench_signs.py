"""Z-up body frame wrench 부호 검증 — DP 컨트롤러를 우회한 순수 액추에이션 진단.

`envs/env.py`의 `_ZUP_TO_SNAME_SIGN`(T6 대각) → `B_pinv` → 스러스터 경로가
정말 의도한 축/부호로 힘·토크를 내는지 확인한다. step_2의
`physics_tests/bottom_up.py::test_rotation`과 동일한 방법론(축별 고정 명령을
넣고 실제 body-frame 각속도/선속도 부호를 확인)이되, step_2는 RL 액션(6-dim,
f_max 스케일)을 넣는 반면 여기서는 DPController가 내보내는 것과 정확히 같은
형식의 **Z-up body frame wrench [Fx,Fy,Fz,Tx,Ty,Tz] (N, N·m)**를 직접 주입한다.

기대: +Tx → +roll rate(ang_vel_b[0]), +Ty → +pitch, +Tz → +yaw,
      +Fx → +surge(lin_vel_b[0]), +Fy → +sway, +Fz → +heave.
어긋나면 `_ZUP_TO_SNAME_SIGN` 또는 그 적용 지점이 결함이다.

주의: 부력/중력·유체항력이 동시에 작용하므로 짧은 구간의 "부호"만 본다
(크기는 축별 F_max/관성/댐핑이 달라 서로 비교하지 않는다).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration_s", type=float, default=1.0)
parser.add_argument("--force_n", type=float, default=30.0)
parser.add_argument("--torque_nm", type=float, default=5.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from envs.env_cfg import NBVBROVEnvCfg
from envs.env import NBVBROVEnv

# 축 정의: (라벨, wrench 인덱스, 확인할 속도 배열, 그 배열의 인덱스)
_CASES = [
    ("Fx (surge)", 0, "lin", 0),
    ("Fy (sway)",  1, "lin", 1),
    ("Fz (heave)", 2, "lin", 2),
    ("Tx (roll)",  3, "ang", 0),
    ("Ty (pitch)", 4, "ang", 1),
    ("Tz (yaw)",   5, "ang", 2),
]

cfg = NBVBROVEnvCfg()
cfg.scene.num_envs = args.num_envs
# 진단 중 목표 추종/DR이 개입하지 않도록 물리를 nominal로 고정
cfg.dr_enable_mass = False
cfg.dr_volume_range = (0.014665, 0.014665)
cfg.dr_cob_radius = 0.0
cfg.dr_added_mass_rot_range = (1.0, 1.0)
# 짧은 구간만 보므로 decimation을 물리 스텝 단위로 되돌린다(5초 통짜 X)
cfg.decimation = 1
cfg.sim.render_interval = 1

env = None
try:
    env = NBVBROVEnv(cfg)
    # CoB까지 정확히 0으로 — 트림모멘트가 부호 판정을 오염시키지 않게
    env._hydro._r_cob_ned[:] = 0.0
    env._hydro._nominal_r_cob_ned[:] = 0.0

    n_steps = int(args.duration_s / cfg.sim.dt)
    print(f"[diag] {n_steps} physics steps/axis, force={args.force_n}N torque={args.torque_nm}N*m")

    # DP 컨트롤러를 우회: _apply_action이 고정 wrench를 쓰도록 교체
    injected = {"wrench": torch.zeros(env.num_envs, 6, device=env.device)}

    class _FixedWrench:
        def reset(self, env_ids):
            pass

        def compute(self, **kwargs):
            return injected["wrench"]

    env._dp = _FixedWrench()

    print(f"\n{'축':<14} {'주축 응답':>12} {'교차축 크기':>12}   판정")
    print("-" * 58)
    failures = []
    for label, w_idx, vel_kind, vel_idx in _CASES:
        env.reset()
        env._hydro._r_cob_ned[:] = 0.0

        magnitude = args.force_n if w_idx < 3 else args.torque_nm
        w = torch.zeros(env.num_envs, 6, device=env.device)
        w[:, w_idx] = magnitude
        injected["wrench"] = w

        actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
        for _ in range(n_steps):
            env.step(actions)

        vel = (
            env._robot.data.root_lin_vel_b if vel_kind == "lin"
            else env._robot.data.root_ang_vel_b
        )[0]
        main = vel[vel_idx].item()
        cross_idx = [i for i in range(3) if i != vel_idx]
        cross = float((vel[cross_idx[0]] ** 2 + vel[cross_idx[1]] ** 2).sqrt())

        if abs(main) < 1e-3:
            verdict = "X 반응 없음"
            failures.append(label)
        elif main < 0:
            verdict = "X 부호 반대!"
            failures.append(label)
        elif cross > abs(main) * 0.5:
            verdict = "△ 부호는 맞으나 교차축 과다"
        else:
            verdict = "OK"
        print(f"{label:<14} {main:>+12.5f} {cross:>12.5f}   {verdict}")

    print("-" * 58)
    if failures:
        print(f"[diag] RESULT: FAIL — 부호/응답 이상 축: {', '.join(failures)}")
    else:
        print("[diag] RESULT: PASS — 6축 전부 기대 부호로 응답")
finally:
    if env is not None:
        env.close()

simulation_app.close()
