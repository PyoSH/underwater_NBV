"""`envs/depth_corruption.py` 수학 검증 — Isaac 씬 없이 CPU에서 돈다.

왜 별도 테스트인가: realized AbsRel이 ②a 스윕의 **가로축**이다. 이 값이 틀리면
"AbsRel 0.XX까지 견딘다"는 결론 자체가 무의미해지고, 그 오류는 스윕 결과를
아무리 들여다봐도 보이지 않는다(그럴듯한 숫자가 나오기 때문). 그래서 이론값이
있는 항목은 이론값과 대조한다.

    /isaac-sim/python.sh -u tools/test_depth_corruption.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from envs.depth_corruption import DepthCorruptionCfg, DepthCorruptor

E, DEV = 64, "cpu"
IDS = torch.arange(E)
D_TRUE = torch.rand(E, 24, 32) * 3.0 + 0.5      # 0.5~3.5 m


def make(**kw):
    cfg = DepthCorruptionCfg()
    cfg.enabled = True
    for k, v in kw.items():
        setattr(cfg, k, v)
    c = DepthCorruptor(cfg, E, DEV)
    c.reset(IDS)
    return c


def main() -> int:
    print("[1] 비활성이면 항등")
    c0 = DepthCorruptor(DepthCorruptionCfg(), E, DEV)
    c0.reset(IDS)
    assert torch.equal(c0.depth(D_TRUE), D_TRUE), "depth 항등 실패"
    p, _ = c0.apply_pose(torch.zeros(E, 3), torch.tensor([[1., 0, 0, 0]]).expand(E, 4))
    assert torch.equal(p, torch.zeros(E, 3)), "pose 항등 실패"
    print("   OK — 학습 경로(오염 off)의 동작은 변하지 않는다")

    print("\n[2] 스케일은 에피소드 내 상수")
    # 프레임마다 독립이면 융합에서 평균으로 상쇄돼 실제보다 훨씬 관대한 시험이
    # 된다. 단안 depth의 스케일 오차는 장면 단위로 상관돼 있다.
    c = make(scale_sigma=0.10)
    s1 = c.scale.clone()
    c.depth(D_TRUE); c.depth(D_TRUE)
    assert torch.equal(c.scale, s1), "프레임마다 스케일이 변한다"
    c.reset(IDS)
    assert not torch.equal(c.scale, s1), "리셋해도 스케일이 그대로다"
    print(f"   OK — 프레임 간 고정, 리셋 시 재추첨 (σ=0.1 표본 std={s1.std():.3f})")

    print("\n[3] realized AbsRel vs 이론값   E|s−1| ≈ σ√(2/π) = 0.798σ")
    print(f"{'조건':>20}{'측정':>9}{'이론':>9}{'오차':>8}")
    worst = 0.0
    for sg in (0.05, 0.10, 0.20, 0.30):
        c = make(scale_sigma=sg)
        for _ in range(30):
            c.depth(D_TRUE)
            c.reset(IDS)
        th = sg * math.sqrt(2 / math.pi)
        err = abs(c.realized_absrel - th) / th
        worst = max(worst, err)
        print(f"{'scale σ=' + str(sg):>20}{c.realized_absrel:>9.4f}{th:>9.4f}{err*100:>7.1f}%")
    for rn in (0.05, 0.10, 0.20):
        c = make(rel_noise=rn)
        for _ in range(30):
            c.depth(D_TRUE)
        th = rn * math.sqrt(2 / math.pi)
        err = abs(c.realized_absrel - th) / th
        worst = max(worst, err)
        print(f"{'noise r=' + str(rn):>20}{c.realized_absrel:>9.4f}{th:>9.4f}{err*100:>7.1f}%")
    assert worst < 0.10, f"AbsRel 측정이 이론값과 {worst*100:.0f}% 어긋난다"

    print("\n[4] 잡음이 거리에 비례하는가")
    c = make(rel_noise=0.10)
    near, far = torch.full((E, 8, 8), 1.0), torch.full((E, 8, 8), 3.0)
    en = (c.depth(near) - near).abs().mean().item()
    ef = (c.depth(far) - far).abs().mean().item()
    print(f"   1 m에서 {en:.4f} m, 3 m에서 {ef:.4f} m → 비율 {ef/en:.2f} (기대 3.00)")
    assert 2.7 < ef / en < 3.3, "거리 비례가 아니다"

    print("\n[5] pose 드리프트 = random walk (25결정 후 std = rate·√25)")
    for pd in (0.005, 0.02):
        c = make(pos_drift_std=pd)
        for _ in range(25):
            c.step()
        meas, th = c.pos_offset.std().item(), pd * math.sqrt(25)
        print(f"   {pd*100:.1f} cm/결정 → 축별 std {meas*100:.2f} cm "
              f"(이론 {th*100:.2f} cm), |δ| 평균 {c.pos_offset.norm(dim=-1).mean()*100:.1f} cm")
        assert abs(meas - th) / th < 0.3, "random walk 스케일이 다르다"
        c.reset(IDS)
        assert c.pos_offset.abs().max() == 0, "리셋으로 드리프트가 0이 안 된다"

    print("\n[6] yaw 드리프트가 쿼터니언에 실리는가")
    c = make(yaw_drift_std=0.02)
    for _ in range(25):
        c.step()
    q0 = torch.tensor([[1., 0, 0, 0]]).expand(E, 4)
    _, q = c.apply_pose(torch.zeros(E, 3), q0)
    yaw = 2 * torch.atan2(q[:, 3], q[:, 0])
    th = 0.02 * math.sqrt(25)
    print(f"   누적 yaw std {yaw.std()*57.3:.2f}° (이론 {th*57.3:.2f}°)")
    assert abs(yaw.std().item() - th) / th < 0.3, "yaw 누적이 다르다"

    print("\n[7] 조건 간 누수 방지 — 필드를 되돌리면 AbsRel도 0으로 돌아온다")
    c = make(scale_sigma=0.2, rel_noise=0.1)
    c.depth(D_TRUE)
    assert c.realized_absrel > 0.1
    c.cfg.scale_sigma = 0.0
    c.cfg.rel_noise = 0.0
    c.reset(IDS)
    c.clear_stats()
    c.depth(D_TRUE)
    print(f"   되돌린 뒤 AbsRel = {c.realized_absrel:.6f}")
    assert c.realized_absrel < 1e-6, "이전 조건이 남아 있다"

    print("\n전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
