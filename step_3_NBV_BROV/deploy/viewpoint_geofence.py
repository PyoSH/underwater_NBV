"""시점 유효성 검사와 최소 변경 사영. 2026-09-10.

왜 필요한가: Isaac은 표적 물체의 충돌이 꺼져 있어(`tools/convert_obj_to_usd.py:61`
`collision_enabled=False`) 정책이 물체를 관통하는 (theta, phi, psi)를 낼 수 있고
벌을 받은 적이 없다. 수조에서는 실제로 부딪히므로 배포 경로에 사영이 필요하다.

기하 규약 (DEPLOY_3WEEK_PLAN.md §12.0):
  구면 중심 = 물체 **바닥면 원점** (bbox 중심이 아님 — `envs/env.py:171`)
  (theta, phi, psi)가 가리키는 점 = **base_link** (카메라가 아니다).
    `envs/env.py:354` p_target = rock_pos + offset -> _guidance.set_target(root)
    `envs/env.py:851` spawn_pos 도 root. 관측 psi_actual 도 root_pos_w.
    카메라는 body (0.1575, 0.0053, 0.0678) = 물체 쪽으로 0.157 m 더 앞.
  자세는 look_at 이므로 +X_body 가 항상 물체를 향한다
    -> 물체에 가장 가까운 선체 점은 앞면 중앙 = base_link + 0.2285 m
    -> 바닥/천장에 가장 가까운 점은 pitch = -(90 - phi) 로 기울어진 선체 모서리

선체 포락: BlueROV2 Heavy 제조사 외형 457 x 338 x 254 mm.
  Gazebo `bluerov2_heavy` 의 collision box(0.457 x 0.575 x 0.05581)는 부력 플러그인용
  **배수체적** 상자이지 형상이 아니므로 쓰지 않는다(모델 주석에 그렇게 적혀 있다).

물체 포락: 반경 0.801 m, 높이 0.933 m 원기둥 (방위 무관, 보수적).
  실측 발자국 지지반경 0.605 ~ 0.801 m 중 최댓값. 안전 함수는 단순하고 항상
  보수적인 쪽이 낫다 — 대가는 좁은 방위에서 최대 0.196 m 더 밀어내는 것뿐이다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ── 물체 포락 (원기둥) ──
OBJ_HEIGHT_M = 0.933
OBJ_RADIUS_M = 0.801

# ── 선체 포락 (BlueROV2 Heavy, 제조사 외형) ──
HULL_HALF_LEN_M = 0.2285      # X (전후)
HULL_HALF_WID_M = 0.169       # Y (좌우)
HULL_HALF_HGT_M = 0.127       # Z (상하)

SAFETY_M = 0.15               # 위 포락 위에 얹는 여유
FLOOR_CLEAR_M = 0.10          # 선체 최하점과 바닥 사이 최소 간격
SURFACE_CLEAR_M = 0.15        # 선체 최상점과 수면 사이 최소 간격

# 물체 쪽 구속: look_at 이라 앞면이 물체를 향한다
REACH_TOWARD_OBJ_M = HULL_HALF_LEN_M + SAFETY_M       # 0.3785


@dataclass(frozen=True)
class PoolBox:
    """pool 프레임 안전 상자. 원점 = 수조 바닥 중앙(= 물체 바닥면 원점).

    수조 6(Y) x 10(X) x 3 m, 수면 2.7 m.
    x/y 한계 = 반치수 - 선체 반대각(0.283) - 여유 0.3.
    """
    x_abs_max: float = 4.4
    y_abs_max: float = 2.4
    water_depth_m: float = 2.7


def pitch_rad(phi: float) -> float:
    """look_at 자세의 차체 pitch 크기 [rad]. phi=10deg -> 80deg 기수 하향."""
    return abs(math.pi / 2.0 - phi)


def hull_vertical_reach(phi: float) -> float:
    """기울어진 선체가 base_link 위/아래로 뻗는 거리 [m] (상하 대칭)."""
    p = pitch_rad(phi)
    return HULL_HALF_HGT_M * math.cos(p) + HULL_HALF_LEN_M * math.sin(p)


def clearance_to_object(r: float, z: float) -> float:
    """base_link (수평거리 r, 높이 z) 에서 물체 원기둥 표면까지 거리 [m].

    원기둥 {r' <= R, 0 <= z' <= H} 에 대한 정확한 점-입체 거리. 내부면 0.
    """
    dr = r - OBJ_RADIUS_M
    dz_up = z - OBJ_HEIGHT_M
    dz_dn = -z
    if dr <= 0.0:
        return max(dz_up, dz_dn, 0.0)
    if dz_up > 0.0:
        return math.hypot(dr, dz_up)
    if dz_dn > 0.0:
        return math.hypot(dr, dz_dn)
    return dr


def view_position(theta: float, phi: float, psi: float) -> tuple[float, float, float]:
    """구면 -> pool 직교 (base_link). `envs/env.py::_pre_physics_step` 와 동일."""
    return (psi * math.sin(phi) * math.cos(theta),
            psi * math.sin(phi) * math.sin(theta),
            psi * math.cos(phi))


def psi_min_object(phi: float, tol: float = 1e-4) -> float:
    """물체 포락을 벗어나는 최소 psi. 원점에서 뻗는 광선 위에서 거리는 단조라 이분법."""
    lo, hi = 0.0, 10.0
    s, c = math.sin(phi), math.cos(phi)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if clearance_to_object(mid * s, mid * c) >= REACH_TOWARD_OBJ_M:
            hi = mid
        else:
            lo = mid
    return hi


def psi_min_floor(phi: float) -> float:
    """바닥 간격을 지키는 최소 psi. z = psi cos(phi) 이므로 닫힌 형태."""
    c = math.cos(phi)
    if c <= 1e-9:
        return math.inf
    return (hull_vertical_reach(phi) + FLOOR_CLEAR_M) / c


def psi_max_surface(phi: float, pool: PoolBox = PoolBox()) -> float:
    """수면 간격을 지키는 최대 psi."""
    c = math.cos(phi)
    z_cap = pool.water_depth_m - SURFACE_CLEAR_M - hull_vertical_reach(phi)
    return math.inf if c <= 1e-9 else z_cap / c


def psi_min_safe(phi: float) -> float:
    """물체·바닥을 동시에 만족하는 최소 psi [m]."""
    return max(psi_min_object(phi), psi_min_floor(phi))


def is_valid(theta: float, phi: float, psi: float,
             pool: PoolBox = PoolBox()) -> tuple[bool, str]:
    if psi < psi_min_object(phi) - 1e-6:
        return False, "object_envelope"
    if psi < psi_min_floor(phi) - 1e-6:
        return False, "pool_floor"
    if psi > psi_max_surface(phi, pool) + 1e-6:
        return False, "pool_surface"
    x, y, _ = view_position(theta, phi, psi)
    if abs(x) > pool.x_abs_max or abs(y) > pool.y_abs_max:
        return False, "pool_wall"
    return True, ""


def project(theta: float, phi: float, psi: float,
            psi_min: float, psi_max: float,
            pool: PoolBox = PoolBox()) -> tuple[float, float, float, str]:
    """최소 변경 사영. theta/phi는 보존하고 psi만 민다.

    왜 psi만 미는가: theta/phi를 건드리면 정책이 고른 '방향'이 바뀌어 액션 의미가
    무너진다. 물체·바닥 구속은 모두 psi 에 대해 단조(밀어내면 완화)라 이것으로 충분하다.

    반환 reason: "" 무보정 / "object_envelope"·"pool_floor" psi를 밀어 고쳤음 /
    그 외는 사영으로 못 고침 -> 호출자가 거부해야 한다.
    """
    need_obj = psi_min_object(phi)
    need_flr = psi_min_floor(phi)
    need = max(need_obj, need_flr, psi_min)
    reason = ""
    if need > psi + 1e-6:
        reason = "object_envelope" if need_obj >= need_flr else "pool_floor"
    psi_new = min(max(psi, need), psi_max)
    ok, why = is_valid(theta, phi, psi_new, pool)
    if not ok:
        reason = why
    return theta, phi, psi_new, reason


# ── 경로(chord) 검사와 호 분할 (2026-09-10, Phase 4 ②) ──
#
# guidance 는 두 pose 사이를 **직선**으로 간다. 끝점이 유효해도 chord 가 물체
# 원기둥을 지날 수 있다: psi=1.0 에서 theta/phi 를 동시에 30° 옮기면 chord 중점이
# 중심에서 0.93 m, 선체 reach 0.38 을 빼면 0.55 < R 0.80. 바닥·수면·벽은 볼록 상자라
# 끝점이 유효하면 chord 도 유효하므로 물체 원기둥만 검사한다.
#
# 이동 중 자세는 slew 중이라 앞면이 물체를 향한다는 보장이 없다 → 선체 반경은
# 방향 무관하게 REACH_TOWARD_OBJ_M(=최대 반길이+여유) 를 쓴다. 보수적이다.

def segment_clearance(p0, p1, samples: int = 33) -> float:
    """base_link 두 점을 잇는 직선 위에서 물체 원기둥까지의 최소 여유 [m].

    선체 반경(REACH_TOWARD_OBJ_M)을 뺀 값. 음수면 충돌 위험.
    """
    if samples < 2:
        raise ValueError("samples must be >= 2")
    best = math.inf
    for i in range(samples):
        s = i / (samples - 1)
        x = p0[0] + s * (p1[0] - p0[0])
        y = p0[1] + s * (p1[1] - p0[1])
        z = p0[2] + s * (p1[2] - p0[2])
        c = clearance_to_object(math.hypot(x, y), z) - REACH_TOWARD_OBJ_M
        if c < best:
            best = c
    return best


def segment_is_clear(p0, p1, start_inside_ok: bool = False) -> bool:
    """chord 가 물체 포락을 비끼는가.

    start_inside_ok: 첫 chord 의 시작점은 **측정 pose** 라 도착 오차(수 cm)만큼 포락 경계
    안쪽일 수 있다(run #13 실측 → reject_chord). 그 경우 시작점 자체의 여유는 묻지 않고,
    chord 위 어느 점도 시작점보다 나빠지지 않으면(멀어지기만 하면) 통과시킨다.
    """
    if not start_inside_ok:
        return segment_clearance(p0, p1) >= 0.0
    c0 = clearance_to_object(math.hypot(p0[0], p0[1]), p0[2]) - REACH_TOWARD_OBJ_M
    floor = min(0.0, c0)
    samples = 33
    for i in range(1, samples):
        t = i / (samples - 1)
        x = p0[0] + t * (p1[0] - p0[0]); y = p0[1] + t * (p1[1] - p0[1]); z = p0[2] + t * (p1[2] - p0[2])
        if clearance_to_object(math.hypot(x, y), z) - REACH_TOWARD_OBJ_M < floor - 1e-9:
            return False
    return True


def shortest_theta_delta(theta0: float, theta1: float) -> float:
    """theta0 -> theta1 최단 방위 변화 [rad] in (-pi, pi]."""
    d = (theta1 - theta0) % (2.0 * math.pi)
    return d - 2.0 * math.pi if d > math.pi else d


# 경유점을 포락 경계 **위**가 아니라 살짝 밖에 둔다: 경계 위 두 점을 잇는 chord 는
# (경계가 볼록하므로) 항상 안으로 처지고, psi_min_safe 의 이분법 오차(1e-4)도 있다.
# 7° 간격 chord 의 처짐은 1.6 m 에서 3 mm — 2 cm 면 충분하다.
ARC_LIFT_MARGIN_M = 0.02


def arc_split(sph0, sph1, max_pieces: int = 6):
    """구면좌표 두 시점 사이를, 각 chord 가 물체를 비끼는 **최소 조각 수**로 나눈다.

    반환 (경유 시점 목록[(theta,phi,psi)...] 끝점 포함, 조각 수). 못 나누면 (None, max_pieces).
    theta 는 최단 방위 경로, phi/psi 는 선형 보간 — 구면 위 호에 가깝다.
    sph0 는 측정 pose 에서 되읽은 값이라 구면 밖(psi 범위 밖)일 수 있다; 상관없다.
    """
    th0, ph0, ps0 = sph0
    th1, ph1, ps1 = sph1
    dth = shortest_theta_delta(th0, th1)
    for k in range(1, max_pieces + 1):
        pts = []
        for i in range(1, k + 1):
            th = (th0 + dth * i / k) % (2.0 * math.pi)
            ph = ph0 + (ph1 - ph0) * i / k
            ps = ps0 + (ps1 - ps0) * i / k
            # 중간점도 끝점과 같은 사영을 받는다: psi_min_safe 는 phi 에 따라 최대
            # 1.61 m(phi 41°)까지 올라가므로, phi 를 가로지르는 호는 그 등고선을
            # 따라 psi 를 밀어 올려야 chord 꼭짓점이 포락 안에 들어가지 않는다.
            if i < k:
                ps = max(ps, psi_min_safe(ph) + ARC_LIFT_MARGIN_M)
            pts.append((th, ph, ps))
        chain = [view_position(*sph0)] + [view_position(*s) for s in pts]
        if all(segment_is_clear(chain[i], chain[i + 1], start_inside_ok=(i == 0))
               for i in range(len(chain) - 1)):
            return pts, k
    return None, max_pieces
