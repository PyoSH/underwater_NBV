"""
BlueROV CAD -> IsaacLab base_link USD 빌드 파이프라인
=====================================================

BROV2 Heavy (BROV_0706 / final-N-BouyantFoam_부력) 재구축 과정에서 만든 절차를
새로운 BlueROV CAD 모델에도 적용할 수 있도록 일반화한 스크립트.

필요한 입력 파일 (보통 2개, CAD에서 별도로 export):
  1. HULL_USD : 계층 구조가 있는 원본 (개별 thruster mesh + joint 포함,
                외형이 시각적으로 정확한 원본). 스러스터 위치/개수 파악용.
  2. BUOY_USD : "내압용기 내부까지 채운" watertight solid 버전
                (Fusion360 등에서 부력 체적 계산용으로 별도 export).
                최종 base_link mesh 소스로 사용.

이 스크립트가 하는 일:
  1. BUOY_USD의 모든 solid를 world-space로 구워서 base_link 하나로 병합
     (visuals + collisions, convexHull collision 근사)
  2. HULL_USD의 조인트 이름 패턴으로 스러스터 위치를 찾아 참조 Xform 생성
     (mesh는 만들지 않음 — 이미 BUOY_USD 병합 mesh 안에 포함되어 있음)
  3. 센서류(이름 키워드 매칭)도 동일하게 위치만 참조 Xform으로 생성
  4. 축을 raw 소스 그대로 방치하지 않고, 지정한 permutation으로 한 번에
     구워서 upAxis=Z self-contained 파일로 저장 (스폰 시점의 암묵적 변환에 의존 X)
  5. (선택) Fusion360 "물리적 특성" 리포트에서 얻은 실측 질량/무게중심/관성을
     MassAPI로 base_link에 직접 적용

사용법 (파일 맨 아래 CONFIG 블록만 새 모델에 맞게 수정 후 실행):
    /isaac-sim/python.sh build_base_link.py
  또는 pxr이 Kit 익스텐션 안에만 있는 Isaac Sim 환경이라면 실행 전에:
    EXT=$(ls -d /isaac-sim/extscache/omni.usd.libs-*/ | head -1)
    export PYTHONPATH="${EXT}:${PYTHONPATH}"
    export LD_LIBRARY_PATH="${EXT}bin:${LD_LIBRARY_PATH}"

주의 (BROV2 재구축 과정에서 실제로 겪었던 함정들):
  - USD joint의 physics:localRot0 는 실제 추력 방향과 일치한다는 보장이 없다.
    (BROV2에서 직접 확인: joint 회전으로 역산한 추력방향이 test_straight_line으로
     검증된 값과 전혀 안 맞았음) thrust:direction은 반드시 실제 동역학 테스트로
    검증한 뒤 THRUSTERS 설정의 direction 필드에 채워 넣을 것 — 이 스크립트는
    direction을 비워두거나(None) 이미 검증된 값이 있을 때만 채운다.
  - USD stage의 upAxis 메타데이터는 실제 형상과 다를 수 있다(라벨 오류 사례 확인됨).
    라벨을 믿지 말고, 반드시 bbox 크기 비교 등으로 실제 축 배치를 직접 확인한 뒤
    AXIS_PERMUTATION을 설정할 것.
  - 질량 정보 없이 두면 PhysX가 collision mesh convex hull 부피 x 기본 밀도로
    질량을 자동 추정한다 — 이건 의도된 값이 아니므로, CAD의 실측 질량/무게중심/
    관성 리포트가 있다면 반드시 MASS_PROPS에 채워 넣을 것.
"""

from dataclasses import dataclass, field
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
import os


# ==============================================================================
# 유틸 함수 (모델에 무관하게 재사용 가능)
# ==============================================================================

def make_axis_permutation(order: str) -> Gf.Matrix3d:
    """
    'zxy' 같은 문자열을 axis permutation 행렬로 변환.
    new_x, new_y, new_z 가 각각 raw 소스의 어느 축에서 오는지를 지정.
    예: 'zxy' -> new_x=raw_z, new_y=raw_x, new_z=raw_y (BROV2 Y-up -> Z-up 에 사용한 것)
    'xyz' -> 항등(변환 없음, 이미 Z-up으로 잘 export된 경우)
    """
    idx = {'x': 0, 'y': 1, 'z': 2}
    rows = []
    for ch in order.lower():
        row = [0.0, 0.0, 0.0]
        row[idx[ch]] = 1.0
        rows.append(row)
    m = Gf.Matrix3d(*[v for row in rows for v in row])
    det = m.GetDeterminant()
    if abs(det - 1.0) > 1e-6:
        raise ValueError(
            f"axis_permutation '{order}' 의 determinant={det:.3f} (반사 변환 포함). "
            "오른손 좌표계를 유지하는 permutation만 허용 (예: 'zxy', 'yzx', 'xyz')."
        )
    return m


def permute_point(m: Gf.Matrix3d, p) -> Gf.Vec3d:
    return m * Gf.Vec3d(p[0], p[1], p[2])


def permute_rotation(m: Gf.Matrix3d, quat) -> Gf.Quatd:
    old_m = Gf.Matrix3d(Gf.Rotation(quat))
    new_m = m * old_m * m.GetTranspose()
    return new_m.ExtractRotation().GetQuat()


def merge_all_solids(buoy_usd_path: str, axis_perm: Gf.Matrix3d):
    """BUOY_USD 안의 모든 UsdGeom.Mesh를 world-space로 구워서(m 단위, axis 변환 적용)
    하나로 병합. returns (points[Vec3f], faceVertexCounts, faceVertexIndices)."""
    stage = Usd.Stage.Open(buoy_usd_path)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)

    all_points, all_counts, all_indices = [], [], []
    vert_offset = 0
    mesh_count = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts_local = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts_local or not counts or not idx:
            continue
        xform = mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for pt in pts_local:
            wp = xform.Transform(pt)
            wp_m = (wp[0] * mpu, wp[1] * mpu, wp[2] * mpu)
            pv = permute_point(axis_perm, wp_m)
            all_points.append(Gf.Vec3f(pv[0], pv[1], pv[2]))
        all_counts.extend(counts)
        all_indices.extend(i + vert_offset for i in idx)
        vert_offset += len(pts_local)
        mesh_count += 1

    print(f"[merge_all_solids] {mesh_count} solids -> {len(all_points)} points, {len(all_counts)} faces")
    return all_points, all_counts, all_indices


def compute_enclosed_volume_m3(buoy_usd_path: str) -> float:
    """world-space divergence theorem으로 전체 체적(m^3) 계산.
    CAD 자체 물성 리포트(Fusion360 '물리적 특성' 등)가 있으면 그 값이 더 정확하므로
    이 계산값은 어디까지나 교차검증용으로만 사용할 것."""
    stage = Usd.Stage.Open(buoy_usd_path)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    total = 0.0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts_local = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts_local or not counts or not idx:
            continue
        xform = mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pts = [xform.Transform(p) for p in pts_local]
        vol = 0.0
        ptr = 0
        for c in counts:
            face = idx[ptr:ptr + c]
            ptr += c
            v0 = pts[face[0]]
            for i in range(1, c - 1):
                v1, v2 = pts[face[i]], pts[face[i + 1]]
                vol += (v0[0] * (v1[1] * v2[2] - v1[2] * v2[1])
                        - v0[1] * (v1[0] * v2[2] - v1[2] * v2[0])
                        + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0]))
        total += abs(vol) / 6.0
    return total * (mpu ** 3)


def find_reference_positions(hull_usd_path: str, joints_scope_path: str, axis_perm: Gf.Matrix3d):
    """HULL_USD의 조인트들에서 body0(부모) 기준 localPos0를 뽑아 axis 변환.
    조인트 이름 -> permuted position dict 로 반환. 스러스터 식별/위치 확보용."""
    stage = Usd.Stage.Open(hull_usd_path)
    joints_prim = stage.GetPrimAtPath(joints_scope_path)
    if not joints_prim.IsValid():
        raise ValueError(f"joints scope not found: {joints_scope_path}")

    result = {}
    for j in joints_prim.GetChildren():
        pos0 = j.GetAttribute('physics:localPos0').Get()
        if pos0 is None:
            continue
        result[j.GetName()] = permute_point(axis_perm, pos0)
    return result


# ==============================================================================
# 빌드 본체
# ==============================================================================

@dataclass
class ThrusterSpec:
    name: str                      # 예: 'thruster_1_ccw'
    position: tuple                # (x,y,z) m, base_link-local, 이미 axis 변환된 값
    direction: tuple | None = None  # (x,y,z) 단위벡터. **실제 동역학 테스트로 검증 전이면 None으로 둘 것**


@dataclass
class SensorSpec:
    name: str
    position: tuple


@dataclass
class MassProps:
    """CAD 물성 리포트(Fusion360 등)에서 그대로 옮겨 적는 용도.
    principal_axes_deg 는 작은 각도(수 도 이내)면 무시하고 identity로 근사해도
    충분한 경우가 많음 — 근사할 경우 반드시 로그로 남길 것."""
    mass_kg: float
    center_of_mass_m: tuple         # base_link-local, axis 변환 후 값
    diagonal_inertia_kgm2: tuple    # (I1,I2,I3), axis 변환 후 값
    principal_axes_quat: tuple = (1.0, 0.0, 0.0, 0.0)  # (w,x,y,z), 기본은 identity 근사


def build(
    buoy_usd_path: str,
    output_usd_path: str,
    axis_permutation: str,
    thrusters: list,
    sensors: list,
    mass_props: MassProps | None = None,
):
    axis_perm = make_axis_permutation(axis_permutation)

    points, counts, indices = merge_all_solids(buoy_usd_path, axis_perm)
    volume_m3 = compute_enclosed_volume_m3(buoy_usd_path)
    print(f"[build] mesh-integrated volume check: {volume_m3:.8f} m^3 "
          f"(CAD 물성 리포트 값과 비교해서 몇 % 이내로 맞는지 확인할 것)")

    if os.path.exists(output_usd_path):
        os.remove(output_usd_path)
    stage = Usd.Stage.CreateNew(output_usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = stage.DefinePrim('/Root', 'Xform')
    stage.SetDefaultPrim(root)

    base_link = stage.DefinePrim('/Root/base_link', 'Xform')
    UsdPhysics.RigidBodyAPI.Apply(base_link)
    UsdPhysics.ArticulationRootAPI.Apply(base_link)

    # visuals
    mesh_prim = UsdGeom.Mesh.Define(stage, '/Root/base_link/visuals/merged_mesh')
    mesh_prim.CreatePointsAttr(points)
    mesh_prim.CreateFaceVertexCountsAttr(counts)
    mesh_prim.CreateFaceVertexIndicesAttr(indices)
    mesh_prim.CreateSubdivisionSchemeAttr('none')

    mat = UsdShade.Material.Define(stage, '/Root/base_link/visuals/DefaultMaterial')
    shader = UsdShade.Shader.Define(stage, '/Root/base_link/visuals/DefaultMaterial/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.6, 0.6, 0.62))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.5)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(mat)

    # collisions (동일 mesh, convexHull 근사 — collision_from_visuals 방식)
    coll_mesh = UsdGeom.Mesh.Define(stage, '/Root/base_link/collisions/merged_mesh_collision')
    coll_mesh.CreatePointsAttr(points)
    coll_mesh.CreateFaceVertexCountsAttr(counts)
    coll_mesh.CreateFaceVertexIndicesAttr(indices)
    coll_mesh.CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI.Apply(coll_mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(coll_mesh.GetPrim()).CreateApproximationAttr('convexHull')

    # 질량/무게중심/관성 (CAD 실측값이 있을 때만)
    if mass_props is not None:
        mass_api = UsdPhysics.MassAPI.Apply(base_link)
        mass_api.CreateMassAttr(mass_props.mass_kg)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*mass_props.center_of_mass_m))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*mass_props.diagonal_inertia_kgm2))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(*mass_props.principal_axes_quat))
    else:
        print("[build] WARNING: mass_props 없음 -> PhysX가 collision mesh 부피 x 기본 밀도로 "
              "질량을 자동 추정합니다. CAD 물성 리포트를 받으면 반드시 채워 넣을 것.")

    # 스러스터 참조 Xform (mesh 없음 — 이미 merged_mesh 안에 포함됨)
    for t in thrusters:
        p = stage.DefinePrim(f'/Root/base_link/{t.name}', 'Xform')
        UsdGeom.Xformable(p).AddTranslateOp().Set(Gf.Vec3d(*t.position))
        if t.direction is not None:
            attr = p.CreateAttribute('thrust:direction', Sdf.ValueTypeNames.Vector3f, custom=True)
            attr.Set(Gf.Vec3f(*t.direction))
        else:
            print(f"[build] NOTE: {t.name} 의 thrust:direction 미설정 — "
                  f"test_straight_line 등으로 검증 후 채울 것.")

    # 센서 참조 Xform
    for s in sensors:
        p = stage.DefinePrim(f'/Root/base_link/{s.name}', 'Xform')
        UsdGeom.Xformable(p).AddTranslateOp().Set(Gf.Vec3d(*s.position))

    stage.GetRootLayer().Save()
    prim_count = len(list(stage.Traverse()))
    print(f"[build] exported to {output_usd_path}")
    print(f"[build] file size: {os.path.getsize(output_usd_path) / 1e6:.2f} MB, "
          f"prim count: {prim_count}")


# ==============================================================================
# CONFIG — 새 CAD 모델 적용 시 이 아래만 수정
# ==============================================================================

if __name__ == '__main__':
    DATA_DIR = '/workspace/OceanRL_test/robots/data/BROV2'

    # 1) 축 변환: raw 소스가 실제로 어떤 배치인지 bbox 비교로 먼저 확인할 것.
    #    BROV2는 raw Y-up 이었고 (Xu,Yu,Zu) -> IsaacLab Z-up (Zu,Xu,Yu) 였으므로 'zxy'.
    #    소스가 이미 올바른 Z-up이면 'xyz'로 둘 것 (permute 안 함).
    AXIS_PERMUTATION = 'zxy'

    # 2) 스러스터 — 위치는 HULL_USD의 joint localPos0에서 자동 추출 가능
    #    (find_reference_positions 사용), direction은 반드시 실제 테스트로 검증 후 채울 것.
    #    아래는 BROV2 Heavy 예시 (이미 test_straight_line으로 검증된 _DIR 값을 넣은 경우).
    thruster_positions = find_reference_positions(
        hull_usd_path=f'{DATA_DIR}/BROV_0706 (copy).usd',
        joints_scope_path='/Root/final_N_BouyantFoam_eng/Joints',
        axis_perm=make_axis_permutation(AXIS_PERMUTATION),
    )
    # SNAME(X=fwd,Y=starboard,Z=down) _DIR -> IsaacLab Z-up: (X,-Y,-Z). 검증 완료된 값만 사용.
    _dir_sname = {
        'Joint_thruster_1_CCW': (-0.7071,  0.7071, 0.0),
        'Joint_thruster_2_CCW': (-0.7071, -0.7071, 0.0),
        'Joint_thruster_3_CW':  ( 0.7071,  0.7071, 0.0),
        'Joint_thruster_4_CW':  ( 0.7071, -0.7071, 0.0),
        'Joint_thruster_5_CCW': ( 0.0,     0.0,    1.0),
        'Joint_thruster_6_CW':  ( 0.0,     0.0,    1.0),
        'Joint_thruster_7_CW':  ( 0.0,     0.0,    1.0),
        'Joint_thruster_8_CCW': ( 0.0,     0.0,    1.0),
    }
    THRUSTERS = [
        ThrusterSpec(
            name=name.replace('Joint_', '') + '_stub',
            position=tuple(pos),
            direction=(sx, -sy, -sz) if name in _dir_sname else None,
        )
        for name, pos in thruster_positions.items()
        for (sx, sy, sz) in [_dir_sname.get(name, (None, None, None))]
        if pos is not None
    ]

    # 3) 센서 — 위치를 이미 알고 있다면(이전 rework에서처럼 world bbox center로 구했다면) 여기 채움
    SENSORS = [
        SensorSpec('brov_dvl_link',        (-0.172583, -0.098000, -0.206662)),
        SensorSpec('brov_light_l_link',    ( 0.196233,  0.191768, -0.048609)),
        SensorSpec('brov_light_r_link',    ( 0.196246, -0.184768, -0.048572)),
        SensorSpec('brov_cam_servo_link',  ( 0.160587,  0.032387,  0.071827)),
        SensorSpec('brov_cam_link',        ( 0.157512,  0.005286,  0.067842)),
    ]

    # 4) 질량/무게중심/관성 — CAD "물리적 특성" 리포트(density=1.0 g/cm^3 로 뽑은 값 등)를
    #    그대로 옮겨 적을 것. raw 소스 좌표계 기준값을 axis_perm으로 변환해서 넣는다.
    _perm = make_axis_permutation(AXIS_PERMUTATION)
    _com_raw_m = (0.003442, 0.017983, 0.008446)
    _com = permute_point(_perm, _com_raw_m)
    # 주축 회전각(수 도 이내)이 작아 identity로 근사, I1≈Ixx_raw,I2≈Iyy_raw,I3≈Izz_raw 가정.
    _I1, _I2, _I3 = 291402.185e-6, 346222.775e-6, 277421.046e-6  # kg*mm^2 -> kg*m^2
    MASS_PROPS = MassProps(
        mass_kg=13.093,
        center_of_mass_m=(_com[0], _com[1], _com[2]),
        diagonal_inertia_kgm2=(_I3, _I1, _I2),  # axis 순서도 동일 permutation 적용
        principal_axes_quat=(1.0, 0.0, 0.0, 0.0),
    )

    build(
        buoy_usd_path=f'{DATA_DIR}/final-N-BouyantFoam_부력.usdc',
        output_usd_path=f'{DATA_DIR}/BROV_base_link.usd',
        axis_permutation=AXIS_PERMUTATION,
        thrusters=THRUSTERS,
        sensors=SENSORS,
        mass_props=MASS_PROPS,
    )
