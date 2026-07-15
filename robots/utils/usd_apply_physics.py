"""
brov2_custom.usdc  ->  brov2_custom_physics.usda

Adds ONLY what Blender cannot write:
    - UsdPhysics.RigidBodyAPI   on /BROV2_Heavy
    - UsdPhysics.MassAPI        on /BROV2_Heavy   (mass, COM, diagonal inertia)
    - UsdPhysics.CollisionAPI   on the two collision boxes
    - collisionEnabled = False  on the visual mesh

Nothing else is touched: geometry, frames, materials, custom properties
(userProperties:axis / spin / role) all pass through unchanged.

Run with:
    pip install usd-core
    python add_mass.py
"""

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
import os

SRC = "brov2_custom.usdc"
DST = "brov2_custom_physics.usda"
BODY = "/BROV2_Heavy"

# ── measured values (Fusion; mass cross-checked on a scale) ─────────────────
MASS    = 14.635                    # kg
COM     = (0.001, 0.000, 0.003)     # m, body frame
INERTIA = (0.289, 0.329, 0.337)     # kg m^2, diagonal -- NO added mass baked in

path_SRC = os.path.join("../data/BROV2",SRC)
path_DST = os.path.join("../data/BROV2",DST)

stage = Usd.Stage.Open(path_SRC)
assert stage, f"cannot open {path_SRC}"

body = stage.GetPrimAtPath(BODY)
assert body, f"{BODY} not found -- check the prim path in {SRC}"

# ── rigid body + mass ─────────────────────────────────────────────────────
UsdPhysics.RigidBodyAPI.Apply(body)

mp = UsdPhysics.MassAPI.Apply(body)
mp.CreateMassAttr().Set(MASS)
mp.CreateCenterOfMassAttr().Set(Gf.Vec3f(*COM))
mp.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*INERTIA))
mp.CreatePrincipalAxesAttr().Set(Gf.Quatf(1, 0, 0, 0))   # inertia already axis-aligned

# ── visual mesh must not collide ─────────────────────────────────────────
mesh = stage.GetPrimAtPath(f"{BODY}/BROV2_Heavy_mesh")
assert mesh, "visual mesh not found"
mesh.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)

# ── collision boxes: find the Cube prim under each collision_* Xform ────────
found = 0
for child in body.GetChildren():
    if not child.GetName().startswith("collision_"):
        continue
    cubes = [c for c in child.GetChildren() if c.IsA(UsdGeom.Mesh)]
    if not cubes:
        print(f"  !! no mesh under {child.GetPath()}")
        continue
    UsdPhysics.CollisionAPI.Apply(cubes[0])
    found += 1
    print(f"  collider -> {cubes[0].GetPath()}")

assert found > 0, "no collision_* boxes found -- check names in the USD"

stage.GetRootLayer().Export(path_DST)

print()
print(f"mass            = {MASS} kg")
print(f"centerOfMass    = {COM}")
print(f"diagonalInertia = {INERTIA}")
print(f"colliders       = {found}")
print(f"-> {path_DST}")
