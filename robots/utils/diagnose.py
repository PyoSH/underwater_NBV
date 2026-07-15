from pxr import Usd, UsdGeom, UsdPhysics

STAGE_PATH = "/workspace/OceanRL_test/robots/data/BROV2/BROV_0706 (copy).usd"

stage = Usd.Stage.Open(STAGE_PATH)
print("=== stage upAxis:", UsdGeom.GetStageUpAxis(stage), "===\n")

def dump(prim, depth=0):
    indent = "  " * depth
    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps() if xformable else []
    op_strs = []
    for op in ops:
        try:
            v = op.Get(Usd.TimeCode.Default())
        except Exception as e:
            v = f"<err:{e}>"
        op_strs.append(f"{op.GetOpName()}={v}")

    tags = []
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        tags.append("RigidBody")
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        tags.append("ArticulationRoot")
    if prim.IsA(UsdPhysics.Joint):
        tags.append(f"Joint({prim.GetTypeName()})")

    print(f"{indent}{prim.GetPath()}  [{prim.GetTypeName()}]  {tags}")
    if op_strs:
        print(f"{indent}  xformOps: {op_strs}")

    for child in prim.GetChildren():
        dump(child, depth + 1)

root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
dump(root)
