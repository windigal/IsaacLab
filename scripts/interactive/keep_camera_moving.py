import numpy as np
from scipy.spatial.transform import Rotation as R
from isaacsim.core.api.world import World
world = World()

import omni.usd
stage = omni.usd.get_context().get_stage()

from pxr import UsdGeom, Gf

# base_link_prim_path = "/base_link/base_link"
# camera_prim_path = "/base_link/Camera"
base_link_prim_path = "/World/envs/env_0/Robot/base_link"
camera_prim_path = "/World/envs/env_0/Robot/Camera"

def get_father_transform(prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    translate = np.array(ops[0].Get(0), np.float32)
    quatf = ops[1].Get(0)
    orientation = np.array([quatf.GetImaginary()[0], quatf.GetImaginary()[1], quatf.GetImaginary()[2], quatf.GetReal()], np.float32)
    return translate, orientation

def set_father_transform(prim_path, translate, orientation):
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    ops[0].Set(Gf.Vec3f(translate[0], translate[1], translate[2]))
    quat = Gf.Quatd(orientation[3], orientation[0], orientation[1], orientation[2])
    # quat = Gf.Quatd(quat)# .Normalize()
    ops[1].Set(quat)

def get_camera_and_base_link_transform():
    """ 在Pause下获取转移矩阵
    translate: 父xform到子xform的平移向量
    orientation: 父xform到子xform的四元数 (x,y,z,w)
    """
    t1, o1 = get_father_transform(base_link_prim_path)
    R1 = R.from_quat(o1)
    print(t1, o1, R1.as_euler("xyz", degrees=True))
    t2, o2 = get_father_transform(camera_prim_path)
    R2 = R.from_quat(o2)
    print(t2, o2, R2.as_euler("xyz", degrees=True))
    # 从base_link到camera的变换矩阵
    dt = t2 - t1
    dR = R2 * R1.inv()
    print("Final translate and xyz rotation", dt, dR.as_euler("xyz", degrees=True))

# get_camera_and_base_link_transform()

dt = np.array([-0.012, 4.386, 1.415])
dR = R.from_euler("xyz", [75, 0, 180], degrees=True)

def set_camera(_):
    t, o = get_father_transform(base_link_prim_path)
    new_t = dt + t
    new_R = dR * R.from_quat(o)
    x, y, z = new_R.as_euler("xyz", degrees=True)
    # print(x, y, z)
    new_orientation = R.from_euler("xyz", [75, 0, 180], degrees=True).as_quat()
    # print(R.from_quat(new_orientation).as_euler("xyz", degrees=True))
    # print(new_t, new_orientation, new_R.as_euler("xyz", degrees=True))
    set_father_transform(camera_prim_path, new_t, new_orientation)

world.add_render_callback("set_camera", set_camera)
# set_camera(None)
