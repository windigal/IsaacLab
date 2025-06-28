import joblib
import torch
from pathlib import Path
from .utils import leg_joint_idx

PROJECT_ROOT = Path(__file__).resolve().parents[7]
DATA_DIR = PROJECT_ROOT / "data"

def get_interpolated_value(phi: torch.Tensor, loop_range: tuple, im_data: torch.Tensor) -> torch.Tensor:
    """
    Convert phi to a two-hot encoded tensor.
    """
    # 保证 im_data 为 tensor 且在同一设备上
    if not isinstance(im_data, torch.Tensor):
        im_data = torch.tensor(im_data, device=phi.device)
    elif im_data.device != phi.device:
        im_data = im_data.to(phi.device)
    index = (phi * (loop_range[1] - loop_range[0])) + loop_range[0]
    lower = torch.floor(index).long()
    upper = torch.ceil(index).long()
    frac = index - lower.float()
    ref = (1 - frac) * im_data[lower] + frac * im_data[upper]
    return ref


def walk_gait_ref(phi, num_envs, action_dim, device, asset, robot_name) -> torch.Tensor:
    motion_file = DATA_DIR / "kuavo42" / "walk.pkl"
    motion_data = joblib.load(motion_file)
    id = list(motion_data.keys())[8]
    loop_range = (10, 44)
    ref_dof_pos = torch.zeros(num_envs, action_dim, device=device)
    joint_ids = leg_joint_idx(robot_name, ['left_hip_pitch', 'right_hip_pitch',
                                        'left_knee', 'right_knee', 
                                        'left_ankle_pitch', 'right_ankle_pitch'])
    ref_dof_pos[:, joint_ids[0]] = get_interpolated_value(phi[:, 0], loop_range, motion_data[id]["dof"][:,2])
    ref_dof_pos[:, joint_ids[2]] = get_interpolated_value(phi[:, 0], loop_range, motion_data[id]["dof"][:,3])
    ref_dof_pos[:, joint_ids[4]] = get_interpolated_value(phi[:, 0], loop_range, motion_data[id]["dof"][:,4])
    ref_dof_pos[:, joint_ids[1]] = get_interpolated_value(phi[:, 0], loop_range, motion_data[id]["dof"][:,8])
    ref_dof_pos[:, joint_ids[3]] = get_interpolated_value(phi[:, 0], loop_range, motion_data[id]["dof"][:,9])
    ref_dof_pos[:, joint_ids[5]] = get_interpolated_value(phi[:, 0], loop_range, motion_data[id]["dof"][:,10])
    return ref_dof_pos