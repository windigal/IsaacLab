import torch
from scipy.interpolate import UnivariateSpline
S = [i / 10 for i in range(1, 11)]

peaks = torch.tensor(
        [[[-0.222904856, -0.398410658], [ 0.74801878,  0.388095905], [-0.151327282, -0.253756036]],
         [[-0.202674384, -0.464732418], [ 0.87288400,  0.346149355], [-0.112754998, -0.273432473]],
         [[-0.125644614, -0.568563143], [ 1.08990022,  0.332829624], [-0.010730416, -0.306470563]],
         [[ 0.060455044, -0.624765793], [ 1.19037282,  0.297708362], [ 0.023672292, -0.353795697]],
         [[ 0.222318540, -0.675252140], [ 1.25288773,  0.208098198], [ 0.067569586, -0.418386022]],
         [[ 0.324639658, -0.709793766], [ 1.27832635,  0.165800378], [ 0.105877625, -0.467730373]],
         [[ 0.432440480, -0.752252241], [ 1.30005475,  0.074556691], [ 0.132878924, -0.478076627]],
         [[ 0.501398434, -0.794730047], [ 1.31349838,  0.005957472], [ 0.144416720, -0.480022808]],
         [[ 0.538123230, -0.833924949], [ 1.33908252, -0.000000553], [ 0.148099209, -0.481795669]],
         [[ 0.564531346, -0.857205749], [ 1.38976808, -0.000004539], [ 0.140434856, -0.492357562]]]
)


interpolation = {'max': [], 'min': []}
for j in range(3):
    max_values = peaks[:, j, 0]
    min_values = peaks[:, j, 1]

    cs_max = UnivariateSpline(S, max_values, s=0.3)
    interpolation['max'].append(cs_max)

    cs_min = UnivariateSpline(S, min_values, s=0.3)
    interpolation['min'].append(cs_min)


def trun_sin(x, min: float, max: float):
    return (torch.sin(x) + 1) / 2 * (max - min) + min

def leg_joint_idx(robot_name: str, joint_name: list[str]) -> list[int]:
    """
    Get the index of a joint in the robot's joint list.
    :param robot_name: Name of the robot.
    :param joint_name: Name of the joint.
    :return: Index of the joint in the robot's joint list.
    """
    robot_joint_list = {
        "G1": ['left_hip_pitch', 'right_hip_pitch', 'left_hip_roll', 
               'right_hip_roll', 'left_hip_yaw', 'right_hip_yaw', 
               'left_knee', 'right_knee', 'left_ankle_pitch', 
               'right_ankle_pitch', 'left_ankle_roll', 'right_ankle_roll'],
        "leju": ['left_hip_roll', 'right_hip_roll', 'left_hip_yaw', 
                  'right_hip_yaw', 'left_hip_pitch', 'right_hip_pitch', 
                  'left_knee', 'right_knee', 'left_ankle_pitch', 
                  'right_ankle_pitch', 'left_ankle_roll', 'right_ankle_roll']
    }
    
    if robot_name not in robot_joint_list:
        raise ValueError(f"Robot '{robot_name}' not found.")
    indices = []
    for j in joint_name:
        try:
            idx = robot_joint_list[robot_name].index(j)
            indices.append(idx)
        except ValueError:
            raise ValueError(f"Joint '{j}' not found in robot '{robot_name}'.")
    return indices

def walk_gait_ref(phi, num_envs, action_dim, device, delta, asset, robot_name) -> torch.Tensor:
    ref_dof_pos = torch.zeros(num_envs, action_dim, device=device)
    joint_ids = leg_joint_idx(robot_name, ['left_hip_pitch', 'right_hip_pitch',
                                        'left_knee', 'right_knee', 
                                        'left_ankle_pitch', 'right_ankle_pitch'])
    ref_dof_pos[:, joint_ids[0]] = trun_sin(2 * torch.pi * phi[:, 0], -0.52 - delta * 0.18, -0.02)
    ref_dof_pos[:, joint_ids[2]] = trun_sin(2 * torch.pi * (phi[:, 0] - 1/2), 0.02, 1.02 + delta * 0.2)
    ref_dof_pos[:, joint_ids[2]] = torch.max(ref_dof_pos[:, 6], asset.data.default_joint_pos[:, 6])
    ref_dof_pos[:, joint_ids[4]] = trun_sin(2 * torch.pi * phi[:, 0], -0.55, -0.05)
    ref_dof_pos[:, joint_ids[4]] = torch.where(ref_dof_pos[:, 8] < asset.data.default_joint_pos[:, 8], 
                                    ref_dof_pos[:, 8], - ref_dof_pos[:, 4] - ref_dof_pos[:, 6])
    # right foot stance phase set to default joint pos, r3, r4, r5
    ref_dof_pos[:, joint_ids[1]] = trun_sin(2 * torch.pi * phi[:, 1], -0.52 - delta * 0.18, -0.02)
    ref_dof_pos[:, joint_ids[3]] = trun_sin(2 * torch.pi * (phi[:, 1] - 1/2), 0.02, 1.02 + delta * 0.2)
    ref_dof_pos[:, joint_ids[3]] = torch.max(ref_dof_pos[:, 7], asset.data.default_joint_pos[:, 7])
    ref_dof_pos[:, joint_ids[5]] = trun_sin(2 * torch.pi * phi[:, 1], -0.55, -0.05)
    ref_dof_pos[:, joint_ids[5]] = torch.where(ref_dof_pos[:, 9] < asset.data.default_joint_pos[:, 9], 
                                    ref_dof_pos[:, 9], - ref_dof_pos[:, 5] - ref_dof_pos[:, 7])
    
    return ref_dof_pos

def run_gait_ref(phi, num_envs, action_dim, device, robot_name) -> torch.Tensor:
    
    def run_ankle(phi):
        y = torch.zeros_like(phi)
        y[(phi > 2/16) & (phi < 12/16)] = 0.1209 * torch.sin(5.9381 * 16/10 * (phi[(phi > 2/16) & (phi < 12/16)] - 2/16) - 3.0487) - 0.2934
        y[(phi >= 12/16) & (phi <= 1)] = 0.2033 * torch.sin(3.6140 * 16/6 * (phi[(phi >= 12/16) & (phi <= 1)] - 12/16) - 0.6329) - 0.0944
        y[phi <= 2/16] = 0.2033 * torch.sin(3.6140 * 16/6 * (phi[phi <= 2/16] + 9/40) - 0.6329) - 0.0944
        return y

    joint_ids = leg_joint_idx(robot_name, ['left_hip_pitch', 'right_hip_pitch',
                                    'left_knee', 'right_knee', 
                                    'left_ankle_pitch', 'right_ankle_pitch'])
    ref_dof_pos = torch.zeros(num_envs, action_dim, device=device)
    ref_dof_pos[:, joint_ids[0]] = -0.3735 * torch.sin(2 * torch.pi * phi[:, 0] + 6.2393) - 0.3703
    ref_dof_pos[:, joint_ids[2]] = -0.6497 * torch.sin(2 * torch.pi * phi[:, 0] + 1.2083) + 1.1049
    ref_dof_pos[:, joint_ids[4]] = -run_ankle(phi[:, 0])
    ref_dof_pos[:, joint_ids[1]] = -0.3735 * torch.sin(2 * torch.pi * phi[:, 1] + 6.2393) - 0.3703
    ref_dof_pos[:, joint_ids[3]] = -0.6497 * torch.sin(2 * torch.pi * phi[:, 1] + 1.2083) + 1.1049
    ref_dof_pos[:, joint_ids[5]] = -run_ankle(phi[:, 1])
    
    return ref_dof_pos

def jump_gait_ref(phi, num_envs, action_dim, device, robot_name) -> torch.Tensor:
    
    def dif(x, min_val, max_val):
        y = torch.zeros_like(x)
        y = torch.where(x < min_val, min_val - x, y)
        y = torch.where(x > max_val, max_val - x, y)
        return y

    def jump_hip(phi):
        return 0.7127 * torch.sin(2 * torch.pi * phi - 0.7077) - 0.9902

    def jump_knee(phi):
        return -0.5537 * torch.sin(34 / 9 * torch.pi *
                            (phi - 3 / 17 + 0.35 * dif(phi, 0.25, 0.6)) + 0.0142) + 1.5105
        
    def linear_fit(phi):
        y = 1.7391 - (1.7391 - 1.3365) / (8/17) * (phi - 12/17 + (phi < 3 / 17).float())
        y = y + (1.6 - y) * torch.minimum(phi - 12/17 + (phi < 3 / 17).float(), 3/17 - phi + (phi > 12 / 17).float()) / (8/17)
        return y
        
    def jump_knee_final(phi):
        y = torch.where((phi > 3/17) & (phi < 12/17), jump_knee(phi), linear_fit(phi))
        return y
        
    def jump_ankle(phi):
        y = torch.zeros_like(phi)
        y[(phi > 5/16) & (phi < 10/16)] = 0.3568 * torch.sin(32/5 * torch.pi * (phi[(phi > 5/16) & (phi < 10/16)] - 5/16) - 1.3156) - 0.2421
        y[(phi >= 10/16) & (phi <= 1)] = 0.1443 * torch.sin(32/11 * torch.pi * (phi[(phi >= 10/16) & (phi <= 1)] - 10/16) - 0.0940) - 0.6516
        y[phi <= 5/16] = 0.1443 * torch.sin(32/11 * torch.pi * (phi[phi <= 5/16] + 11/32) - 0.0940) - 0.6516
        return y

    joint_ids = leg_joint_idx(robot_name, ['left_hip_pitch', 'right_hip_pitch',
                                        'left_knee', 'right_knee', 
                                        'left_ankle_pitch', 'right_ankle_pitch'])
    ref_dof_pos = torch.zeros(num_envs, action_dim, device=device)
    ref_dof_pos[:, joint_ids[0]] = jump_hip(phi[:, 0])
    ref_dof_pos[:, joint_ids[2]] = jump_knee_final(phi[:, 0])
    ref_dof_pos[:, joint_ids[4]] = jump_ankle(phi[:, 0])
    ref_dof_pos[:, joint_ids[1]] = jump_hip(phi[:, 1])
    ref_dof_pos[:, joint_ids[3]] = jump_knee_final(phi[:, 1])
    ref_dof_pos[:, joint_ids[5]] = jump_ankle(phi[:, 1])
    
    return ref_dof_pos