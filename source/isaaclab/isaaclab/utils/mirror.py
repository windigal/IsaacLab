import torch

def _joint_symmetry(joint_attribute: torch.Tensor) -> torch.Tensor:
    assert joint_attribute.shape[1] % 12 == 0, "joint_attribute should have shape (N, 12 * k)"
    mirror_th = torch.zeros_like(joint_attribute)
    k = joint_attribute.shape[1] // 12
    for i in range(k):
        for j in range(6):
            mirror_th[:, 12 * i + 2 * j] = joint_attribute[:, 12 * i + 2 * j + 1]
            mirror_th[:, 12 * i + 2 * j + 1] = joint_attribute[:, 12 * i + 2 * j]
    return mirror_th


def mirror_func(obs, actions, env, obs_type):
    if obs is not None:
        if obs_type == "policy":
            obs = policy_mirror_func(obs)
        elif obs_type == "critic":
            obs = critic_mirror_func(obs)
    if actions is not None:
        actions = actions_mirror_func(actions)
    return obs, actions

def policy_mirror_func(obs):
    mirror_obs = obs.clone()
    mirror_obs[:, 45:105] = _joint_symmetry(obs[:, 45:105])
    mirror_obs[:, 105:165] = _joint_symmetry(obs[:, 105:165])
    mirror_obs[:, 165:225] = _joint_symmetry(obs[:, 165:225])
    mirror_obs[:, [250, 252, 254, 256, 258]] = obs[:, [251, 253, 255, 257, 259]]
    mirror_obs[:, [251, 253, 255, 257, 259]] = obs[:, [250, 252, 254, 256, 258]]
    return obs

def critic_mirror_func(obs):
    mirror_obs = obs.clone()
    mirror_obs[:, 9:21] = _joint_symmetry(obs[:, 9:21])
    mirror_obs[:, 21:33] = _joint_symmetry(obs[:, 21:33])
    mirror_obs[:, 33:45] = _joint_symmetry(obs[:, 33:45])
    mirror_obs[:, 50] = obs[:, 51]
    mirror_obs[:, 51] = obs[:, 50]
    mirror_obs[:, 52:64] = _joint_symmetry(obs[:, 52:64])
    return obs

def actions_mirror_func(actions):
    return _joint_symmetry(actions)