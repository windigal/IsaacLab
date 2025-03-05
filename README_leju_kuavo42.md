# Leju Kuavo42 IsaacSim Notes
## Model
Usd file: `E:/isaacsim/models/biped_s42_fine/xml/biped_s42_collision/biped_s42_noworld_mass_singlelayer_fixed_head.usd`
Robot config file: `source/extensions/omni.isaac.lab_assets/omni/isaac/lab_assets/leju.py`
Rough env config file: `source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/config/leju/rough_env_cfg.py`
Env Config file (e.x.: inference Hz): `source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/velocity_env_cfg_leju.py`
PPO config file: `source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/config/leju/agents/rsl_rl_ppo_cfg.py`

## Actions
Add action func: `source/extensions/omni.isaac.lab/omni/isaac/lab/managers/action_manager.py`

## Observations
Check joint_names: `source/extensions/omni.isaac.lab/omni/isaac/lab/envs/mdp/observations.py:121 line`

## Commands
- checkout config: `python e:/IsaacLab/source/standalone/benchmarks/benchmark_load_robot.py --robot leju`
- train: `python ./source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Flat-leju-v0 --headless`
- evaluate: `python ./source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-Flat-leju-v0 --num_envs 1`
- tensorboard: `tensorboard --logdir=logs\rsl_rl\leju_flat\{timestamp}`