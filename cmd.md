# Leju Kuavo42 IsaacSim Notes
## Model
- Usd file: `./models/biped_s42_fine/xml/biped_s42_collision/biped_s42_noworld_mass_singlelayer_fixed_head.usd`
- Robot config file: `source/isaaclab_assets/isaaclab_assets/robots/leju.py`
- Rough env config file: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/leju/rough_env_cfg.py`
- Env Config file (e.x.: inference Hz): `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg_leju.py`
- PPO config file: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/leju/agents/rsl_rl_ppo_cfg.py`
- Register a new environment:
1. Add a new file and new `ArticulationCfg` name, for example: `source/isaaclab_assets/isaaclab_assets/leju.py`
2. Add a new Rough env config: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/leju/rough_env_cfg.py`
3. (Optional) Update Env class: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg_leju.py`
4. Add flat env config: `source/isaaclab_tasks/isaaclab_tasks\manager_based\locomotion\velocity\config\leju\flat_env_cfg.py`
5. Add your rough env name to `gym.register`: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/leju/__init__.py`

## Actions
Add action func: `source/isaaclab/isaaclab/managers/action_manager.py`

## Observations
Check joint_names: `source/isaaclab/isaaclab/envs/mdp/observations.py:121 line`

## Commands
- checkout config: `python ./scripts/benchmarks/benchmark_load_robot.py --robot leju`
- train: `python ./scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-leju-v2 --headless`
- evaluate: `python ./scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-leju-v2 --num_envs 1`
- tensorboard: `tensorboard --logdir=logs\rsl_rl\leju_flat\{timestamp}`

# Update Infos
- 2025.3.7.
    - Fix Leju Kuavo42 V1 error `feet_alternate` reward function left and right feets indexs
    - Add a punishment for `long_air_time_indices` and `long_contact_time_indices` in `feet_air_time_positive_biped` reward function
    - Add a reward class `LejuV1Rewards` for Leju Kuavo42 V1
    - Change `leg_l4_joint` and `leg_r4_joint` upper limit from 150 to 90 degrees

- 2025.3.8.
    - Decrease the punishment for `long_air_time_indices` and `long_contact_time_indices` in `feet_air_time_positive_biped` reward function
    - Change `leg_l3_joint` and `leg_r3_joint` upper limit from -60 to 60 degrees

- 2025.3.14.
    - Add leju-V2 for IsaacSim, which prohibit the move of arm joints.
    - Merge different versions of Leju cfg into the same `leju.py` file

- 2025.3.21.
    - Add ref joint pos and the reward.
    - Remove the feet alternate reward, update the feed air time reward
    - Change some PPO cfgs
    - Lightweight the repo for leju training