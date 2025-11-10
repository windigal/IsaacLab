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
- train:
- - `python ./scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-leju-v2 --headless`
- - `python ./scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-leju-wbc --headless`
- - `python ./scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-wbc --headless`
- - `python ./scripts/reinforcement_learning/bmpc/train.py --task Isaac-Velocity-Flat-leju-wbc --headless --num_envs 1`
- evaluate
- - `python ./scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-leju-v2 --num_envs 1 --device=cpu`
- - `python ./scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-leju-wbc --num_envs 1 --device=cpu`
- - `python ./scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-G1-wbc --num_envs 1 --device=cpu`
- evaluate with joystick control
- - `python ./scripts/reinforcement_learning/rsl_rl/play_joystick.py --task Isaac-Velocity-Flat-leju-v2 --num_envs 1 --device=cpu`
- - `python ./scripts/reinforcement_learning/rsl_rl/play_joystick.py --task Isaac-Velocity-Flat-leju-wbc --num_envs 1 --device=cpu`
- - `python ./scripts/reinforcement_learning/rsl_rl/play_joystick.py --task Isaac-Velocity-Flat-G1-wbc --num_envs 1 --device=cpu`
- tensorboard: `tensorboard --logdir=logs\{experiment}`


- amp:
- - `python ./scripts/reinforcement_learning/skrl/train.py --task Isaac-Leju-AMP-Walk-Direct-v0 --headless --algorithm AMP`
- - `python ./scripts/reinforcement_learning/skrl/play.py --task Isaac-Leju-AMP-Walk-Direct-v0 --num_envs 1 --algorithm AMP`
- - `python ./scripts/reinforcement_learning/skrl/train.py --task Isaac-Velocity-Flat-leju-amp --headless --algorithm AMP`
- - `python ./scripts/reinforcement_learning/skrl/play.py --task Isaac-Velocity-Flat-leju-amp --algorithm AMP --num_envs 1`

- using `isaaclab.bat`
- `isaaclab.bat -p ./scripts/reinforcement_learning/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --headless`
- `isaaclab.bat -p ./scripts/reinforcement_learning/rsl_rl/train.py --task Unitree-Go2-Velocity --headless`
- `isaaclab.bat -p ./scripts/reinforcement_learning/rsl_rl/play_joystick.py --task Unitree-G1-29dof-Velocity --num_envs 1`
- `isaaclab.bat -p ./scripts/reinforcement_learning/rsl_rl/play.py --task Unitree-Go2-Velocity --num_envs 128`
- `cd _isaac_sim && python.bat -m tensorboard.main --logdir`
- 