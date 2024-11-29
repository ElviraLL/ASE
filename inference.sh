python ase/run.py --test --task HumanoidAMPCarry --num_envs 16 \
--cfg_env ase/data/cfg/env/humanoid_amp_carry.yaml \
--cfg_train ase/data/cfg/train/rlg/amp_carry.yaml \
   --motion_file ase/data/cfg/data/pick.yaml \
   --checkpoint ./output/Humanoid_20241012-00-44-15/nn/Humanoid.pth