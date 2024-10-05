#!/bin/bash
# python ase/run.py --task HumanoidAMP --cfg_env ase/data/cfg/env/humanoid_amp_smpl.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid_im.yaml --motion_file ase/data/dataset_multi_style/cfgs/ours.yaml 
python ase/run.py \
--task HumanoidAMPCarry \
--cfg_env ase/data/cfg/env/humanoid_amp_carry.yaml \
--cfg_train ase/data/cfg/train/rlg/amp_carry.yaml \
--motion_file ase/data/cfg/data/carry.yaml \
--num_envs 4006 \
--resume 1 \
--checkpoint ./output/Humanoid_01-22-44-30/nn/Humanoid.pth \
# --headless