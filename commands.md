# on main branch
## run AMP 
```
python ase/run.py --task HumanoidAMP --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy --headless
```

## visualization
```
python ase/run.py --test --task HumanoidViewMotion --num_envs 2 --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file /home/jing/Documents/projs/amass/npys/0-ACCAD-Female1Walking-c3d-B12-walkturnright-90--poses.npy
```

# on amass branch
## run visualization
```
python ase/run.py --test --task HumanoidViewMotion --num_envs 2 --cfg_env ase/data/cfg/humanoid_amp_smpl.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file /home/jing/Documents/projs/amass/npys/0-ACCAD-Female1Walking-c3d-B12-walkturnright-90--poses.npy
```

## run AMP training
```
python ase/run.py --task HumanoidAMP --cfg_env ase/data/cfg/humanoid_amp_smpl.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file /home/jing/Documents/projs/amass/npys/0-ACCAD-Female1Walking-c3d-B12-walkturnright-90--poses.npy --headless
```

# run AMP testing
```
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_amp_smpl.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file /home/jing/Documents/projs/amass/npys/0-ACCAD-Female1Walking-c3d-B12-walkturnright-90--poses.npy 
```