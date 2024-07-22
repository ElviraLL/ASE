"""
This script reads a pkl motion file that wrapped the AMASS data using HPC https://github.com/ZhengyiLuo/PHC, and rewrite it into a .npy file
that fits this repo's data structure.
Auther: Jingwen Liang 
Time: 06/19/2024
"""

import os
import joblib
import numpy as np
import rootutils
import torch
from collections import OrderedDict
from tqdm import tqdm
from pathlib import Path

ROOT_PATH = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from ase.poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonTree
from ase.poselib.poselib.core.tensor_utils import tensor_to_dict, TensorUtils

def rewrite_pkl(pkl_path, root_dir):
    """
    Read the pkl file and rewrite each one into a .pkl file
    """
    data = joblib.load(pkl_path)
    for key in data.keys():
        print(key)
        clip = data[key]
        joblib.dump(clip, f"{root_dir}/{key}.pkl")
        


def create_SekeltonMotion_from_dict(motion_dict: OrderedDict, skeleton_tree: SkeletonTree, *args, **kwargs):
        rot = TensorUtils.from_dict(motion_dict["rotation"], *args, **kwargs)
        rt = TensorUtils.from_dict(motion_dict["root_translation"], *args, **kwargs)
        vel = SkeletonMotion._compute_velocity(rot, 1 / motion_dict["fps"])
        avel = SkeletonMotion._compute_angular_velocity(rot, 1 / motion_dict["fps"])
        return SkeletonMotion(
            SkeletonMotion._to_state_vector(rot, rt, vel, avel),
            skeleton_tree=skeleton_tree,
            is_local=True,
            fps=motion_dict["fps"],
        )


def pkl_to_npy(pkl_path, root_dir):
    """
    Read the pkl file and rewrite each one into a .npy file
    """
    clip = joblib.load(pkl_path)
    skeleton_tree = SkeletonTree.from_mjcf("./ase/data/assets/mjcf/smpl_humanoid_1.xml")

    # for key, value in clip.items():
    #     if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
    #         print(f" {key}: , shape: {value.shape}")
    #     else:
    #         print(f" {key}: {value}")

    motion_dict = OrderedDict(
        [
            ("rotation", tensor_to_dict(torch.from_numpy(clip["pose_quat"]))),
            ("root_translation", tensor_to_dict(torch.from_numpy(clip["trans_orig"]))),
            ("fps", clip["fps"]),
        ]
    )
    motion = create_SekeltonMotion_from_dict(motion_dict, skeleton_tree)
    # remove extra space in the file name
    file_name = pkl_path.split("/")[-1].replace(" ", "").replace("(", "-").replace(")","-").replace("_", "-")
    motion.to_file(f"{root_dir}/{file_name.replace('.pkl', '.npy')}")
        


if __name__ == "__main__":

    # step 1: read the over_all big pkl and rewrite them into smaller pkl for each clip
    pkl_path = Path("/home/jing/Documents/projs/amass/all_data/amass_isaac_im_train_take6_upright_slim.pkl")
    pkl_folder = Path("/home/jing/Documents/projs/amass/pkls")
    # pkl_folder.mkdir(exist_ok=True)

    # rewrite_pkl(pkl_path, pkl_folder)

    # step 2: rewrite each pkl into npy

    npy_folder = Path("/home/jing/Documents/projs/amass/npys")
    for pkl_path in tqdm(os.listdir(pkl_folder), total=len(os.listdir(pkl_folder))):
        if "ACCAD" in pkl_path and "Female1Walking" in pkl_path:
            pkl_to_npy(f"{pkl_folder}/{pkl_path}", npy_folder)
            print(f"Finish converting {pkl_path} to .npy file.")
    
    
    