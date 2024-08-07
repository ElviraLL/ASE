import sys
sys.path.append("./")

import os
import os.path as osp
import torch
import trimesh
import numpy as np
import torchgeometry as tgm
import glob

from amp.poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from amp.poselib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from amp.poselib.poselib.core.rotation3d import quat_mul, quat_from_angle_axis, quat_mul_norm, quat_rotate, quat_identity

from body_models.model_loader import get_body_model

from lpanlib.isaacgym_utils.vis.api import vis_motion_use_scenepic_animation
from lpanlib.others.colors import name_to_rgb

joints_to_use = {
    "from_smpl_original_to_smpl_humanoid": np.array([0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]),
    "from_smpl_original_to_smpl_humanoid_v2": np.array([0, 1, 4, 7, 2, 5, 8, 3, 6, 9, 12, 13, 16, 18, 20, 14, 17, 19, 21]),
}

###### SMPL Model Joints
# 0 Pelvis
# 1 L_Hip
# 2 R_Hip
# 3 Torso
# 4 L_Knee
# 5 R_Knee
# 6 Spine
# 7 L_Ankle
# 8 R_Ankle
# 9 Chest
# 10 L_Toe
# 11 R_Toe
# 12 Neck
# 13 L_Thorax
# 14 R_Thorax
# 15 Head
# 16 L_Shoulder
# 17 R_Shoulder
# 18 L_Elbow
# 19 R_Elbow
# 20 L_Wrist
# 21 R_Wrist
# 22 L_Hand
# 23 R_Hand

if __name__ == '__main__':

    all_files = glob.glob(osp.join(osp.dirname(__file__), "motions/*/*/smpl_params.npy"))

    # parameters for motion editing
    candidates = {
        "ACCAD+__+Female1General_c3d+__+A15_-_skip_to_stand_stageii": [0, 60],
        "ACCAD+__+Female1Walking_c3d+__+B20_-_walk_with_box_stageii": [35, 128],
        "SFU+__+0018+__+0018_Catwalk001_stageii": [220, 315],
        "ACCAD+__+Female1General_c3d+__+A3_-_Swing_t2_stageii": [45, 180],
        "ACCAD+__+Female1General_c3d+__+A6-_lift_box_t2_stageii": [70, 180],
    }

    # load skeleton of smpl_humanoid
    smpl_humanoid_xml_path = osp.join(osp.dirname(__file__), "../assets/mjcf/smpl_humanoid.xml")
    smpl_humanoid_skeleton = SkeletonTree.from_mjcf(smpl_humanoid_xml_path)

    # load skeleton of smpl_humanoid_v2
    smpl_humanoid_v2_xml_path = osp.join(osp.dirname(__file__), "../assets/mjcf/smpl_humanoid_v2.xml")
    smpl_humanoid_v2_skeleton = SkeletonTree.from_mjcf(smpl_humanoid_v2_xml_path)

    # load skeleton of smpl_original
    bm = get_body_model("SMPL", "NEUTRAL", batch_size=1, debug=False)
    jts_global_trans = bm().joints[0, :24, :].cpu().detach().numpy()
    jts_local_trans = np.zeros_like(jts_global_trans)
    for i in range(jts_local_trans.shape[0]):
        parent = bm.parents[i]
        if parent == -1:
            jts_local_trans[i] = jts_global_trans[i]
        else:
            jts_local_trans[i] = jts_global_trans[i] - jts_global_trans[parent]

    skel_dict = smpl_humanoid_skeleton.to_dict()
    skel_dict["node_names"] = [
        "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine", "L_Ankle", "R_Ankle",
        "Chest", "L_Toe", "R_Toe", "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder", 
        "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
    ]
    skel_dict["parent_indices"]["arr"] = bm.parents.numpy()
    skel_dict["local_translation"]["arr"] = jts_local_trans
    smpl_original_skeleton = SkeletonTree.from_dict(skel_dict)

    # create tposes
    smpl_humanoid_tpose = SkeletonState.zero_pose(smpl_humanoid_skeleton)
    smpl_original_tpose = SkeletonState.zero_pose(smpl_original_skeleton)
    smpl_humanoid_v2_tpose = SkeletonState.zero_pose(smpl_humanoid_v2_skeleton)

    body_model = get_body_model("SMPL", "NEUTRAL", 1, debug=False)

    for f in all_files:
        skill = f.split("/")[-3]
        seq_name = f.split("/")[-2]

        print("processing [skill: {}] [seq_name: {}]".format(skill, seq_name))
        
        raw_params = np.load(f, allow_pickle=True).item()

        if seq_name in list(candidates.keys()):
            f_start = candidates[seq_name][0]
            f_end = candidates[seq_name][1]
            poses = torch.tensor(raw_params["poses"][f_start:f_end], dtype=torch.float32)
            trans = torch.tensor(raw_params["trans"][f_start:f_end], dtype=torch.float32)
        else:
            poses = torch.tensor(raw_params["poses"], dtype=torch.float32)
            trans = torch.tensor(raw_params["trans"], dtype=torch.float32)
        
        fps = raw_params["fps"]

        # compute world absolute position of root joint
        trans = body_model(
            global_orient=poses[:, 0:3], 
            body_pose=poses[:, 3:72],
            transl=trans[:, :],
        ).joints[:, 0, :].cpu().detach()

        poses = poses.reshape(-1, 24, 3)

        # angle axis ---> quaternion
        poses_quat = tgm.angle_axis_to_quaternion(poses.reshape(-1, 3)).reshape(poses.shape[0], -1, 4)

        # switch quaternion order
        # wxyz -> xyzw
        poses_quat = poses_quat[:, :, [1, 2, 3, 0]]

        # generate motion
        skeleton_state = SkeletonState.from_rotation_and_root_translation(smpl_original_skeleton, poses_quat, trans, is_local=True)
        motion = SkeletonMotion.from_skeleton_state(skeleton_state, fps=fps)

        # plot_skeleton_motion_interactive(motion)

        configs = {
            "smpl_humanoid": {
                "skeleton": smpl_humanoid_skeleton,
                "xml_path": smpl_humanoid_xml_path,
                "tpose": smpl_humanoid_tpose,
                "joints_to_use": joints_to_use["from_smpl_original_to_smpl_humanoid"],
                "root_height_offset": 0.015,
            },
            "smpl_humanoid_v2": {
                "skeleton": smpl_humanoid_v2_skeleton,
                "xml_path": smpl_humanoid_v2_xml_path,
                "tpose": smpl_humanoid_v2_tpose,
                "joints_to_use": joints_to_use["from_smpl_original_to_smpl_humanoid_v2"],
                "root_height_offset": 0.08,
            },
        }

        ###### retargeting ######
        for k, v in configs.items():

            target_origin_global_rotation = v["tpose"].global_rotation.clone()
            
            # 用一个相对于静止的世界坐标系进行旋转 对齐两个初始Tpose
            target_aligned_global_rotation = quat_mul_norm( 
                torch.tensor([-0.5, -0.5, -0.5, 0.5]), target_origin_global_rotation
            )

            # viz_pose = SkeletonState.from_rotation_and_root_translation(
            #     skeleton_tree=v["skeleton"],
            #     r=target_aligned_global_rotation,
            #     t=v["tpose"].root_translation,
            #     is_local=False,
            # )
            # plot_skeleton_state(viz_pose)

            # retargeting... 太TMD简单了 居然搞了两天
            target_final_global_rotation = quat_mul_norm(
                skeleton_state.global_rotation.clone()[..., v["joints_to_use"], :], target_aligned_global_rotation.clone()
            )
            target_final_root_translation = skeleton_state.root_translation.clone()

            new_skeleton_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree=v["skeleton"],
                r=target_final_global_rotation,
                t=target_final_root_translation,
                is_local=False,
            ).local_repr()
            new_motion = SkeletonMotion.from_skeleton_state(new_skeleton_state, fps=fps)

            new_motion_params_root_trans = new_motion.root_translation.clone()
            new_motion_params_local_rots = new_motion.local_rotation.clone()

            # 对每一帧都进行防穿模处理
            num_frames = new_motion.global_translation.shape[0]
            for i in range(num_frames):
                min_h = torch.min(new_motion.global_translation[i, :, 2])
                new_motion_params_root_trans[i, 2] += -min_h
            
            # adjust the height of the root to avoid ground penetration
            root_height_offset = v["root_height_offset"]
            new_motion_params_root_trans[:, 2] += root_height_offset

            # update new_motion
            new_skeleton_state = SkeletonState.from_rotation_and_root_translation(v["skeleton"], new_motion_params_local_rots, new_motion_params_root_trans, is_local=True)
            new_motion = SkeletonMotion.from_skeleton_state(new_skeleton_state, fps=fps)

            # save retargeted motion
            save_dir = osp.join(osp.dirname(f), k)
            os.makedirs(save_dir, exist_ok=True)
            save_path = osp.join(save_dir, "ref_motion.npy")
            new_motion.to_file(save_path)

            # plot_skeleton_motion_interactive(new_motion)

            # scenepic animation
            vis_motion_use_scenepic_animation(
                asset_filename=v["xml_path"],
                rigidbody_global_pos=new_motion.global_translation,
                rigidbody_global_rot=new_motion.global_rotation,
                fps=fps,
                up_axis="z",
                color=name_to_rgb['AliceBlue'] * 255,
                output_path=osp.join(save_dir, "ref_motion_render.html"),
            )
