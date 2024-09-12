import os
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--amass_dir", type=str, required=True) # AMASS (SMPL-X N)
    args = parser.parse_args()

    # input/output dirs
    amass_dir = args.amass_dir
    output_dir = os.path.join(os.path.dirname(__file__), "motions")

    os.makedirs(output_dir, exist_ok=True)

    # selected motions
    candidates = {
        "style_walk_forwards": [
            "ACCAD+__+Female1General_c3d+__+A15_-_skip_to_stand_stageii",
            "ACCAD+__+Female1Walking_c3d+__+B3_-_walk1_stageii",
            "ACCAD+__+Female1Walking_c3d+__+B20_-_walk_with_box_stageii",
            "SFU+__+0005+__+0005_Jogging001_stageii",
            "SFU+__+0018+__+0018_Catwalk001_stageii",
        ],
        "style_stand": [
            "ACCAD+__+Female1General_c3d+__+A3_-_Swing_t2_stageii",
            "ACCAD+__+Female1General_c3d+__+A6-_lift_box_t2_stageii",
            "BMLmovi+__+Subject_1_F_MoSh+__+Subject_1_F_2_stageii",
            "BMLmovi+__+Subject_1_F_MoSh+__+Subject_1_F_3_stageii",
            "BMLmovi+__+Subject_1_F_MoSh+__+Subject_1_F_5_stageii",
        ]
    }

    for cate, seqs in candidates.items():

        curr_output_dir = os.path.join(output_dir, cate)
        os.makedirs(curr_output_dir, exist_ok=True)

        for seq in seqs:
        
            subset_name = seq.split("+__+")[0]
            subject = seq.split("+__+")[1]
            action = seq.split("+__+")[2]

            # load raw params from AMASS dataset
            fname = os.path.join(amass_dir, subset_name, subject, action + ".npz")
            raw_params = dict(np.load(fname, allow_pickle=True))

            poses = raw_params["poses"]
            trans = raw_params["trans"]

            # downsample from 120hz to 30hz
            source_fps = raw_params["mocap_frame_rate"]
            target_fps = 30
            skip = int(source_fps // target_fps)
            poses = poses[::skip]
            trans = trans[::skip]

            # extract 24 SMPL joints from 55 SMPL-X joints
            joints_to_use = np.array(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 40]
            )
            joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)
            poses = poses[:, joints_to_use]

            required_params = {}
            required_params["poses"] = poses
            required_params["trans"] = trans
            required_params["fps"] = target_fps

            save_path = os.path.join(curr_output_dir, f"{subset_name}+__+{subject}+__+{action}", "smpl_params.npy")
            print("saving {}".format(save_path))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, required_params)
