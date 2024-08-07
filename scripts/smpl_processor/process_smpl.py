import smplx
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as sRot


def process_amass_data(data):
    """
    Process amass data into a format that motionlib can read
    Needed data: ['local_rotation', 'root_translation', 'fps'] which should be pose_quat and trans_orig in the dataset output
    from PHC
    Args:
        data: (dict) the AMASS data, which contains ['betas', 'dmpls', 'gender', 'mocap_framerate', 'poses', 'trans']
            betas: (np.ndarray) shape (16,)  # controls the body shape. Body shape is static
            dmpls: (np.ndarray) shape (239, 8)  # controls soft tissue dynamics
            trans: (np.ndarray) shape (239, 3)  # controls the global body position, root translation
            poses: (np.ndarray) shape (239, 156)  # controls the body pose, local rotation
                [:, :3] # controls the global root orientation
                [:, 3:66] # controls the body pose
                [:, 66:] # controls the hand pose
    Returns:
        new_data: (dict) the data in the format that motionlib can read
    """
    betas = data['betas']
    dmpls = data['dmpls'] if 'dmpls' in data else None
    gender = data['gender'].item().decode('utf-8') if 'gender' in data else None
    framerate = data['mocap_framerate']

    skip = int(framerate / 30)

    root_trans = data['trans'][::skip, :]
    pose_aa = np.concatenate([data['poses'][::skip, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    N = pose_aa.shape[0]


    # smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
    # pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
    # pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

    # beta = np.zeros((16))
    # gender_number, beta[:], gender = [0], 0, "neutral"


