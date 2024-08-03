import tqdm
import os
import os.path as osp
import numpy as np

all_sequences = [
    "ACCAD",
    "BMLmovi",
    "BioMotionLab_NTroje",
    "CMU",
    "DFaust_67",
    "EKUT",
    "Eyes_Japan_Dataset",
    "HumanEva",
    "KIT",
    "MPI_HDM05",
    "MPI_Limits",
    "MPI_mosh",
    "SFU",
    "SSM_synced",
    "TCD_handMocap",
    "TotalCapture",
    "Transitions_mocap",
    "BMLhandball",
    "DanceDB"
]

def read_data(folder, sequences):
    # sequences = [osp.join(folder, x) for x in sorted(os.listdir(folder)) if osp.isdir(osp.join(folder, x))]

    if sequences == "all":
        sequences = all_sequences

    db = {}
    print(folder)
    for seq_name in sequences:
        print(f"Reading {seq_name} sequence...")
        seq_folder = osp.join(folder, seq_name)

        datas = read_single_sequence(seq_folder, seq_name)
        db.update(datas)
        print(seq_name, "number of seqs", len(datas))

    return db


def read_single_sequence(folder, seq_name):
    """
    Read a sequence of data using seq_name from AMASS for example: ACCAD or CMU, each sequence contains multiple motion clips
    Args:
        folder: (str) the folder of that sequence
        seq_name: (str) name of the sequence
    Returns:
        datas
    """
    subjects = os.listdir(folder) # list folder, this should returns all the sub-folders

    datas = {}

    for subject in tqdm(subjects):
        if not osp.isdir(osp.join(folder, subject)):
            continue
        actions = [
            x for x in os.listdir(osp.join(folder, subject)) if x.endswith(".npz") 
        ] # list all the motion files end with npz

        for action in actions:
            fname = osp.join(folder, subject, action)

            if fname.endswith("shape.npz"):
                continue

            data = dict(np.load(fname))
            # data['poses'] = pose = data['poses'][:, joints_to_use]

            # shape = np.repeat(data['betas'][:10][np.newaxis], pose.shape[0], axis=0)
            # theta = np.concatenate([pose,shape], axis=1)
            vid_name = f"{seq_name}_{subject}_{action[:-4]}"

            datas[vid_name] = data
            # thetas.append(theta)

    return datas
