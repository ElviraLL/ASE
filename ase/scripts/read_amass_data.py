import argparse
from pathlib import Path
import joblib
from tqdm import tqdm
import os
import numpy as np

all_sequences = [
    "ACCAD",
    "BMLmovi",
    # "BioMotionLab_NTroje",
    # "CMU",
    # "DFaust_67",
    # "EKUT",
    # "Eyes_Japan_Dataset",
    # "HumanEva",
    # "KIT",
    # "MPI_HDM05",
    # "MPI_Limits",
    # "MPI_mosh",
    # "SFU",
    # "SSM_synced",
    # "TCD_handMocap",
    # "TotalCapture",
    # "Transitions_mocap",
    # "BMLhandball",
    # "DanceDB"
]

def process_data(folder, sequences, output_dir):
    # sequences = [osp.join(folder, x) for x in sorted(os.listdir(folder)) if osp.isdir(osp.join(folder, x))]

    if sequences == "all":
        sequences = all_sequences

    print(folder)
    for seq_name in sequences:
        print(f"Reading {seq_name} sequence...")
        compressed_file = os.path.join(folder, f"{seq_name}.tar.bz2")
        seq_folder = os.path.join(folder, seq_name)
        
        # extract the sequence folder
        if os.path.exists(compressed_file) and not os.path.exists(seq_folder):
            print(f"Extracting {compressed_file}...")
            os.system(f"tar -xjf {compressed_file} -C {folder}")
        # Note that the extracted folder contains subfolders for subjects, and each of them contains npz files for actions

        datas = process_single_sequence(seq_folder, seq_name, output_dir)



def process_single_sequence(folder, seq_name, output_dir):
    """
    Read a sequence of data using seq_name from AMASS for example: ACCAD or CMU, each sequence contains multiple motion clips
    Args:
        folder: (str) the folder of that sequence
        seq_name: (str) name of the sequence
    Returns:
        datas
    """
    # list all files in the input folder, should be a license.txt file and the motion folder
    subjects = os.listdir(folder) 


    for subject in tqdm(subjects):
        if not os.path.isdir(os.path.join(folder, subject)): # skip the license file
            continue
        actions = [
            x for x in os.listdir(os.path.join(folder, subject)) if x.endswith(".npz") 
        ] # list all the motion files end with npz

        for action in actions:
            fname = os.path.join(folder, subject, action)
            
            # skip the npz files that end with shape.npz, maybe it is only the shape info
            if fname.endswith("shape.npz"):
                print(f"Skipping the shape file {fname}")
                continue

            data = dict(np.load(fname))

            vid_name = f"{seq_name}_{subject}_{action[:-4]}"
            
            # convert the data to MotionLib format
            motionlib_data = process_single_motion_clip(data)

            # save the data to the output folder
            out_file = os.path.join(output_dir, f"{vid_name}.npz")




def process_single_motion_clip(data):
    """
    Convert the AMASS data to MotionLib format
    Args:
        data: (dict) the AMASS data that contains ['trans', 'gender', 'mocap_frames', 'betas', 'dmpls', 'poses']
    Returns:
        motionlib_data: (dict) the data in MotionLib format
    """
    motionlib_data = {}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="raw data directory, should be a folder contains all unzipped AMASS data")
    parser.add_argument("--output_dir", type=str, help="output directory to save the processed data")
    parser.add_argument("--sequences", type=str, nargs="+", help='which AMASS sequences to use', default='all')

    args = parser.parse_args()
    out_path = Path(args.output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    process_data(folder=args.data_dir, sequences=args.sequences, output_dir=args.output_dir)
    
