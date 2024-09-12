
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'ase'))


from isaacgym.torch_utils import *
from ase.utils.motion_lib import MotionLib

body_names = [
                'Pelvis',
                'L_Hip', 'L_Knee', 'L_Ankle',
                'R_Hip', 'R_Knee', 'R_Ankle',
                'Torso', 'Spine', 'Chest', 'Neck',
                'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
                'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist',
            ]
            
dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17] # len=16
dof_offsets = np.linspace(0, len(dof_body_ids) * 3, len(dof_body_ids) + 1).astype(int)
skeleton_ids = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22]
key_body_ids = [6, 3, 18, 4]
motion_file = 'ase/data/dataset_multi_style/cfgs/compare.yaml'


motion_lib = MotionLib(motion_file=motion_file,
    skill='loco',
    dof_body_ids=dof_body_ids,
    dof_offsets=dof_offsets,
    key_body_ids=key_body_ids,
    device='cpu',
    skeleton_ids=skeleton_ids
)
# print(motion_lib.motion_names)
length_start = motion_lib.length_starts
global_velocity = motion_lib.gvs
global_angular_velocity = motion_lib.gavs

# plot global velocity
num_motions = len(length_start)
gvs = []
for i in range(num_motions - 1):
    gvs.append(global_velocity[length_start[i]:length_start[i+1]])
gvs.append(global_velocity[length_start[-1]:])
print("")
# velocity is 229 x 19 x 3, 229 is length of motion, 19 is number of body parts, 3 is xyz, plot gvs for each bone and each motion

velocities_1 = gvs[0]
velocities_2 = gvs[1]

# Create a 3D plot for each bone's velocity
fig = plt.figure(figsize=(20, 15))
for bone_idx in range(19):
    # Add a subplot for each bone in a 5x4 grid
    ax = fig.add_subplot(5, 4, bone_idx + 1, projection='3d')
    
    # Plot the velocity vectors for Motion 1
    ax.plot(velocities_1[:, bone_idx, 0], velocities_1[:, bone_idx, 1], velocities_1[:, bone_idx, 2], label='Motion 1')
    
    # Plot the velocity vectors for Motion 2
    ax.plot(velocities_2[:, bone_idx, 0], velocities_2[:, bone_idx, 1], velocities_2[:, bone_idx, 2], label='Motion 2', linestyle='dashed')
    
    # Set title and labels
    ax.set_title(f'Bone {bone_idx + 1}')
    ax.set_xlabel('Vx')
    ax.set_ylabel('Vy')
    ax.set_zlabel('Vz')
    
    # Add a legend
    ax.legend()

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()