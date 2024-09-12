from collections import OrderedDict
import numpy as np

# path = "ase/data/dataset_multi_style/motions/style_walk_forwards/ACCAD+__+Female1Walking_c3d+__+B3_-_walk1_stageii/smpl_humanoid_v2/ref_motion.npy"

# path = '/home/jing/Documents/projs/amass/npys/0-ACCAD-Female1Walking-c3d-B3-walk1-poses.npy'
# path = '/home/jing/Documents/projs/interaction_AMP-main/interactive_amp/data/motions/reference_motion.npy'
path = '/home/jing/Documents/projs/interaction_AMP-main/interactive_amp/data/motions/chair/chair_mo_sit2sit_stageII.npy'
d = np.load(path, allow_pickle=True).item()
for k in d:
    print(k)
print("")
print(d['rotation']['arr'].shape)
print(d['root_translation']['arr'].shape)
print(d['global_velocity']['arr'].shape)
print(d['global_angular_velocity']['arr'].shape)
print("__name__", d["__name__"])
print(d['is_local'])
print(d['fps'])
print("")
print(d['skeleton_tree']['node_names'], len(d['skeleton_tree']['node_names']))
print(d['skeleton_tree']['parent_indices']['arr'])
print(d['skeleton_tree']['local_translation']['arr'].shape)
# print(d['skeleton_tree']['local_xml_rotation']['arr'].shape)

for k in d:
    print(k)