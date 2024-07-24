from collections import OrderedDict
import numpy as np

# path = "ase/data/dataset_multi_style/motions/style_walk_forwards/ACCAD+__+Female1Walking_c3d+__+B3_-_walk1_stageii/smpl_humanoid_v2/ref_motion.npy"

path = '/home/jing/Documents/projs/amass/npys/0-ACCAD-Female1Walking-c3d-B3-walk1-poses.npy'
d = np.load(path, allow_pickle=True).item()
print(d['rotation']['arr'].shape)
print(d['root_translation']['arr'].shape)
print(d['global_velocity']['arr'].shape)
print(d['global_angular_velocity']['arr'].shape)
print("__name__", d["__name__"])
print(d['skeleton_tree']['node_names'])
print(d['skeleton_tree']['parent_indices']['arr'])
print(d['skeleton_tree']['local_translation']['arr'].shape)
# print(d['skeleton_tree']['local_xml_rotation']['arr'].shape)
print(d['is_local'])
print(d['fps'])
for k in d:
    print(k)