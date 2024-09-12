# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import yaml
import json
import joblib
from collections import OrderedDict

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
from poselib.poselib.core.rotation3d import *
from poselib.poselib.core.backend.abstract import json_numpy_obj_hook
from isaacgym.torch_utils import *

from utils import torch_utils

import torch

USE_CACHE = True
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy

class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)  
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
        
        print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out


class MotionLib():
    def __init__(self, motion_file, dof_body_ids, dof_offsets,
                 key_body_ids, device, skill=None, skeleton_ids=None):
        self.skeleton_ids = skeleton_ids
        self._dof_body_ids = dof_body_ids
        self._dof_offsets = dof_offsets
        self._num_dof = dof_offsets[-1]
        self._key_body_ids = torch.tensor(key_body_ids, device=device)
        self._device = device
        self._load_motions(motion_file, skill)

        motions = self._motions
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float()
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float()
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float()
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float()
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float()
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float()

        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float()
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float()

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

        return

    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._motion_weights, num_samples=n, replacement=True)
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def sample_time_rsi(self, motion_ids, truncate_time=None):
        # this function is designed for reference state init
        # it gives motion times sampled in allowed range
        n = len(motion_ids)
        succ_sample_times = torch.zeros(n, dtype=torch.float32, device=self._device)
        succ_sample_record = torch.zeros(n, dtype=torch.bool, device=self._device)
        while torch.sum(succ_sample_record) < n:
            # print(torch.sum(succ_sample_record))
            mask = (succ_sample_record == False)

            phase = torch.rand(motion_ids[mask].shape, device=self._device)
            
            motion_len = self._motion_lengths[motion_ids[mask]]
            if (truncate_time is not None):
                assert (truncate_time >= 0.0)
                motion_len -= truncate_time

            motion_time = phase * motion_len

            # check if it is within the skipped range
            skipped_range = self._motion_rsi_skipped_ranges[motion_ids[mask]]

            curr_num_samples = motion_time.shape[0]
            curr_succ = torch.where(
                torch.logical_and(motion_time >= skipped_range[:, 0], motion_time <= skipped_range[:, 1]),
                torch.zeros(curr_num_samples, dtype=torch.bool, device=self._device), # input, 
                torch.ones(curr_num_samples, dtype=torch.bool, device=self._device), # other
            )
            curr_succ_inds = torch.nonzero(mask)[curr_succ, 0]
            succ_sample_times[curr_succ_inds] = motion_time[curr_succ]
            succ_sample_record[curr_succ_inds] = True

        return succ_sample_times

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_motion_state(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        root_vel = self.grvs[f0l]

        root_ang_vel = self.gravs[f0l]
        
        key_pos0 = self.gts[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_pos1 = self.gts[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]

        dof_vel = self.dvs[f0l]

        vals = [root_pos0, root_pos1, local_rot0, local_rot1, root_vel, root_ang_vel, key_pos0, key_pos1]
        for v in vals:
            assert v.dtype != torch.float64


        blend = blend.unsqueeze(-1)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

        root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)

        # linear interpolation in maximal coordinate may not give the right values, see https://github.com/nv-tlabs/ASE/pull/53
        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        
        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof(local_rot)

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos
    
    def get_motion_state_max(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        body_pos0 = self.gts[f0l]
        body_pos1 = self.gts[f1l]
        body_rot0 = self.grs[f0l]
        body_rot1 = self.grs[f1l]
        body_vel = self.gvs[f0l]
        body_ang_vel = self.gavs[f0l]

        vals = [body_pos0, body_pos1, body_rot0, body_rot1, body_vel, body_ang_vel]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)

        # linear interpolation in maximal coordinate may not give the right values, see https://github.com/nv-tlabs/ASE/pull/53
        body_pos = (1.0 - blend_exp) * body_pos0 + blend_exp * body_pos1
        body_rot = torch_utils.slerp(body_rot0, body_rot1, blend_exp)

        return body_pos, body_rot, body_vel, body_ang_vel

    def _load_motions(self, motion_file, skill=None):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motion_rsi_skipped_ranges = []

        total_len = 0.0

        motion_files, motion_weights, motion_rsi_skipped_ranges = self._fetch_motion_files(motion_file, skill)
        
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            if curr_file.endswith(".pkl"):
                curr_motion = self._load_motion_from_pkl_file(curr_file)
            else:
                curr_motion = SkeletonMotion.from_file(curr_file, skeleton_ids=self.skeleton_ids)

            # # increase 2 on z axis for global translation
            # curr_motion.global_translation[:, :, 2] += 2.0 # shape (n, 19, 3)
            # # curr_motion.global_translation[:, :, 2] = -curr_motion.global_translation[:, 0, 2]
            # # global_root_rotation [N, 4], rotate 90 degrees around x axis
            # rot_angle = torch.tensor([np.pi/2])
            # rot_theta = torch.zeros(3)
            # rot_theta[0] = 1.0
            # n_frames = curr_motion.global_root_rotation.shape[0]
            # curr_motion.global_root_rotation[:] = quat_mul(quat_from_angle_axis(rot_angle, rot_theta).repeat(n_frames, 1), curr_motion.global_root_rotation)[:]
            # # assign that value to global rotations and local rotations
            # curr_motion.global_rotation[:, 0, ...] = curr_motion.global_root_rotation
            # curr_motion.local_rotation[:, 0, ...] =curr_motion.global_root_rotation
            # curr_motion.rotation[:, 0, ...] = curr_motion.global_root_rotation


            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)

            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            # Moving motion tensors to the GPU
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)                
            else:
                curr_motion.tensor = curr_motion.tensor.to(self._device)
                curr_motion._skeleton_tree._parent_indices = curr_motion._skeleton_tree._parent_indices.to(self._device)
                curr_motion._skeleton_tree._local_translation = curr_motion._skeleton_tree._local_translation.to(self._device)
                curr_motion._rotation = curr_motion._rotation.to(self._device)

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            
            curr_weight = motion_weights[f]
            self._motion_weights.append(curr_weight)
            self._motion_files.append(curr_file)

            curr_rsi_skipped_ranges = [frame_id / num_frames * curr_len for frame_id in motion_rsi_skipped_ranges[f]] # times
            self._motion_rsi_skipped_ranges.append(curr_rsi_skipped_ranges)

        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)

        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()

        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)

        self._motion_rsi_skipped_ranges = torch.tensor(self._motion_rsi_skipped_ranges, device=self._device, dtype=torch.float32)

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        # one-hot codes
        self._motion_labels = torch.eye(num_motions, device=self._device)

        if skill is not None:
            print("[Skill: {:s}] Loaded {:d} motions with a total length of {:.3f}s.".format(skill, num_motions, total_len))
        else:
            print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return
    
    def _load_motion_from_pkl_file(self, curr_file):
        """
        Load motion from file
        Args:
            curr_file: (str) the file path
        Returns:
            curr_motion: (SkeletonMotion) the motion object
        """
        # TODO: rewrite this to only load pkls
        if curr_file.endswith(".json"):
            with open(curr_file, "r") as f:
                d = json.load(f, object_hook=json_numpy_obj_hook)
        elif curr_file.endswith(".npy"):
            d = np.load(curr_file, allow_pickle=True).item()
        elif curr_file.endswith(".pkl"):
            d = joblib.load(curr_file)
        else:
            assert False, "failed to load {} from {}".format(SkeletonMotion.__name__, curr_file)
        # assert d["__name__"] == SkeletonMotion.__name__, "the file belongs to {}, not {}".format(
        #     d["__name__"], SkeletonMotion.__name__
        # )
        skeleton_tree = SkeletonTree.from_mjcf("./ase/data/assets/mjcf/smpl_humanoid_v2.xml") # the original 24 bone skeletontree TODO: don't use hard coded values
        # skeleton_tree = self._update_skeleton_tree(d['skeleton_tree'], self.skeleton_ids)
        self._update_motion_data_with_new_skeleton(d, self.skeleton_ids, skeleton_tree)
        trans = d['root_trans_offset']
        pose_quat_global = torch.from_numpy(d['pose_quat_global'])
        sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, pose_quat_global, trans, is_local=False)
        curr_motion = SkeletonMotion.from_skeleton_state(sk_state, d.get("fps", 30))
        return curr_motion
    
    def _update_skeleton_tree(self, skeleton_tree, selected_ids):
        if selected_ids is not None and len(selected_ids) < len(skeleton_tree['node_names']):
            original_node_names = skeleton_tree["node_names"]
            original_parent_indices = skeleton_tree["parent_indices"]["arr"]

            old_to_new_index = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_ids)}

            new_node_names = [original_node_names[idx] for idx in selected_ids]
            new_parent_indices = []
            def find_valid_parent(idx):
                original_parent_idx = original_parent_indices[idx]
                # Check if the parent is in the selected skeleton_ids
                while original_parent_idx not in selected_ids and original_parent_idx != -1:
                    original_parent_idx = original_parent_indices[original_parent_idx]
                # Return the new index if valid parent found, else -1
                return old_to_new_index.get(original_parent_idx, -1)
            
            for idx in selected_ids:
                new_parent_idx = find_valid_parent(idx)
                new_parent_indices.append(new_parent_idx)

            updated_skelton_tree = OrderedDict()
            updated_skelton_tree["node_names"] = new_node_names
            updated_skelton_tree["parent_indices"] = {"arr": np.array(new_parent_indices), 'context': {'dtype': 'int64'}}
            updated_skelton_tree['local_translation'] = {'arr': skeleton_tree['local_translation']['arr'][selected_ids], 'context': {'dtype': 'float32'}}
        return updated_skelton_tree


    def _update_motion_data_with_new_skeleton(self, d, selected_ids, skeleton_tree):
        """
        Update the motion series, with new skeleton
        Technically, we need to recalculate the new positions by retunning FK, and update the local rotations
        we can simply select the corresponding motion data with their selected skeleton ids here
        """
        # TODO: Implement the FK if needed
        if selected_ids is not None and len(selected_ids) < len(skeleton_tree.node_names):
            d['pose_quat_global'] = d['pose_quat_global'][:, selected_ids]
            d['pose_aa'] = d['pose_aa'][:, selected_ids]


    def _fetch_motion_files(self, motion_file, skill=None):
        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            motion_weights = []
            motion_rsi_skipped_ranges = []

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config['motions'][skill]
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                assert(curr_weight >= 0)

                curr_file = os.path.join(dir_name, curr_file)
                motion_weights.append(curr_weight)
                motion_files.append(curr_file)

                curr_rsi_skipped_range = motion_entry.get('rsi_skipped_range', []) # new added
                if len(curr_rsi_skipped_range) == 0:
                    curr_rsi_skipped_range = [np.inf, -np.inf]
                else:
                    assert len(curr_rsi_skipped_range) == 2
                    assert curr_rsi_skipped_range[0] < curr_rsi_skipped_range[1]

                motion_rsi_skipped_ranges.append(curr_rsi_skipped_range)

        else:
            motion_files = [motion_file]
            motion_weights = [1.0]
            motion_rsi_skipped_ranges = [[np.inf, -np.inf]]

        return motion_files, motion_weights, motion_rsi_skipped_ranges

    def _calc_frame_blend(self, time, len, num_frames, dt):

        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _compute_motion_dof_vels(self, motion):
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            frame_dof_vel = frame_dof_vel
            dof_vels.append(frame_dof_vel)
        
        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)

        return dof_vels
    
    def _local_rotation_to_dof(self, local_rot):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_q = local_rot[:, body_id]
                joint_exp_map = torch_utils.quat_to_exp_map(joint_q)
                dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map
            elif (joint_size == 1):
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                joint_theta = joint_theta * joint_axis[..., 1] # assume joint is always along y axis

                joint_theta = normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        dof_vel = torch.zeros([self._num_dof], device=self._device)

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1) # inverse rot0 * rot1
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel

            elif (joint_size == 1):
                assert(joint_size == 1)
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[1] # assume joint is always along y axis

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_vel

    def get_motion_label(self, motion_ids):
        return self._motion_labels[motion_ids]

    def sample_motion_labels(self, n):
        motion_ids = self.sample_motions(n)
        return self._motion_labels[motion_ids]
