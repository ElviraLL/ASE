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

import torch
import random
import yaml
import os
from copy import deepcopy


from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *


from env.tasks.humanoid_amp_multi_agent import HumanoidAMPMultiAgent


class HumanoidViewMultiMotion(HumanoidAMPMultiAgent):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        #print(f"ase.env.tasks.humanoid_view_motion.HumanoidViewMotion: initializing motion viewer, headless={headless}")
        control_freq_inv = cfg["env"]["controlFrequencyInv"]
        self._motion_dt = control_freq_inv * sim_params.dt

        cfg["env"]["controlFrequencyInv"] = 1
        cfg["env"]["pdControl"] = False

        motion_file = cfg['env']['motion_file']
        ext = motion_file.split('.')[-1]
        if ext == 'yaml':
            self.load_motion_cfg(motion_file)
            cfg["env"]["num_agents"] = self.num_motions
            # override cfg["env"]["numEnvs"] = self.num_motions

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        num_motions = self._motion_lib.num_motions()
        # self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        # self._motion_ids = torch.remainder(self._motion_ids, num_motions)
        self._motion_ids_list = []
        for i in range(self.num_agents):
            self._motion_ids = torch.tensor([i] * self.num_envs, device=self.device, dtype=torch.long)
            self._motion_ids_list.append(self._motion_ids)
        return
    
    def load_motion_cfg(self, motion_file):
        """
        Load motion configuration file, know how many motion clips to visualize, load colors, for each motion
        """

        with open(motion_file, 'r') as f:
            self.motion_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        num_motions = 0
        self.character_colors = []
        for skill in self.motion_cfg['motions']:
            num_motions += len(self.motion_cfg['motions'][skill])
            motion_list = self.motion_cfg['motions'][skill]
            for motion_entry in motion_list:
                character_color = motion_entry.get('character_color', None)
                if character_color is None:
                    # generate a random color
                    character_color = gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))  
                else:
                    character_color = gymapi.Vec3(character_color[0], character_color[1], character_color[2])
                self.character_colors.append(character_color)
        self.num_motions = num_motions  
        return
    

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        #asset_options.fix_base_link = True
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, self._body_right_foot_name) # left_Ankle
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, self._body_left_foot_name) # right_Ankle
        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        # TODO: Jingwen, use PHD's function to create humanoid force sensors, joint is copied from PHD too
        # self.create_humanoid_force_sensors(humanoid_asset, ["L_Ankle", "R_Ankle"]) 


        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, humanoid_asset, self.character_colors, self.num_motions)
            self.envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        #print(f"ase.env.tasks.humanoid: dof_prop: {dof_prop}")
        # log the dof limits
        # ('hasLimits', 'lower', 'upper', 'driveMode', 'velocity', 'effort', 'stiffness', 'damping', 'friction', 'armature')

        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        # if self.control_mode == "pd":
        #     self.torque_limits = torch.ones_like(self.dof_limits_upper) * 1000 # ZL: hacking 

        if self._pd_control:
            self._build_pd_action_offset_scale()
        return


    def _build_env(self, env_id, env_ptr, humanoid_asset, character_colors, num_motions):
        # TODO: Jingwen, use PHD's function to build humanoid, they have humanoid_masses and humanoid_limb_and_weights
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        char_h = 0.89

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # generate a list of start_pose for different posisiotns, based on num_motions, the center one should be centered at default start_pose.p
        # e.g. if num_motions = 1, do nothing
        # if num_motions = 2, mid = 2//2 = 1, loop from (0, 1), only 1 loop, update start_pose[0], and start_poses[2 - 0 - 1] = start_poses[1], 1 is the center skip
        # if num_motions = 3, mid = 3//2 = 1, loop from (0, 1), 1 loop, update start_pose[0] and start_pose[3 - 0  -1]
        start_poses = [start_pose] * num_motions
        offset = 0.5
        for idx in range(0, num_motions // 2):
            start_poses[idx] = deepcopy(start_pose)
            j = num_motions // 2 - idx # number of character away from the center character
            start_poses[idx].p.x -= j * offset
            if num_motions - 1 - idx > num_motions // 2:
                if num_motions % 2 == 0:
                    start_poses[num_motions - 1 - idx] = deepcopy(start_pose)
                    start_poses[num_motions - 1 - idx].p.x += (j - 1) * offset
                else:
                    start_poses[num_motions - 1 - idx] = deepcopy(start_pose)
                    start_poses[num_motions - 1 - idx].p.x += j * offset

        for i in range(num_motions):
            character_color = character_colors[i]
            humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_poses[i], "humanoid", col_group, col_filter, segmentation_id)
            self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, character_color)

            if (self._pd_control):
                dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
                dof_prop["driveMode"] = gymapi.DOF_MODE_POS
                self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)
            self.humanoid_handles.append(humanoid_handle)
        return

        


    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        forces = torch.zeros_like(self.actions)
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        return

    def post_physics_step(self):
        super().post_physics_step()
        self._motion_sync()
        return
    
    def _get_humanoid_collision_filter(self):
        return 1 # disable self collisions

    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = self._motion_ids_list
        motion_times = self.progress_buf * self._motion_dt
        
        root_poses, root_rots, dof_poses, root_vels, root_ang_vels, dof_vels, key_poses = [], [], [], [], [], [], []
        for i in range(self.num_agents):
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = self._motion_lib.get_motion_state(self._motion_ids_list[i], motion_times)  
            root_poses.append(root_pos)
            root_rots.append(root_rot)
            dof_poses.append(dof_pos)
            # root_vels.append(root_vel)
            # root_ang_vels.append(root_ang_vel)
            # dof_vels.append(dof_vel)
            key_poses.append(key_pos) 
        
            root_vels.append(torch.zeros_like(root_vel))
            root_ang_vels.append(torch.zeros_like(root_ang_vel))
            dof_vels.append(torch.zeros_like(dof_vel))


        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_poses, 
                            root_rot=root_rots, 
                            dof_pos=dof_poses, 
                            root_vel=root_vels, 
                            root_ang_vel=root_ang_vels, 
                            dof_vel=dof_vels)


        # env_ids_int32 = self._humanoid_actor_ids[env_ids]
        env_ids_int32 = env_ids.int()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _compute_reset(self):
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        self.reset_buf[:], self._terminate_buf[:] = compute_view_motion_reset(self.reset_buf, motion_lengths, self.progress_buf, self._motion_dt)
        return

    def _reset_actors(self, env_ids):
        return

    def _reset_env_tensors(self, env_ids):
        num_motions = self._motion_lib.num_motions()
        self._motion_ids[env_ids] = torch.remainder(self._motion_ids[env_ids] + self.num_envs, num_motions)
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

@torch.jit.script
def compute_view_motion_reset(reset_buf, motion_lengths, progress_buf, dt):
    # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    motion_times = progress_buf * dt
    reset = torch.where(motion_times > motion_lengths, torch.ones_like(reset_buf), terminated)
    return reset, terminated