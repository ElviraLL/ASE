"""
This scripts defines the Env class for amp training single character HOI, i.e. carrying box task
* TODO:
    - [ ] Figure out how to load object_pos, object_rot, object_vel, object_ang_vel
    - [ ] Figure out how to reset object states in reset function
    - [ ] Make sure the data grabbing methods are all correct
    - [ ] Add config for object in env.yaml under assets section
"""
import math
import numpy as np
import torch
import os

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils
from env.tasks.humanoid import dof_to_obs


class HumanoidAMPCarry(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        
        self._interaction_amp = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float) # TODO: HOI, is it only 3 here?
        
        # Object-related variables
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)

        num_actors = self.get_num_actors_per_env()
        self._object_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 1, :] # num_envs, 13
        self._object_pos = self._object_root_states[:, 0:3]
        self._object_rot = self._object_root_states[:, 3:7]
        self._object_vel = self._object_root_states[:, 7:10] 
        self._object_ang_vel = self._object_root_states[:, 10:13]

        # Initialize object states
        self._initial_object_root_states = self._object_root_states.clone()
        self._object_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32) + 1

        # Add these new variables
        self._tar_humanoid_speed_min = cfg["env"]["tarHumanoidSpeedMin"]
        self._tar_humanoid_speed_max = cfg["env"]["tarHumanoidSpeedMax"]
        self._tar_object_speed_min = cfg["env"]["tarObjectSpeedMin"]
        self._tar_object_speed_max = cfg["env"]["tarObjectSpeedMax"]
        self._tar_dist_max = cfg["env"].get("tarDistMax", 6)

        # Initialize tensors for target speeds and position
        self._tar_humanoid_speed = torch.ones([self.num_envs], device=self.device, dtype=torch.float)
        self._tar_object_speed = torch.ones([self.num_envs], device=self.device, dtype=torch.float)
        self._target_object_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
    
        # Add these for velocity calculation
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_object_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        carry_bodies = self.cfg["env"]["carryBodies"]
        self._carry_body_ids = self._build_carry_body_ids_tensor(carry_bodies)

        if (not self.headless):
            self._build_marker_state_tensors()

        return
    
    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 2, :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_actor_ids = self._humanoid_actor_ids + 2
        return
    
    def _update_marker(self):
        self._marker_pos[..., 0:2] = self._target_object_pos[..., :2]
        self._marker_pos[..., 2] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(self._marker_actor_ids), len(self._marker_actor_ids))
        return

    def _build_carry_body_ids_tensor(self, hand_bodies):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in hand_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        self._prev_object_pos[:] = self._object_pos # TODO: HOI Make sure that this is correctly updated
        return
    
    def _setup_character_props(self, key_bodies):
        # TODO: self.obs_buf has different size, i.e. it is not updated why?
        super()._setup_character_props(key_bodies)
        self._num_obs += 7 # Add box position, rotation, 
        # self._num_amp_obs_per_step += 7 # Add box position, rotation

    def _reset_task(self, env_ids):
        self._reset_objects(env_ids)
        self._reset_target(env_ids)
        return

    def _reset_objects(self, env_ids):
        n = len(env_ids)
        # Randomize object position and rotation
        self._object_root_states[env_ids, 0:2] = torch.rand((len(env_ids), 2), device=self.device) * 2 - 1  # Random values between -1 and 1 for x and y
        self._object_root_states[env_ids, 2] = self.object_height / 2 # fixed height # TODO: HOI this need to be changed for other objects
        random_angles = torch.rand(len(env_ids), device=self.device) * (2 * math.pi)
        cos_half = torch.cos(random_angles / 2)
        sin_half = torch.sin(random_angles / 2)
        
        self._object_root_states[env_ids, 3:7] = torch.stack([
            torch.zeros_like(cos_half),  # x
            torch.zeros_like(cos_half),  # y
            sin_half,                    # z
            cos_half                     # w
        ], dim=-1)
        
        # Reset velocities to zero
        self._object_root_states[env_ids, 7:13] = 0
        
        # self._reset_target(env_ids)

    def _reset_target(self, env_ids):
        n = len(env_ids)
        
        # Reset target humanoid speed
        tar_humanoid_speed = torch.rand(n, device=self.device) * (self._tar_humanoid_speed_max - self._tar_humanoid_speed_min) + self._tar_humanoid_speed_min
        self._tar_humanoid_speed[env_ids] = tar_humanoid_speed

        # Reset target object speed
        tar_object_speed = torch.rand(n, device=self.device) * (self._tar_object_speed_max - self._tar_object_speed_min) + self._tar_object_speed_min
        self._tar_object_speed[env_ids] = tar_object_speed


        object_pos = self._object_root_states[env_ids, 0:3]
        rand_pos = self._tar_dist_max * (2.0 * torch.rand([n, 3], device=self.device) - 1.0)

        # Reset target object position and rotation
        # self._target_object_pos[env_ids] = torch.rand((len(env_ids), 3), device=self.device) * 2 - 1  # Random values between -1 and 1
        tar_object_pos = object_pos + rand_pos
        tar_object_pos[..., 2] = self.object_height / 2
        self._target_object_pos[env_ids] = tar_object_pos

    def _reset_env_tensors(self, env_ids):
        # Set humanoid root state tensor
        humanoid_actor_ids_int32 = self._humanoid_actor_ids[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(humanoid_actor_ids_int32),
            len(humanoid_actor_ids_int32)
        )
        
        # dof state tensor -> 1536, 2 = num_envs * num_dof_per_env * 2
        # where [：, 0] -> dof_pos
        # where [：, 1] -> dof_vel
        # humanoid_actor_ids -> [0, 2, 4, 6, 8, ...] are the actor idx
        # The function set_dof_state_tensor_indexed applies the values in the given tensor to the actors specified in the actor_index_tensor. 
        # The other actors remain unaffected. This is very useful when resetting only selected actors or environments. The actor indices must 
        # be 32-bit integers, like those obtained from get_actor_index.
        # Set dof state tensor (based on actor index according to the documentation
        self.gym.set_dof_state_tensor_indexed(self.sim, # TODO: set dof state tensor indexed, Maybe it needs its own id
            gymtorch.unwrap_tensor(self._dof_state),  
            gymtorch.unwrap_tensor(humanoid_actor_ids_int32),   
            len(humanoid_actor_ids_int32)
        )

        # We can actially set actor_root_state by passing all index of humanoid and object together and use only one function call for this
        object_actor_ids_int32 = self._object_ids[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(object_actor_ids_int32),
            len(object_actor_ids_int32)
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def _load_humanoid_asset(self):
        """
        Load humanoid assets for all agents and save them in self.humanoid_asset
        """
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]


        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        actuator_props = self.gym.get_asset_actuator_properties(self._humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(self._humanoid_asset, self._body_right_foot_name) # left_Ankle
        left_foot_idx = self.gym.find_asset_rigid_body_index(self._humanoid_asset, self._body_left_foot_name) # right_Ankle
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(self._humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(self._humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(self._humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(self._humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(self._humanoid_asset)

    def _load_object_asset(self):
        """
        Load object assets for all agents and save them in self.object_asset
        """
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["object"]["assetFileName"]

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        # Check if object configuration exists
        if "object" in self.cfg["env"]["asset"]:
            # load materials from meshes
            asset_options.use_mesh_materials = self.cfg["env"]["asset"]["object"].get("useMeshMaterials", False)
        else:
            # If object configuration doesn't exist, set a default value or skip
            asset_options.use_mesh_materials = False

        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

        # override the bogus inertia tensors and center of mass properties in the UCB assets
        # These flags will force the inertial properties to be recomputed from geometry
        asset_options.override_inertia = True
        asset_options.override_com = True

        # use default convex decomposition params
        asset_options.vhacd_enabled = True

        # box mass
        desired_mass = 2.0  # for example, 10 kg

        # Calculate the density based on the desired mass and box dimensions
        unit_length = 0.3
        self.object_height = unit_length
        box_volume = unit_length * unit_length * unit_length  # volume of a 1x1x1 box
        density = desired_mass / box_volume
        asset_options.density = density

        # self._object_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self._object_asset = self.gym.create_box(self.sim, unit_length, unit_length, unit_length, asset_options)

        # self._object_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def _load_marker_asset(self):
        asset_root = "ase/data/assets/mjcf/"
        asset_file = "location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return
    
    def _create_envs(self, num_envs, spacing, num_per_row): # been called
        """
        Create envs for each agent
        """
        self.object_handles = []
        if (not self.headless):
            self._marker_handles = []
            self._load_marker_asset()
        self._load_object_asset() # save to self._object_asset
        super()._create_envs(num_envs, spacing, num_per_row)
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_object(env_id, env_ptr, self._object_asset)
        if (not self.headless):
            self._build_marker(env_id, env_ptr)

    def _build_object(self, env_id, env_ptr, object_asset):
        """
        Build objects for each env
        """
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*get_axis_params(1, self.up_axis_idx))
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        object_handle = self.gym.create_actor(env_ptr, object_asset, pose, "object", col_group, col_filter, segmentation_id)
        # self.gym.set_rigid_body_color(self.sim, object_handle, 0, gymapi.Vec3(0.1, 0.9, 0.1))
        self.object_handles.append(object_handle)
        return
    
    def _build_marker(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 0
        default_pose = gymapi.Transform()
        
        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self._marker_handles.append(marker_handle)
        return

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        object_pos = self._object_pos
        
        carry_reward = compute_carry_reward(
            object_pos, 
            self._prev_object_pos,
            self._target_object_pos[..., :2],
            self._tar_object_speed,
            torch.mean(self._rigid_body_pos[:, self._carry_body_ids, 2], dim=1), # average height of hands #HOI TODO: add self._hand_body_ids when setting character
            object_pos[:, 2], # object height
            self.dt
        )
        
        walk_reward = compute_walk_reward(
            root_pos, 
            self._prev_root_pos,
            root_rot,
            object_pos[:, :2], # target position is in x-y plane
            self._tar_humanoid_speed,
            self.dt
        )
        
        # Combine rewards (you may want to adjust the weights)
        self.rew_buf[:] = 0.6 * carry_reward + 0.4 * walk_reward
        # self.rew_buf[:] = walk_reward
        return

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            interaction = self._rigid_body_pos[:, 0] - self._object_pos[:]
            object_rot = self._object_rot[:]
        else:
            body_pos = self._kinematic_humanoid_rigid_body_states[env_ids, :, 0:3]
            body_rot = self._kinematic_humanoid_rigid_body_states[env_ids, :, 3:7]
            interaction = self._kinematic_humanoid_rigid_body_states[env_ids, 0, 0:3] - self._object_pos[env_ids]
            object_rot = self._object_rot[env_ids]
        
        obs = compute_box_observation(body_pos, body_rot, interaction, object_rot)
        return obs

    def _draw_task(self):
        self._update_marker()
        
        self.gym.clear_lines(self.viewer)

        for i, env_ptr in enumerate(self.envs):
            humanoid_root_pos = self._humanoid_root_states[i, 0:3]
            box_pos = self._object_pos[i, 0:3]
            target_pos = self._marker_pos[i, 0:3]

            # Draw line from humanoid to box
            self._draw_line(env_ptr, humanoid_root_pos, box_pos, color=[1.0, 0.0, 0.0])  # Red line

            # Draw line from box to target
            self._draw_line(env_ptr, box_pos, target_pos, color=[1.0, 0.0, 0.0])  # Red line

        return

    def _draw_line(self, env_ptr, start, end, color):
        # type: (Any, Tensor, Tensor, List[float]) -> None
        verts = torch.cat([start, end]).reshape(1, 6).cpu()
        colors = torch.tensor([color], dtype=torch.float32, device=self.device).cpu()
        self.gym.add_lines(self.viewer, env_ptr, verts.shape[0], verts.numpy(), colors.numpy())
    

@torch.jit.script
def compute_box_observation(body_pos, body_rot, interaction, object_rot):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    interaction = quat_rotate(heading_rot, interaction)
    object_rot = quat_mul(heading_rot, object_rot)
    obs = torch.cat((interaction, object_rot), dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_observations_max_interaction(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, interaction, object_rot):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]


    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    interaction = quat_rotate(heading_rot, interaction)
    object_rot = quat_mul(heading_rot, object_rot)

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1)) # 为了对每一个物体做heading旋转
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                                  heading_rot_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1],
                                                 local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0],
                                                 local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0],
                                                         body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0],
                                                         body_ang_vel.shape[1] * body_ang_vel.shape[2])

    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, interaction, object_rot), dim=-1)
    return obs

@torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs


@torch.jit.script
def build_amp_observations_interaction(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos,
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets, interaction, object_rot):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int], Tensor, Tensor) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    interaction = quat_rotate(heading_rot, interaction)
    object_rot = quat_mul(heading_rot, object_rot)

    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
                                           local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0],
                                            local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat(
        (root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos, interaction, object_rot), dim=-1)
    return obs


@torch.jit.script
def compute_carry_reward(
    object_pos, 
    prev_object_pos,
    target_object_pos,
    tar_object_speed,
    humanoid_hand_height, 
    box_height,
    dt
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    distance_threshold = 0.5

    pos_err_scale = 0.5
    vel_err_scale = 4.0
    height_err_scale = 10.0
    carry_near_err_scale = 10.0

    pos_diff = target_object_pos - object_pos[..., 0:2] # TODO: make target pos in x-y plane
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    tar_dir = target_object_pos - object_pos[..., 0:2] # TODO: make target pose in x-y plane
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

    delta_object_pos = object_pos - prev_object_pos
    object_vel = delta_object_pos / dt
    
    tar_dir_speed = torch.sum(tar_dir * object_vel[..., :2], dim=-1) # project object vel onto tar dir
    tar_vel_err = tar_object_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <=0
    vel_reward[speed_mask] = 0



    height_diff = humanoid_hand_height - box_height
    height_err = torch.sum(height_diff * height_diff, dim=-1)
    height_reward = torch.exp(-height_err_scale * height_err)

    
    carry_far_reward = pos_reward + vel_reward + height_reward
    carry_near_reward = torch.exp(-carry_near_err_scale * pos_err)
    
    # Combine rewards based on distance condition

    r = torch.where(pos_err > 0.5, 
                    carry_far_reward + carry_near_reward, 
                    0.2 + carry_near_reward)
    
    return r

@torch.jit.script
def compute_walk_reward(root_pos, prev_root_pos, root_rot, tar_pos, tar_speed, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    dist_threshold = 0.5

    pos_err_scale = 0.5
    vel_err_scale = 4.0

    pos_reward_w = 0.5
    vel_reward_w = 0.4
    face_reward_w = 0.1
    
    pos_diff = tar_pos - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    tar_dir = tar_pos - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    
    
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0


    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)


    dist_mask = pos_err < dist_threshold
    facing_reward[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    reward = pos_reward_w * pos_reward + vel_reward_w * vel_reward + face_reward_w * facing_reward

    return reward