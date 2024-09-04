"""
This scripts defines the Env class for amp training single character HOI, i.e. carrying box task
* TODO:
    - [ ] Figure out how to load object_pos, object_rot, object_vel, object_ang_vel
    - [ ] Figure out how to reset object states in reset function
    - [ ] Make sure the data grabbing methods are all correct
    - [ ] Add config for object in env.yaml under assets section
"""
import numpy as np
import torch

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from env.tasks.humanoid_amp import HumanoidAMP # TODO: maybe we should extend HumanoidAMPTask instead?


class Humanoid_amp_carry(HumanoidAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # TODO: Need to add object's tracking variables here (matches those in Humanoid)
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        
        
        # Object-related variables
        self._object_root_states = None
        self._object_actor_ids = None
        self._object_pos = None
        self._object_rot = None
        self._object_vel = None
        self._object_ang_vel = None

        # Initialize object states
        self._initial_object_root_states = None
        self._initial_object_pos = None
        self._initial_object_rot = None

        # Target object states (to be randomly set for each environment)
        self._target_object_pos = None
        self._target_object_rot = None

        # Function to set random target positions and rotations
        self._set_random_target_states()

    def _set_random_target_states(self):
        # Set random target positions and rotations for each environment
        self._target_object_pos = torch.rand((self.num_envs, 3), device=self.device) * 2 - 1  # Random values between -1 and 1
        self._target_object_rot = torch.nn.functional.normalize(torch.rand((self.num_envs, 4), device=self.device), dim=1)  # Random unit quaternions


        self._object_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self._object_rot = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
        self._demo_object_pos = torch.tensor((0.03717821929603815, 0.8663888734129428, 0.3713016912362804),
                                             device=self.device, dtype=torch.float)
        self._demo_object_rot = torch.tensor((0,0,0,1),
                                             device=self.device, dtype=torch.float)

        self._object_rot[:] = torch.tensor((0,0,0,1), device=self.device, dtype=torch.float)
        self._object_pos[:] = torch.tensor((0.03717821929603815, 0.8663888734129428, 0.3713016912362804),
                                           device=self.device, dtype=torch.float)
        self._tar_hip_pos = torch.tensor((0.0426, 0.8398, 0.6286),device=self.device, dtype=torch.float)
        #self._object_pos[:] = torch.tensor((-3.57, 0, 0), device=self.device, dtype=torch.float)
        # self._demo_object_pos = torch.tensor((-3.57, 0, 0), device=self.device, dtype=torch.float)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        object_asset_file = self.cfg["env"]["asset"]["object"]["assetFileName"]
        object_asset_root = self.cfg["env"]["asset"]["object"]["assetRoot"]

        humanoid_asset, object_asset = self._load_assets(asset_root, asset_file, object_asset_root, object_asset_file)

        self.humanoid_handles = []
        self.object_handles = []
        self.envs = []

        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)

            # create humanoid actor
            humanoid_handle = self._create_humanoid_actor(env_ptr, humanoid_asset, i)
            self.humanoid_handles.append(humanoid_handle)

            # create object actor
            object_handle = self._create_object_actor(env_ptr, object_asset, i)
            self.object_handles.append(object_handle)

        self.humanoid_actor_ids = torch.arange(num_envs, dtype=torch.int32, device=self.device)
        self.object_actor_ids = torch.arange(num_envs, dtype=torch.int32, device=self.device) + num_envs

    def _load_assets(self, asset_root, asset_file, object_asset_root, object_asset_file):
        asset_path = os.path.join(asset_root, asset_file)
        object_asset_path = os.path.join(object_asset_root, object_asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        object_asset = self.gym.load_asset(self.sim, object_asset_root, object_asset_file, asset_options)

        return humanoid_asset, object_asset

    def _create_humanoid_actor(self, env_ptr, humanoid_asset, env_id):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*get_axis_params(1.34, self.up_axis_idx))
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, pose, "humanoid", env_id, 0, 0)
        self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, self._dof_props)

        return humanoid_handle

    def _create_object_actor(self, env_ptr, object_asset, env_id):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*get_axis_params(0.5, self.up_axis_idx))
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        object_handle = self.gym.create_actor(env_ptr, object_asset, pose, "object", env_id, 0, 0)

        return object_handle

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        self._reset_humanoids(env_ids)
        self._reset_objects(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        self._compute_amp_observations(env_ids)

        return self.obs_buf[env_ids]

    def _reset_humanoids(self, env_ids):
        # Implement humanoid reset logic here
        pass

    def _reset_objects(self, env_ids):
        # Implement object reset logic here
        pass

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _compute_observations(self, env_ids):
        # Implement observation computation logic here
        pass

    def _compute_amp_observations(self, env_ids):
        # Implement AMP-specific observation computation logic here
        pass

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
        right_foot_idx = self.gym.find_asset_rigid_body_index(self._humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(self._humanoid_asset, "left_foot")
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
        asset_root = self.cfg["env"]["asset"]["object"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["object"]["assetFileName"]

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
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
        asset_options.compute_inertia = True
        asset_options.override_com = True

        # use default convex decomposition params
        asset_options.vhacd_enabled = True
        self._object_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def _create_envs(self, num_envs, spacing, num_per_row):
        """
        Create envs for each agent
        """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self._load_humanoid_asset() # save to self._humanoid_asset
        self._load_object_asset() # save to self._object_asset

        self.humanoid_handles = []
        self.object_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        # self._load_scene_asset()
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, self._humanoid_asset) # this function is defined in Humanoid.py
            self._build_objects(i, env_ptr, self._object_asset) # use different function to build scene (load the movable object)
            self.envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()
        return

    def _build_objects(self, env_idx, env_ptr, object_asset):
        """
        Build objects for each env
        """
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.r = gymapi.Quat(0, 0.0, 0.0, 1.0)

        object_handle = self.gym.create_actor(env_ptr, object_asset, default_pose, "object", col_group, col_filter, segmentation_id)
        self.object_handles.append(object_handle)
        return

    def _compute_reward(self, actions):  
        # TODO: HOI compute reward (for box right now)
        return
    
    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            interaction = self._rigid_body_pos[:, 0] - self._object_pos[:]
            object_rot = self._object_rot[:]
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            interaction = self._rigid_body_pos[env_ids, 0] - self._object_pos[env_ids]
            object_rot = self._object_rot[env_ids]
        obs = compute_humanoid_observations_max_interaction(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                self._root_height_obs, interaction, object_rot)
        return obs


    def build_amp_obs_demo(self, motion_ids, motion_times0):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        interaction = root_pos[:] - self._object_pos[0]
        chair_rot = torch.zeros((root_pos.shape[0], 4), device=self.device)
        chair_rot[:] = self._demo_object_rot
        amp_obs_demo = build_amp_observations_interaction(root_pos, root_rot, root_vel, root_ang_vel,
                                                dof_pos, dof_vel, key_pos,
                                                self._local_root_obs, self._root_height_obs,
                                                self._dof_obs_size, self._dof_offsets, interaction,chair_rot)

        return amp_obs_demo

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        interaction = root_pos[:] - self._object_pos[0]
        chair_rot = torch.zeros((root_pos.shape[0],4),device=self.device)
        chair_rot[:] = self._demo_object_rot
        amp_obs_demo = build_amp_observations_interaction(root_pos, root_rot, root_vel, root_ang_vel,
                                                dof_pos, dof_vel, key_pos,
                                                self._local_root_obs, self._root_height_obs,
                                                self._dof_obs_size, self._dof_offsets, interaction, chair_rot)

        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return


    def _compute_amp_observations(self, env_ids=None):
        """
        Compute interaction observations for each env
        """
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        self._interaction_amp[:] = self.rigid_body_pos[:, 0] - self._object_pos[:] # TODO: how to load self._object_pos？

        if env_ids is None:
            self._curr_amp_obs_buf[:] = build_amp_observations_interaction(
                self.rigid_body_pos[:, 0, :],
                self.rigid_body_rot[:, 0, :],
                self.rigid_body_vel[:, 0, :],
                self.rigid_body_ang_vel[:, 0, :],
                self.dof_pos,
                self.dof_vel,
                key_body_pose,
                self._local_root_obs,
                self._root_height_obs,
                self._dof_obs_size,
                self._dof_offsets,
                self._interaction_amp[:],
                self._object_rot[:]
            )
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations_interaction(
                self._rigid_body_pos[env_ids][:, 0, :],
                self._rigid_body_rot[env_ids][:, 0, :],
                self._rigid_body_vel[env_ids][:, 0, :],
                self._rigid_body_ang_vel[env_ids][:, 0, :],
                self._dof_pos[env_ids], 
                self._dof_vel[env_ids], 
                key_body_pos[env_ids],
                self._local_root_obs, 
                self._root_height_obs,
                self._dof_obs_size, 
                self._dof_offsets, 
                self._interaction_amp[env_ids],
                self._object_rot[env_ids]
            )
        return




@torch.jit.script
def compute_humanoid_observations_max_interaction(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, interaction, object_rot):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor) -> Tensor
    root_pos = body_pos[:, 0, :]
    object_rot = body_rot[:, 0, :]


    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(object_rot)

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
def compute_box_reward(object_pos, target_object_pos, object_rot, target_object_rot):
    """
    Compute the reward based on the distance between the object and the target object
    """
    # TODO: Implement the reward computation
    pass
