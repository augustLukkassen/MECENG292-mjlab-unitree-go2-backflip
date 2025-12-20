"""Velocity tracking task configuration.

This module defines the base configuration for velocity tracking tasks.
Robot-specific configurations are located in the config/ directory.
"""

import math
from copy import deepcopy

from mjlab.entity.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  ActionTermCfg,
  CommandTermCfg,
  CurriculumTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=5,
  ),
  num_envs=1,
  extent=2.0,
)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="",  # Override in robot cfg.
  distance=3.0,
  elevation=-5.0,
  azimuth=90.0,
)

SIM_CFG = SimulationCfg(
  nconmax=35,
  njmax=300,
  mujoco=MujocoCfg(
    timestep=0.005,
    iterations=10,
    ls_iterations=20,
  ),
)


def create_velocity_env_cfg(
  robot_cfg: EntityCfg,
  action_scale: float | dict[str, float],
  viewer_body_name: str,
  site_names: tuple[str, ...],
  feet_sensor_cfg: ContactSensorCfg,
  self_collision_sensor_cfg: ContactSensorCfg,
  foot_friction_geom_names: tuple[str, ...] | str,
  posture_std_standing: dict[str, float],
  posture_std_walking: dict[str, float],
  posture_std_running: dict[str, float],
) -> ManagerBasedRlEnvCfg:
  """Create a velocity locomotion task configuration.

  Args:
    robot_cfg: Robot configuration (with sensors).
    action_scale: Action scaling factor(s).
    viewer_body_name: Body for camera tracking.
    site_names: List of site names for foot height/clearance.
    feet_sensor_cfg: Contact sensor config for feet-ground contact.
    self_collision_sensor_cfg: Contact sensor config for self-collision.
    foot_friction_geom_names: Geometry names for friction randomization.
    posture_std_standing: Joint std devs for standing posture reward.
    posture_std_walking: Joint std devs for walking posture reward.
    posture_std_running: Joint std devs for running posture reward.

  Returns:
    Complete ManagerBasedRlEnvCfg for velocity task.
  """
  scene = deepcopy(SCENE_CFG)

  scene.entities = {"robot": robot_cfg}

  scene.sensors = (
    feet_sensor_cfg,
    self_collision_sensor_cfg,
  )

  # Enable curriculum mode for terrain generator.
  if scene.terrain is not None and scene.terrain.terrain_generator is not None:
    scene.terrain.terrain_generator.curriculum = True

  viewer = deepcopy(VIEWER_CONFIG)
  viewer.body_name = viewer_body_name

  # Actions are provided for you.
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      asset_name="robot",
      actuator_names=(".*",),
      scale=action_scale,
      use_default_offset=True,
    )
  }

  # ---------------------------------------------------------------------------
  # Part2 (b) Specify command
  # ---------------------------------------------------------------------------
  # TODO(b): define command(s) for the velocity tracking task.
  # The task is to track a desired linear and yaw velocity (twist).
  # Hint: use a `UniformVelocityCommandCfg`.
  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      asset_name="robot",
      resampling_time_range=(3.0, 8.0), 
      rel_standing_envs=0.1,
      heading_command=True,  
      rel_heading_envs=0.3,  
      heading_control_stiffness=1.0, 
      # ranges=UniformVelocityCommandCfg.Ranges(
      #   lin_vel_x=(-1.0, 1.0),  
      #   lin_vel_y=(-1.0, 1.0),  
      #   ang_vel_z=(-1.0, 1.0), 
      #   heading=(-math.pi, math.pi),  
      # ),
      ranges=UniformVelocityCommandCfg.Ranges( #manually overriding here
        lin_vel_x=(0.0, 0.0),  
        lin_vel_y=(0.5, 0.5),  
        ang_vel_z=(0.0, 0.0), 
        heading=(-math.pi, math.pi),  
      ),
    )
  }

  # ---------------------------------------------------------------------------
  # Part2 (f) Writing observations
  # ---------------------------------------------------------------------------
  # TODO(f): define observation terms for the policy and critic.
  # Hint: include IMU linear/angular velocities, projected gravity,
  # joint positions/velocities, last actions, and the command.
  # Joint positions are provided as a reference. 
  policy_terms: dict[str, ObservationTermCfg] = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.base_lin_vel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "last_action": ObservationTermCfg(
      func=mdp.last_action,
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
  }

  critic_terms = {
    **policy_terms,
    # ---------------------------------------------------------------------------
    # Part3 (b) Writing asymmetric critic -- extra critic observations
    # ---------------------------------------------------------------------------
    # TODO(b): add extra observations for the critic here.
    # Hint: Consider gait information such as foot contact, air time, or height.

    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=site_names)},
    ),

    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": feet_sensor_cfg.name},
    ),

    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": feet_sensor_cfg.name},
    ),

    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": feet_sensor_cfg.name},
    ),
  }


    

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True #Policy gets noisy observations prolly true
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False, #Critic gets clean observations (asymmetric) false
    ),
  }

  # ---------------------------------------------------------------------------
  # Events (reset, pushes, domain randomization)
  # ---------------------------------------------------------------------------
  events = {
    # Reset functions for the start of an episode are provided for you. 
    # Reset the base position and orientation is necessary because the robot's initial state can vary in the environment.
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
      },
    ),
    # Reset the robot joints to a default position at the start of an episode.
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # -------------------------------------------------------------------------
    # Part2 (g) Domain Randomization
    # -------------------------------------------------------------------------
    # TODO(g): add domain randomization event(s).

    # (1) Randomize the ground friction of the feet using `mdp.randomize_field`.
    "foot_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=foot_friction_geom_names),
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
        "operation": "abs",
      },
    ),

    # (2) Add random velocity perturbations to the base to learn recovery behavior using `mdp.push_by_setting_velocity`.
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
      },
    ),
  }

  # ---------------------------------------------------------------------------
  # Rewards
  # ---------------------------------------------------------------------------
  rewards = {
    # -------------------------------------------------------------------------
    # Part2 (c) Writing objectives (main task objectives)
    # -------------------------------------------------------------------------
    # TODO(c): add objective reward terms.
    # Hint: track commanded linear and angular velocity.

    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={"std": math.sqrt(0.25), "command_name": "twist", "asset_cfg": SceneEntityCfg("robot"),},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={"std": math.sqrt(0.25), "command_name": "twist", "asset_cfg": SceneEntityCfg("robot"),},
    ),

    # -------------------------------------------------------------------------
    # Part2 (d) Writing regularization
    # -------------------------------------------------------------------------
    # TODO(d): add regularization rewards.
    # Regularizations are necessary to induce more smooth, natural and realistic behavior. 
    # For more realistic behavior, add terms such as: 
    # 1. encouraging the robot to remain upright with mdp.flat_orientation
    # 2. penalizing large deviations from default joint positions

    "upright": RewardTermCfg(
      func=mdp.flat_orientation,
      weight=1.0,
      params={"std": math.sqrt(0.2), "asset_cfg": SceneEntityCfg("robot", body_ids=[]),},
    ),
    "default_joint_pos": RewardTermCfg(
      func=mdp.default_joint_position,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot"),},
    ),
    # To prevent reaching physical limits and encourage smooth actions, consider adding terms such as:
    # 3. penalizing norm of action rate
    # 4. penalizing reaching the joint position limits
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.1,
    ),
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    # -------------------------------------------------------------------------
    # Part3 (a) Writing gait terms
    # -------------------------------------------------------------------------
    # TODO(a): add gait-related rewards.
    # Even though with the above rewards, the robot can already walk nicely, its gait is not the most desirable, 
    # i.e. you can observe significant feet dragging and sometimes slipping. 
    # These all create sim-to-real gap because where the robot's feet contact the ground is not perfectly simulated.
    # Thus, we design gait rewards to encourage feet clearance (so that the feet lift sufficiently during walking to avoid dragging) and penalize foot slip.

    # Hint: Consider adding the following gait-related rewards:
    # 1. foot_clearance with mdp.feet_clearance
    # 2. foot_swing_height with mdp.feet_swing_height
    # 3. foot_slip with mdp.feet_slip

    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-2.0,
      params={
        "target_height": 0.10,
        "command_name": "twist",
        "command_threshold": 0.05, 
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),

    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,   
      weight=-0.25,
      params={
        "sensor_name": feet_sensor_cfg.name,  
        "target_height": 0.10,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),

    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": feet_sensor_cfg.name,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),

  }

  # ---------------------------------------------------------------------------
  # Part2 (e) Writing terminations
  # ---------------------------------------------------------------------------
  # TODO(e): define termination conditions.
  # Detect a fall-over / bad-orientation check to avoid continue sampling the environment when the robot is already fallen.
  # Hint: Use mdp.bad_orientation with a threshold of 60 degrees. 
  terminations = {
    "time_out": TerminationTermCfg(
      func=mdp.time_out,
      time_out=True,
    ),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      time_out=False,
      params={"limit_angle": math.pi / 3}, #in fact 60 deg
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    events=events,
    sim=SIM_CFG,
    viewer=viewer,
    decimation=4,
    episode_length_s=20.0,
  )
