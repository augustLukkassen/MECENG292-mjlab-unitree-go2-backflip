"""Go2 backflip environment with motion retargeting (imitation) rewards."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab.asset_zoo.robots.unitree_go2.go2_constants import (
  GO2_ACTION_SCALE,
  get_go2_robot_cfg,
)

from mjlab.tasks.backflip.mdp.simple_backflip_command import SimpleBackflipCommandCfg
from mjlab.tasks.backflip.mdp import simple_rewards
from mjlab.tasks.backflip import mdp

# Foot contact sensor
GO2_FOOT_GEOMS = ("FR_foot_collision", "FL_foot_collision", "RR_foot_collision", "RL_foot_collision")
FEET_SENSOR_CFG = ContactSensorCfg(
  name="feet_contact",
  primary=ContactMatch(mode="geom", pattern=GO2_FOOT_GEOMS, entity="robot"),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found",),
  reduce="none",
  num_slots=1,
  track_air_time=True,
)

SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(terrain_type="plane"),
  num_envs=1,
  extent=2.0,
)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="trunk",
  distance=4.0,
  elevation=-15.0,
  azimuth=90.0,
)

SIM_CFG = SimulationCfg(
  nconmax=50,
  njmax=400,
  mujoco=MujocoCfg(
    timestep=0.005,
    iterations=10,
    ls_iterations=20,
  ),
)

scene = deepcopy(SCENE_CFG)
scene.entities = {"robot": get_go2_robot_cfg()}
scene.sensors = (FEET_SENSOR_CFG,)

actions = {
  "joint_pos": JointPositionActionCfg(
    asset_name="robot",
    actuator_names=(".*",),
    scale=GO2_ACTION_SCALE,
    use_default_offset=True,
  )
}

commands = {
  "backflip": SimpleBackflipCommandCfg(
    asset_name="robot",
    feet_sensor_name="feet_contact",
    resampling_time_range=(100.0, 100.0),
    flip_duration=1.0,
    standing_height=0.35,
    crouch_height=0.18,
    peak_height=0.90,
    backward_displacement=0.3,
  )
}

policy_terms = {
  "phase": ObservationTermCfg(
    func=mdp.backflip_phase,
    params={"command_name": "backflip"},
  ),
  "target_height": ObservationTermCfg(
    func=mdp.backflip_target_height,
    params={"command_name": "backflip"},
  ),
  "target_grav_x": ObservationTermCfg(
    func=mdp.backflip_target_grav_x,
    params={"command_name": "backflip"},
  ),
  "target_grav_z": ObservationTermCfg(
    func=mdp.backflip_target_grav_z,
    params={"command_name": "backflip"},
  ),
  "target_x": ObservationTermCfg(
    func=mdp.simple_backflip_target_x,
    params={"command_name": "backflip"},
  ),
  "base_x": ObservationTermCfg(
    func=mdp.base_x,
    noise=Unoise(n_min=-0.02, n_max=0.02),
  ),
  "base_height": ObservationTermCfg(
    func=mdp.base_height,
    noise=Unoise(n_min=-0.02, n_max=0.02),
  ),
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
}

critic_terms = {
  **policy_terms,
  "base_orientation_quat": ObservationTermCfg(
    func=mdp.base_orientation_quat,
  ),
  "foot_contact": ObservationTermCfg(
    func=mdp.foot_contact,
    params={"sensor_name": "feet_contact"},
  ),
  "all_feet_airborne": ObservationTermCfg(
    func=mdp.all_feet_airborne,
    params={"sensor_name": "feet_contact"},
  ),
}

observations = {
  "policy": ObservationGroupCfg(
    terms=policy_terms,
    concatenate_terms=True,
    enable_corruption=True,
  ),
  "critic": ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
  ),
}

events = {
  "reset_base": EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (0.0, 0.0)},
      "velocity_range": {},
    },
  ),
  "reset_robot_joints": EventTermCfg(
    func=mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (0.0, 0.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  ),
}

# Retargeting rewards: motion imitation + max height + minimal regularization
rewards = {
  # Primary: imitate the keyframe trajectory
  "motion_imitation": RewardTermCfg(
    func=simple_rewards.motion_imitation_reward,
    weight=5.0,
    params={"command_name": "backflip", "std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
  ),
  # Secondary: encourage high jumps
  "max_height": RewardTermCfg(
    func=simple_rewards.max_height_reward,
    weight=3.0,
    params={"command_name": "backflip", "asset_cfg": SceneEntityCfg("robot")},
  ),
  # Track orientation (helps with flip direction)
  "track_orientation": RewardTermCfg(
    func=simple_rewards.track_orientation_simple,
    weight=1.0,
    params={"command_name": "backflip", "std": 0.4, "asset_cfg": SceneEntityCfg("robot")},
  ),
  # Regularization
  "off_axis": RewardTermCfg(
    func=simple_rewards.off_axis_simple,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot")},
  ),
  "action_rate": RewardTermCfg(
    func=simple_rewards.action_rate_simple,
    weight=-0.01,
  ),
}

terminations = {
  "time_out": TerminationTermCfg(
    func=mdp.time_out,
    time_out=True,
  ),
}

RETARGET_GO2_BACKFLIP_ENV_CFG = ManagerBasedRlEnvCfg(
  scene=scene,
  observations=observations,
  actions=actions,
  commands=commands,
  rewards=rewards,
  terminations=terminations,
  events=events,
  sim=SIM_CFG,
  viewer=VIEWER_CONFIG,
  decimation=2,
  episode_length_s=1.5,
)
