import math
from copy import deepcopy

from mjlab.entity.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    ActionTermCfg,
    CommandTermCfg,
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
from mjlab.tasks.backflip import mdp
from mjlab.tasks.backflip.mdp import BackflipPhaseCommandCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

SCENE_CFG = SceneCfg(
    terrain=TerrainImporterCfg(
        terrain_type="plane",
    ),
    num_envs=1,
    extent=2.0,
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name="",  # Override in robot cfg
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


def create_backflip_env_cfg(
    robot_cfg,
    action_scale,
    viewer_body_name,
    feet_sensor_cfg,
    self_collision_sensor_cfg,
    standing_height=0.35,
    peak_height=0.7,
    flip_duration=0.8,
    foot_site_names=("FR", "FL", "RR", "RL"),
):
    scene = deepcopy(SCENE_CFG)

    scene.entities = {"robot": robot_cfg}

    scene.sensors = (
        feet_sensor_cfg,
        self_collision_sensor_cfg,
    )

    if scene.terrain is not None and scene.terrain.terrain_generator is not None:
        scene.terrain.terrain_generator.curriculum = True

    viewer = deepcopy(VIEWER_CONFIG)
    viewer.body_name = viewer_body_name

    actions = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=(".*",),
            scale=action_scale,
            use_default_offset=True,
        )
    }

    commands = {
        "backflip": BackflipPhaseCommandCfg(
            asset_name="robot",
            resampling_time_range=(100.0, 100.0),
            flip_duration=flip_duration,
            standing_height=standing_height,
            peak_height=peak_height,
        )
    }

    # Sensor name for foot-related observations and rewards
    feet_sensor_name = feet_sensor_cfg.name

    policy_terms = {
        "backflip_phase": ObservationTermCfg(
            func=mdp.backflip_phase,
            params={"command_name": "backflip"},
        ),
        "target_height": ObservationTermCfg(
            func=mdp.backflip_target_height,
            params={"command_name": "backflip"},
        ),
        "target_pitch_vel": ObservationTermCfg(
            func=mdp.backflip_target_pitch_vel,
            params={"command_name": "backflip"},
        ),
        "x_error": ObservationTermCfg(
            func=mdp.x_error,
            params={"command_name": "backflip"},
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

    # Privileged critic observations (foot-related)
    critic_terms = {
        **policy_terms,
        "target_pitch": ObservationTermCfg(
            func=mdp.backflip_target_pitch,
            params={"command_name": "backflip"},
        ),
        "base_orientation_quat": ObservationTermCfg(
            func=mdp.base_orientation_quat,
        ),
        # Privileged foot observations (critic only)
        "foot_height": ObservationTermCfg(
            func=mdp.foot_height,
            params={"asset_cfg": SceneEntityCfg("robot", site_names=foot_site_names)},
        ),
        "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time,
            params={"sensor_name": feet_sensor_name},
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": feet_sensor_name},
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
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (0.0, 0.0)},
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

    rewards = {
        "track_height": RewardTermCfg(
            func=mdp.track_height,
            weight=2.0,
            params={
                "command_name": "backflip",
                "std": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "track_x_position": RewardTermCfg(
            func=mdp.track_x_position,
            weight=1.5,
            params={
                "command_name": "backflip",
                "std": 0.2,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "track_pitch_velocity": RewardTermCfg(
            func=mdp.track_pitch_velocity,
            weight=6.0,  # Increased from 3.0 to drive rotation harder
            params={
                "command_name": "backflip",
                "std": 2.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "phase_progress": RewardTermCfg(
            func=mdp.phase_progress,
            weight=15.0,
            params={"command_name": "backflip"},
        ),
        "upright_at_landing": RewardTermCfg(
            func=mdp.upright_at_landing,
            weight=2.0,
            params={
                "command_name": "backflip",
                "std": 0.3,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "action_rate": RewardTermCfg(
            func=mdp.action_rate_penalty,
            weight=-0.01,
        ),
        "off_axis": RewardTermCfg(
            func=mdp.off_axis_penalty,
            weight=-8.0,  # Strong penalty to prevent side roll
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "insufficient_rotation": RewardTermCfg(
            func=mdp.insufficient_rotation_penalty,
            weight=-8.0,  # Penalize slow rotation during mid-flip
            params={
                "command_name": "backflip",
                "min_pitch_vel": 4.0,  # Need at least 4 rad/s to get past vertical
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "takeoff_impulse": RewardTermCfg(
            func=mdp.takeoff_impulse,
            weight=17.0,
            params={
                "command_name": "backflip",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "crouch_incentive": RewardTermCfg(
            func=mdp.crouch_incentive,
            weight=8.0,  # Increased to encourage deeper crouch
            params={
                "standing_height": standing_height,
                "crouch_depth": 0.10,  # Match command generator crouch depth
                "crouch_steps": 80,  # More time to reach deep crouch
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "air_time_during_flip": RewardTermCfg(
            func=mdp.air_time_during_flip,
            weight=5.0,
            params={
                "command_name": "backflip",
                "sensor_name": feet_sensor_name,
            },
        ),
        "rotation_consistency": RewardTermCfg(
            func=mdp.rotation_consistency,
            weight=3.0,  # Reward maintaining backward rotation throughout flip
            params={
                "command_name": "backflip",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        "height_too_low": TerminationTermCfg(
            func=mdp.height_too_low,
            time_out=False,
            params={
                "command_name": "backflip",
                "min_height": 0.1,
                "asset_cfg": SceneEntityCfg("robot"),
            },
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
        decimation=2,
        episode_length_s=3.0,
    )
