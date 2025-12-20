"""Go2 environment configuration for the backflip task."""

from mjlab.asset_zoo.robots.unitree_go2.go2_constants import (
  GO2_ACTION_SCALE,
  get_go2_robot_cfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.backflip.backflip_env_cfg import create_backflip_env_cfg

# Foot geometry names for Go2
GO2_FOOT_NAMES = ("FR", "FL", "RR", "RL")
GO2_FOOT_GEOMS = tuple(f"{name}_foot_collision" for name in GO2_FOOT_NAMES)

# Contact sensor for feet-ground contact
GO2_FEET_SENSOR_CFG = ContactSensorCfg(
  name="feet_ground_contact",
  primary=ContactMatch(mode="geom", pattern=GO2_FOOT_GEOMS, entity="robot"),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
  track_air_time=True,
)

# Self-collision sensor (non-foot body contact with ground)
GO2_SELF_COLLISION_SENSOR_CFG = ContactSensorCfg(
  name="nonfoot_ground_touch",
  primary=ContactMatch(
    mode="geom",
    entity="robot",
    pattern=r".*_collision\d*$",
    exclude=GO2_FOOT_GEOMS,
  ),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found",),
  reduce="none",
  num_slots=1,
)

UNITREE_GO2_BACKFLIP_ENV_CFG = create_backflip_env_cfg(
  robot_cfg=get_go2_robot_cfg(),
  action_scale=GO2_ACTION_SCALE,
  viewer_body_name="trunk",
  feet_sensor_cfg=GO2_FEET_SENSOR_CFG,
  self_collision_sensor_cfg=GO2_SELF_COLLISION_SENSOR_CFG,
  standing_height=0.35,  # Go2 standing height
  peak_height=0.85,      # Target peak height during flip (higher for more room)
  flip_duration=1.0,     # Duration of the backflip (longer = lower required angular velocity)
)
