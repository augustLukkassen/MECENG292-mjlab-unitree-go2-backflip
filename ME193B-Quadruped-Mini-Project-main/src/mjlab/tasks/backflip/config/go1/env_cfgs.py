"""Go1 environment configuration for the backflip task."""

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import (
  GO1_ACTION_SCALE,
  get_go1_robot_cfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.backflip.backflip_env_cfg import create_backflip_env_cfg

# Foot geometry names for Go1
GO1_FOOT_NAMES = ("FR", "FL", "RR", "RL")
GO1_FOOT_GEOMS = tuple(f"{name}_foot_collision" for name in GO1_FOOT_NAMES)

# Contact sensor for feet-ground contact
GO1_FEET_SENSOR_CFG = ContactSensorCfg(
  name="feet_ground_contact",
  primary=ContactMatch(mode="geom", pattern=GO1_FOOT_GEOMS, entity="robot"),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
  track_air_time=True,
)

# Self-collision sensor (non-foot body contact with ground)
GO1_SELF_COLLISION_SENSOR_CFG = ContactSensorCfg(
  name="nonfoot_ground_touch",
  primary=ContactMatch(
    mode="geom",
    entity="robot",
    pattern=r".*_collision\d*$",
    exclude=GO1_FOOT_GEOMS,
  ),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found",),
  reduce="none",
  num_slots=1,
)

UNITREE_GO1_BACKFLIP_ENV_CFG = create_backflip_env_cfg(
  robot_cfg=get_go1_robot_cfg(),
  action_scale=GO1_ACTION_SCALE,
  viewer_body_name="trunk",
  feet_sensor_cfg=GO1_FEET_SENSOR_CFG,
  self_collision_sensor_cfg=GO1_SELF_COLLISION_SENSOR_CFG,
  standing_height=0.28,  # Go1 standing height
  peak_height=0.6,       # Target peak height during flip
  flip_duration=0.7,     # Duration of the backflip
)

