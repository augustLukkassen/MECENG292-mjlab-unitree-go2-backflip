"""Unitree Go1 constants (Student Version with TODOs and Explanations)."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

# ---------------------------------------------------------------------------
# Part2 (a) Specify robot
# ---------------------------------------------------------------------------
# Follow style in src/mjlab/asset_zoo/robots/unitree_go1/go1_constants.py
# Write a Go2 module
# The XML is provided in src/mjlab/asset_zoo/robots/unitree_go2/xmls/go2.xml

# TODO: Load the correct "go1.xml" file for your setup.
GO1_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_go1" / "xmls" / "go1.xml"
)
assert GO1_XML.exists(), f"GO1 XML not found at {GO1_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    """Load mesh/texture assets for the robot."""
    assets: dict[str, bytes] = {}
    update_assets(assets, GO1_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    """Load the MJCF file and attach required assets."""
    spec = mujoco.MjSpec.from_file(str(GO1_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config.
##

# Rotor inertia (from Go1 URDF; rotation about x-axis).
ROTOR_INERTIA = 0.000111842

# Gear ratios for hip and knee joints.
HIP_GEAR_RATIO = 6
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5

# --------------------------------------------------------------------------#
# Students must fill in the missing effort and velocity limits.
# These come from the real Go1 hardware specs. Refer to the writeup for values.
# --------------------------------------------------------------------------#

HIP_ACTUATOR = ElectricActuator(
    reflected_inertia=reflected_inertia(
        ROTOR_INERTIA, HIP_GEAR_RATIO),  # J_r * g^2
    velocity_limit=30.1,   # Max joint velocity (rad/s)
    effort_limit=23.7,     # Torque limit (Nm)
)

# Knee actuator: Uses higher gear ratio (6 * 1.5 = 9) for more torque.
KNEE_ACTUATOR = ElectricActuator(
    reflected_inertia=reflected_inertia(
        ROTOR_INERTIA, KNEE_GEAR_RATIO),  # J_r * g^2
    velocity_limit=20.06,  # Max joint velocity (rad/s)
    effort_limit=35.55,    # Torque limit (Nm)
)

# These are typical choices for stable position-control hardware.
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535   # 10 Hz stiffness shaping
DAMPING_RATIO = 2.0                        # Critically damped-ish behaviour

# --------------------------------------------------------------------------#
# We provide a heuristic formula to compute PD gains as follows,
# stiffness = J * ω^2  and damping = 2ζJω
# for both hip and knee actuators.
# --------------------------------------------------------------------------#

STIFFNESS_HIP = HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ ** 2
DAMPING_HIP = 2.0 * DAMPING_RATIO * HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_KNEE = KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ ** 2
DAMPING_KNEE = 2.0 * DAMPING_RATIO * \
    KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ

# Builtin PD position actuators for hip and knee joints.
GO1_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*_hip_joint", ".*_thigh_joint"),
    stiffness=STIFFNESS_HIP,
    damping=DAMPING_HIP,
    effort_limit=HIP_ACTUATOR.effort_limit,
    armature=HIP_ACTUATOR.reflected_inertia,
)

GO1_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*_calf_joint",),
    stiffness=STIFFNESS_KNEE,
    damping=DAMPING_KNEE,
    effort_limit=KNEE_ACTUATOR.effort_limit,
    armature=KNEE_ACTUATOR.reflected_inertia,
)

##
# Keyframe initial state.
##
# These joint angles represent a stable “standing” pose for Go1.
#

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.278),
    joint_pos={
        ".*thigh_joint": 0.9,
        ".*calf_joint": -1.8,
        ".*R_hip_joint": 0.1,
        ".*L_hip_joint": -0.1,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##
# Students do NOT modify these, but they should understand them.
#
# _foot_regex identifies all foot collision geoms.
# FEET_ONLY_COLLISION: disables all collisions except feet.
# FULL_COLLISION: enables collisions everywhere but with special foot rules.
#

_foot_regex = "^[FR][LR]_foot_collision$"

FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(_foot_regex,),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
    solimp=(0.9, 0.95, 0.023),
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={_foot_regex: 3, ".*_collision": 1},
    priority={_foot_regex: 1},
    friction={_foot_regex: (0.6,)},
    solimp={_foot_regex: (0.9, 0.95, 0.023)},
    contype=1,
    conaffinity=0,
)

##
# Final articulation config (students do not change this).
##

GO1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        GO1_HIP_ACTUATOR_CFG,
        GO1_KNEE_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_go1_robot_cfg() -> EntityCfg:
    """Return a fresh Go1 robot configuration.

    Students should not modify this; the purpose is to ensure
    environment instantiation always receives a clean config.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=GO1_ARTICULATION,
    )


##
# Action scaling computation.
##
# This automatically computes per-joint action scaling based on
# actuator strength and stiffness.
# Students should examine this, but not modify it.
##

GO1_ACTION_SCALE: dict[str, float] = {}
for a in GO1_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.joint_names_expr
    assert e is not None
    for n in names:
        GO1_ACTION_SCALE[n] = 0.25 * (e / s)
