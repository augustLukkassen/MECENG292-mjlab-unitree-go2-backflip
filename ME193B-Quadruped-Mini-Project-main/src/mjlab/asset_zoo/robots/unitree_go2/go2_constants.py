from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

# ---------------------------------------------------------------------------
# Part4 (a) Specify robot
# ---------------------------------------------------------------------------

GO2_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_go2" / "xmls" / "go2.xml"
)
assert GO2_XML.exists(), f"GO2 XML not found at {GO2_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    """Load mesh/texture assets for the Go2 robot."""
    assets: dict[str, bytes] = {}
    update_assets(assets, GO2_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    """Load the MJCF file and attach required assets."""
    spec = mujoco.MjSpec.from_file(str(GO2_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config.
##

# Rotor inertia (same as Go1; rotation about x-axis).
ROTOR_INERTIA = 0.000111842

# Gear ratios for hip and knee joints.
HIP_GEAR_RATIO = 6
KNEE_GEAR_RATIO = 12

# --------------------------------------------------------------------------#
# Students must fill in the missing effort and velocity limits.
# These come from the real Go1 hardware specs. Refer to the writeup for values.
# --------------------------------------------------------------------------#

HIP_ACTUATOR = ElectricActuator(
    reflected_inertia=reflected_inertia(
        ROTOR_INERTIA, HIP_GEAR_RATIO),  # J_r * g^2
    velocity_limit=30.1,   # Max joint velocity (rad/s) - same as Go1
    effort_limit=23.7,     # Torque limit (Nm) - same as Go1
)

# Knee actuator for Go2: Uses gear ratio of 12.
KNEE_ACTUATOR = ElectricActuator(
    reflected_inertia=reflected_inertia(
        ROTOR_INERTIA, KNEE_GEAR_RATIO),  # J_r * g^2
    velocity_limit=15.70,  # Max joint velocity (rad/s) - specified for Go2
    effort_limit=45.43,    # Torque limit (Nm) - specified for Go2
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
GO2_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*_hip_joint", ".*_thigh_joint"),
    stiffness=STIFFNESS_HIP,
    damping=DAMPING_HIP,
    effort_limit=HIP_ACTUATOR.effort_limit,
    armature=HIP_ACTUATOR.reflected_inertia,
)

GO2_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*_calf_joint",),
    stiffness=STIFFNESS_KNEE,
    damping=DAMPING_KNEE,
    effort_limit=KNEE_ACTUATOR.effort_limit,
    armature=KNEE_ACTUATOR.reflected_inertia,
)

##
# Keyframe initial state.
##
# These joint angles represent a stable “standing” pose for Go2.
#

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.35),  # Slightly lower than XML default for stable standing
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

GO2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        GO2_HIP_ACTUATOR_CFG,
        GO2_KNEE_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_go2_robot_cfg() -> EntityCfg:
    """Return a fresh Go2 robot configuration.

    Ensures environment instantiation always receives a clean config.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=GO2_ARTICULATION,
    )


##
# Action scaling computation.
##
# Automatically computes per-joint action scaling based on
# actuator strength and stiffness.
##

GO2_ACTION_SCALE: dict[str, float] = {}
for a in GO2_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.joint_names_expr
    assert e is not None
    for n in names:
        GO2_ACTION_SCALE[n] = 0.25 * (e / s)
