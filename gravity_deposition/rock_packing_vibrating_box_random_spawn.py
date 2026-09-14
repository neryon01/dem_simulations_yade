import gts
from yade import export
from yade import pack
from yade import utils
from yade import Vector3

import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


O.reset()

# Enable YADE's built-in cumulative per-engine timing.
# Each engine stores total execution time across the whole simulation in
# engine.execTime and the number of calls in engine.execCount.
O.timingEnabled = True


# ============================================================
# ALL USER / PARAMETER-STUDY SETTINGS
# ============================================================
# Defaults define the basic case requested here:
#   - irregular STL/GTS rocks
#   - random x/y spawn positions
#   - gravity deposition
#   - no vibration
#   - no compression
#
utils.readParamsFromTable(
    description="gravity_total",
    runId=0,

    # Independent random streams. Changing spawnSeed changes only insertion
    # positions; changing rockTypeSeed changes only the rock-type sequence.
    spawnSeed=24680,
    rockTypeSeed=13579,

    # Rocks and spawning
    nRocks=80,
    # Deterministic repeating order: rock_1, rock_2, rock_3, rock_4, ...
    # Study values are multiples of four, so every rock type occurs equally.
    rockSelectionMode="sequence",
    spawnMode="random",            # "random" or "center"
    spawnWallMarginFracL=0.6,      # horizontal wall margin / L
    spawnZFracL=5.0,               # absolute spawn-center z / L
    insertionPeriodSeconds=0.70,   # physical simulation time between insertions [s]

    # Box geometry. geom.facetBox uses HALF-extents. x and y deliberately
    # share one setting; z can be adjusted independently.
    boxHalfExtentXYFracL=2.0,      # box half-width / L in x and y
    boxHalfExtentZFracL=2.8,       # box half-height / L in z
    boxCenterZFracL=2.0,           # box-center z / L

    # Hertz-Mindlin materials
    # Suhr and Six material table. Friction inputs are coefficients mu;
    # FrictMat receives atan(mu) below because it expects an angle [rad].
    rockYoung=20.0e9,
    rockPoisson=0.20,
    rockFrictionCoefficient=0.45,
    rockDensity=2660.0,
    boxYoung=200.0e9,
    boxPoisson=0.28,
    boxFrictionCoefficient=0.50,
    boxDensity=7833.34,

    # Optional sinusoidal vibration
    enableVibration=0,             # 0/1; default basic case = off
    vibrationAxis="x",            # "x" or "z"
    vibAmplitudeFrac=0.1,          # amplitude / box size along vibration axis
    vibPeriodSteps=3000,           # iterations per full cycle
    vibrationSteps=30000,          # total vibration duration [iterations]
    vibrationRampSteps=3000,       # smooth ramp at each end [iterations]

    # Optional fixed-displacement compression
    enableCompression=0,           # 0/1; may be combined with vibration
    compressionWidthReductionFrac=0.45,
    # Above is TOTAL horizontal width reduction. Each opposing side wall
    # moves by half of this fraction times the initial box width.
    compressionSteps=20000,
    lidHeightMode="above_pile",    # "above_pile" or "fixed_fraction"
    lidHeightFraction=0.8,         # used only for "fixed_fraction"
    lidClearanceFracL=0.05,        # used only for "above_pile"
    compressFloor=0,               # 0/1
    compressTop=0,                 # 0/1; lid exists either way

    # Settling, time step, and output
    # A phase changes only after unbalanced force remains below its threshold
    # for settleHoldSteps consecutive iterations. Every rock must also have a
    # current external contact, so free flight cannot look falsely settled.
    settleUbThreshold=1.0e-3,      # before vibration/compression
    finalUbThreshold=1.0e-3,       # after the final operation
    settleHoldSteps=2000,
    minimumSettlingSteps=5000,
    # Fallback deadline counted only AFTER the final rock is inserted. Normal
    # equilibrium can finish the run earlier. If this limit is reached, the
    # run still exports geometry and performs all requested post-processing.
    maximumIterationsAfterLastInsertion=5000000,
    # Escape guard. A complete clump is erased when the lowest surface of any
    # member sphere lies farther than this margin below the CURRENT box floor.
    # The margin is scaled by the characteristic rock size L.
    escapeBelowFloorMarginFracL=0.50,
    escapeCheckPeriod=100,
    newtonDamping=0.4,
    timestepSafety=0.25,
    vtkExportPeriod=0,              # <= 0 disables periodic VTK; final frame remains
    poseExportPeriod=800,
    residualExportPeriod=1000,
    # Evaluated once, after settling, using YADE's sphere-sphere utility.
    maximumAllowedOverlapFraction=0.01,  # 1% of the relevant sphere radius
    enablePostProcessing=1,        # reconstruct STL + calculate porosity/contacts
    # Axial inset for the side-wall-inclusive global porosity and six-side
    # inset for the separately reported interior porosity. The untrimmed bed
    # remains available as a diagnostic and for the wall-distance profile.
    porosityBoundaryMarginFracL=0.25,

    noTableOk=True,
)
from yade.params import table


# ============================================================
# PARSE AND VALIDATE SETTINGS
# ============================================================

RUN_ID = int(table.runId)
SPAWN_SEED = int(table.spawnSeed)
ROCK_TYPE_SEED = int(table.rockTypeSeed)

# Separate random generators prevent a change in the rock-type draw from
# changing the insertion positions, and vice versa.
spawnRng = random.Random(SPAWN_SEED)
rockTypeRng = random.Random(ROCK_TYPE_SEED)

nRocks = int(table.nRocks)
ROCK_SELECTION_MODE = str(table.rockSelectionMode).strip().lower()
SPAWN_MODE = str(table.spawnMode).strip().lower()
SPAWN_WALL_MARGIN_FRAC_L = float(table.spawnWallMarginFracL)
SPAWN_Z_FRAC_L = float(table.spawnZFracL)
INSERTION_PERIOD_SECONDS = float(table.insertionPeriodSeconds)

BOX_HALF_EXTENT_XY_FRAC_L = float(table.boxHalfExtentXYFracL)
BOX_HALF_EXTENT_Z_FRAC_L = float(table.boxHalfExtentZFracL)
BOX_CENTER_Z_FRAC_L = float(table.boxCenterZFracL)

ROCK_DENSITY = float(table.rockDensity)
ROCK_FRICTION_COEFFICIENT = float(table.rockFrictionCoefficient)
BOX_FRICTION_COEFFICIENT = float(table.boxFrictionCoefficient)

ENABLE_VIBRATION = bool(int(table.enableVibration))
VIBRATION_AXIS = str(table.vibrationAxis).strip().lower()
VIB_AMPLITUDE_FRAC = float(table.vibAmplitudeFrac)
VIB_PERIOD_STEPS = int(table.vibPeriodSteps)
VIBRATION_STEPS = int(table.vibrationSteps)
VIBRATION_RAMP_STEPS = int(table.vibrationRampSteps)

ENABLE_COMPRESSION = bool(int(table.enableCompression))
COMPRESSION_WIDTH_REDUCTION_FRAC = float(table.compressionWidthReductionFrac)
COMPRESSION_STEPS = int(table.compressionSteps)
LID_HEIGHT_MODE = str(table.lidHeightMode).strip().lower()
LID_HEIGHT_FRACTION = float(table.lidHeightFraction)
LID_CLEARANCE_FRAC_L = float(table.lidClearanceFracL)
COMPRESS_FLOOR = bool(int(table.compressFloor))
COMPRESS_TOP = bool(int(table.compressTop))

SETTLE_UB_THRESHOLD = float(table.settleUbThreshold)
FINAL_UB_THRESHOLD = float(table.finalUbThreshold)
SETTLE_HOLD_STEPS = int(table.settleHoldSteps)
MINIMUM_SETTLING_STEPS = int(table.minimumSettlingSteps)
MAXIMUM_ITERATIONS_AFTER_LAST_INSERTION = int(
    table.maximumIterationsAfterLastInsertion
)
ESCAPE_BELOW_FLOOR_MARGIN_FRAC_L = float(table.escapeBelowFloorMarginFracL)
ESCAPE_CHECK_PERIOD = int(table.escapeCheckPeriod)
NEWTON_DAMPING = float(table.newtonDamping)
TIMESTEP_SAFETY = float(table.timestepSafety)
VTK_EXPORT_PERIOD = int(table.vtkExportPeriod)
POSE_EXPORT_PERIOD = int(table.poseExportPeriod)
RESIDUAL_EXPORT_PERIOD = int(table.residualExportPeriod)
MAXIMUM_ALLOWED_OVERLAP_FRACTION = float(table.maximumAllowedOverlapFraction)
ENABLE_POST_PROCESSING = bool(int(table.enablePostProcessing))
POROSITY_BOUNDARY_MARGIN_FRAC_L = float(table.porosityBoundaryMarginFracL)

# The parameter-study runner sets this only while YADE is executing inside the
# Apptainer image.  The image intentionally supplies YADE, whereas the host
# yadepy environment supplies VTK and the remaining analysis packages.  A
# direct/standalone YADE run still performs post-processing here unless this
# launch-only environment flag is set.
DEFER_POST_PROCESSING_TO_RUNNER = (
    os.environ.get("YADE_DEFER_POST_PROCESSING", "0") == "1"
)

if nRocks < 1:
    raise ValueError("nRocks must be at least 1.")
if MAXIMUM_ALLOWED_OVERLAP_FRACTION <= 0.0:
    raise ValueError("maximumAllowedOverlapFraction must be positive.")
if ROCK_SELECTION_MODE not in {"random", "rock_1", "sequence"}:
    raise ValueError(
        "rockSelectionMode must be 'random', 'rock_1', or 'sequence'."
    )
if SPAWN_MODE not in {"random", "center"}:
    raise ValueError("spawnMode must be 'random' or 'center'.")
if VIBRATION_AXIS not in {"x", "y", "z"}:
    raise ValueError("vibrationAxis must be 'x', 'y', or 'z'.")
if ENABLE_VIBRATION and (VIB_PERIOD_STEPS <= 0 or VIBRATION_STEPS <= 0):
    raise ValueError("Vibration period and duration must be positive when vibration is enabled.")
if VIB_AMPLITUDE_FRAC < 0:
    raise ValueError("vibAmplitudeFrac must be non-negative.")
if ENABLE_COMPRESSION and COMPRESSION_STEPS <= 0:
    raise ValueError("compressionSteps must be positive when compression is enabled.")
if not 0.0 <= COMPRESSION_WIDTH_REDUCTION_FRAC < 1.0:
    raise ValueError("compressionWidthReductionFrac must be in [0, 1).")
if LID_HEIGHT_MODE not in {"above_pile", "fixed_fraction"}:
    raise ValueError("lidHeightMode must be 'above_pile' or 'fixed_fraction'.")
if not 0.0 <= LID_HEIGHT_FRACTION <= 1.0:
    raise ValueError("lidHeightFraction must be in [0, 1].")
if INSERTION_PERIOD_SECONDS <= 0:
    raise ValueError("insertionPeriodSeconds must be positive.")
if min(POSE_EXPORT_PERIOD, RESIDUAL_EXPORT_PERIOD) <= 0:
    raise ValueError("Pose-export and residual-export periods must be positive.")
if min(BOX_HALF_EXTENT_XY_FRAC_L, BOX_HALF_EXTENT_Z_FRAC_L) <= 0:
    raise ValueError("Box x/y and z half-extents must be positive.")
if TIMESTEP_SAFETY <= 0:
    raise ValueError("timestepSafety must be positive.")
if ROCK_DENSITY <= 0 or float(table.boxDensity) <= 0:
    raise ValueError("Rock and box densities must be positive.")
if ROCK_FRICTION_COEFFICIENT < 0 or BOX_FRICTION_COEFFICIENT < 0:
    raise ValueError("Friction coefficients mu cannot be negative.")
if SETTLE_UB_THRESHOLD < 0 or FINAL_UB_THRESHOLD < 0:
    raise ValueError("Unbalanced-force settling thresholds cannot be negative.")
if SETTLE_HOLD_STEPS <= 0 or MINIMUM_SETTLING_STEPS < 0:
    raise ValueError("settleHoldSteps must be positive and minimumSettlingSteps non-negative.")
if MAXIMUM_ITERATIONS_AFTER_LAST_INSERTION <= 0:
    raise ValueError(
        "maximumIterationsAfterLastInsertion must be positive."
    )
if ESCAPE_BELOW_FLOOR_MARGIN_FRAC_L < 0:
    raise ValueError("escapeBelowFloorMarginFracL cannot be negative.")
if ESCAPE_CHECK_PERIOD <= 0:
    raise ValueError("escapeCheckPeriod must be positive.")
if POROSITY_BOUNDARY_MARGIN_FRAC_L < 0:
    raise ValueError("porosityBoundaryMarginFracL cannot be negative.")


# ============================================================
# RUN DIRECTORY AND POST-PROCESSING FILES
# ============================================================

ROOT = Path(".").resolve()
RUN_DIR = ROOT / "runs" / f"run_{RUN_ID:03d}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

RECONSTRUCT_SCRIPT = ROOT / "reconstruct_stl_w_distance.py"
POROSITY_SCRIPT = ROOT / "rock_porosity.py"

for stl_name in ["rock_1.stl", "rock_2.stl", "rock_3.stl", "rock_4.stl"]:
    shutil.copy2(ROOT / stl_name, RUN_DIR / stl_name)

wall_time_start = time.time()

print("RUN_ID =", RUN_ID)
print("SPAWN_SEED =", SPAWN_SEED)
print("ROCK_TYPE_SEED =", ROCK_TYPE_SEED)
print("rockSelectionMode =", ROCK_SELECTION_MODE)
print("Basic/random spawn mode =", SPAWN_MODE)
print("insertionPeriodSeconds =", INSERTION_PERIOD_SECONDS)
print("CONTACT_MODEL = Hertz-Mindlin")
print("ENABLE_VIBRATION =", ENABLE_VIBRATION, "axis =", VIBRATION_AXIS)
print("ENABLE_COMPRESSION =", ENABLE_COMPRESSION)
print(
    "Escape guard: erase a complete clump when its lowest member surface is",
    "below the current floor by more than",
    ESCAPE_BELOW_FLOOR_MARGIN_FRAC_L,
    "L",
)


# ============================================================
# LOAD ROCK TEMPLATES AND COMPUTE CHARACTERISTIC SIZE L
# ============================================================

ROCK_GTS_FILES = [
    "rock_1.gts",
    "rock_2.gts",
    "rock_3.gts",
    "rock_4.gts",
]

# Final particle-specific geometry selections. Lattice rotation changes the
# member-sphere arrangement relative to the rock; it is not the initial
# orientation of the complete falling rock.
SELECTED_CLUMP_CONFIGURATIONS = {
    1: {"divisor": 12, "overlapFraction": 0.30, "rotation": "identity", "expectedSpheres": 103},
    2: {"divisor": 9,  "overlapFraction": 0.30, "rotation": "identity", "expectedSpheres": 122},
    3: {"divisor": 11, "overlapFraction": 0.50, "rotation": "identity", "expectedSpheres": 126},
    4: {"divisor": 11, "overlapFraction": 0.30, "rotation": "z_30deg", "expectedSpheres": 118},
}

LATTICE_ROTATIONS = {
    "identity": ((1.0, 0.0, 0.0), 0.0),
    "x_30deg": ((1.0, 0.0, 0.0), 30.0),
    "y_30deg": ((0.0, 1.0, 0.0), 30.0),
    "z_30deg": ((0.0, 0.0, 1.0), 30.0),
}


def rotation_matrix(axis, angle_degrees):
    """Return the right-handed Rodrigues rotation matrix."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x_axis, y_axis, z_axis = axis
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return np.asarray([
        [
            cosine + x_axis * x_axis * one_minus_cosine,
            x_axis * y_axis * one_minus_cosine - z_axis * sine,
            x_axis * z_axis * one_minus_cosine + y_axis * sine,
        ],
        [
            y_axis * x_axis * one_minus_cosine + z_axis * sine,
            cosine + y_axis * y_axis * one_minus_cosine,
            y_axis * z_axis * one_minus_cosine - x_axis * sine,
        ],
        [
            z_axis * x_axis * one_minus_cosine - y_axis * sine,
            z_axis * y_axis * one_minus_cosine + x_axis * sine,
            cosine + z_axis * z_axis * one_minus_cosine,
        ],
    ])


def generate_selected_hexagonal_pack(gts_filename, configuration):
    """Regenerate one selected centre-clipped HEXA candidate exactly."""
    with open(gts_filename, "r") as handle:
        surface = gts.read(handle)

    unrotated_predicate = pack.inGtsSurface(surface, True)
    unrotated_aabb = unrotated_predicate.aabb()
    minimum = np.asarray(unrotated_aabb[0], dtype=float)
    maximum = np.asarray(unrotated_aabb[1], dtype=float)
    center = 0.5 * (minimum + maximum)
    dx, dy, dz = maximum - minimum

    rotation_label = configuration["rotation"]
    rotation_axis, angle_degrees = LATTICE_ROTATIONS[rotation_label]

    if angle_degrees:
        surface.translate(-center[0], -center[1], -center[2])
        surface.rotate(
            rotation_axis[0],
            rotation_axis[1],
            rotation_axis[2],
            math.radians(angle_degrees),
        )
        surface.translate(center[0], center[1], center[2])

    predicate = pack.inGtsSurface(surface, True)  # centre-based clipping
    member_radius = dx / float(configuration["divisor"])
    member_gap = -configuration["overlapFraction"] * member_radius
    member_bodies = pack.regularHexa(
        predicate,
        radius=member_radius,
        gap=member_gap,
    )

    if angle_degrees and member_bodies:
        inverse_rotation = rotation_matrix(rotation_axis, angle_degrees).T
        for member in member_bodies:
            position = np.asarray([
                float(member.state.pos[0]),
                float(member.state.pos[1]),
                float(member.state.pos[2]),
            ])
            mapped = center + inverse_rotation.dot(position - center)
            member.state.pos = Vector3(*mapped)

    return member_bodies, float(dx), float(dy), float(dz), member_radius, member_gap


L = 0.0
spherePacks = []
template_rows = []

for rock_index, gts_filename in enumerate(ROCK_GTS_FILES, start=1):
    configuration = SELECTED_CLUMP_CONFIGURATIONS[rock_index]
    sphere_pack, dx, dy, dz, member_radius, member_gap = (
        generate_selected_hexagonal_pack(gts_filename, configuration)
    )
    L = max(L, dx, dy, dz)

    actual_count = len(sphere_pack)
    expected_count = configuration["expectedSpheres"]
    if actual_count != expected_count:
        raise RuntimeError(
            "Selected clump regeneration mismatch for rock {}: expected {} "
            "member spheres, generated {}. Check that the GTS files are the "
            "same files used in the geometry-screening study.".format(
                rock_index, expected_count, actual_count
            )
        )

    spherePacks.append(sphere_pack)
    template_rows.append({
        "rockType": rock_index,
        "gtsFile": gts_filename,
        "resolutionDivisor": configuration["divisor"],
        "overlapDepthFractionOfRadius": configuration["overlapFraction"],
        "latticeRotation": configuration["rotation"],
        "memberRadius": member_radius,
        "memberGap": member_gap,
        "memberSphereCount": actual_count,
        "dx": dx,
        "dy": dy,
        "dz": dz,
    })
    print(
        "Rock {} template: d={}, overlap={:.2f}r, rotation={}, spheres={}".format(
            rock_index,
            configuration["divisor"],
            configuration["overlapFraction"],
            configuration["rotation"],
            actual_count,
        )
    )

if L <= 0:
    raise RuntimeError("Could not determine a positive characteristic rock size L.")

print("Characteristic rock size L =", L)
pd.DataFrame(template_rows).to_csv(
    RUN_DIR / "clump_template_configurations.csv", index=False
)
print("Saved: clump_template_configurations.csv")


# ============================================================
# MATERIALS AND CONTACT LAW
# ============================================================

rockMat = FrictMat(
    young=float(table.rockYoung),
    poisson=float(table.rockPoisson),
    frictionAngle=math.atan(ROCK_FRICTION_COEFFICIENT),
    density=ROCK_DENSITY,
)
boxMat = FrictMat(
    young=float(table.boxYoung),
    poisson=float(table.boxPoisson),
    frictionAngle=math.atan(BOX_FRICTION_COEFFICIENT),
    density=float(table.boxDensity),
)
matId_rock = O.materials.append(rockMat)
matId_box = O.materials.append(boxMat)

# The study uses Hertz-Mindlin exclusively, with no alternative-law branch or
# contact-law selector in the parameter table.
contact_ip2 = [Ip2_FrictMat_FrictMat_MindlinPhys()]
contact_law2 = [Law2_ScGeom_MindlinPhys_Mindlin()]

print(
    "Rock friction: mu =", ROCK_FRICTION_COEFFICIENT,
    "-> frictionAngle =", rockMat.frictionAngle, "rad"
)
print(
    "Box friction: mu =", BOX_FRICTION_COEFFICIENT,
    "-> frictionAngle =", boxMat.frictionAngle, "rad"
)


# ============================================================
# OPEN-TOP BOX AND RANDOM SPAWN REGION
# ============================================================

BOX_CENTER = Vector3(0, 0, BOX_CENTER_Z_FRAC_L * L)
BOX_HALF_EXTENT_XY = BOX_HALF_EXTENT_XY_FRAC_L * L
BOX_HALF_EXTENT_Z = BOX_HALF_EXTENT_Z_FRAC_L * L

c_box = geom.facetBox(
    BOX_CENTER,
    (BOX_HALF_EXTENT_XY, BOX_HALF_EXTENT_XY, BOX_HALF_EXTENT_Z),
    wallMask=31,  # floor + four sides; open top during insertion
    material=matId_box,
)
O.bodies.append(c_box)

boxIds = [body.id for body in c_box]
boxInitialPos = {bid: Vector3(O.bodies[bid].state.pos) for bid in boxIds}

initialBoxXMin = BOX_CENTER[0] - BOX_HALF_EXTENT_XY
initialBoxXMax = BOX_CENTER[0] + BOX_HALF_EXTENT_XY
initialBoxYMin = BOX_CENTER[1] - BOX_HALF_EXTENT_XY
initialBoxYMax = BOX_CENTER[1] + BOX_HALF_EXTENT_XY
initialBoxFloorZ = BOX_CENTER[2] - BOX_HALF_EXTENT_Z
initialBoxTopZ = BOX_CENTER[2] + BOX_HALF_EXTENT_Z

INITIAL_BOX_WIDTH_X = initialBoxXMax - initialBoxXMin
INITIAL_BOX_WIDTH_Y = initialBoxYMax - initialBoxYMin
INITIAL_BOX_HEIGHT = initialBoxTopZ - initialBoxFloorZ

SPAWN_WALL_MARGIN = SPAWN_WALL_MARGIN_FRAC_L * L
spawnXMin = initialBoxXMin + SPAWN_WALL_MARGIN
spawnXMax = initialBoxXMax - SPAWN_WALL_MARGIN
spawnYMin = initialBoxYMin + SPAWN_WALL_MARGIN
spawnYMax = initialBoxYMax - SPAWN_WALL_MARGIN
spawnZ = SPAWN_Z_FRAC_L * L

if spawnXMin >= spawnXMax or spawnYMin >= spawnYMax:
    raise ValueError("spawnWallMarginFracL leaves no valid horizontal spawn region.")
if spawnZ <= initialBoxTopZ:
    print("WARNING: spawn z is not above the open top of the box.")

print(
    f"Initial box: x=[{initialBoxXMin:.4g},{initialBoxXMax:.4g}] "
    f"y=[{initialBoxYMin:.4g},{initialBoxYMax:.4g}] "
    f"z=[{initialBoxFloorZ:.4g},{initialBoxTopZ:.4g}]"
)
print(
    f"Spawn region: x=[{spawnXMin:.4g},{spawnXMax:.4g}] "
    f"y=[{spawnYMin:.4g},{spawnYMax:.4g}] z={spawnZ:.4g}"
)


# ============================================================
# ROCK INSERTION AND POSE EXPORT
# ============================================================

rockCounter = 0
lastInsertionIter = None

# Cumulative wall-clock time spent specifically in O.bodies.clump(...).
# This is accumulated over every inserted rock in this simulation.
clumpingTimeSeconds = 0.0
clumpingCallCount = 0
rockTypeByBodyId = {}
rockTypeByClumpId = {}
insertedRockTypeSequence = []
localComByClumpId = {}
initialOriByClumpId = {}
rockScaleByClumpId = {}
deletedRockCount = 0
deletedRockIds = []
export.rockTypeByBodyId = rockTypeByBodyId


def insertRock():
    global rockCounter, lastInsertionIter
    global clumpingTimeSeconds, clumpingCallCount

    if rockCounter >= nRocks:
        return

    if SPAWN_MODE == "random":
        spawn_x = spawnRng.uniform(spawnXMin, spawnXMax)
        spawn_y = spawnRng.uniform(spawnYMin, spawnYMax)
    else:
        spawn_x = 0.5 * (initialBoxXMin + initialBoxXMax)
        spawn_y = 0.5 * (initialBoxYMin + initialBoxYMax)

    insertion_center = Vector3(spawn_x, spawn_y, spawnZ)

    if ROCK_SELECTION_MODE == "rock_1":
        rock_type_index = 0
    elif ROCK_SELECTION_MODE == "sequence":
        # rockCounter starts at zero, so the deterministic order is
        # rock_1, rock_2, ..., rock_N, rock_1, ...
        rock_type_index = rockCounter % len(spherePacks)
    else:
        rock_type_index = rockTypeRng.randrange(len(spherePacks))
    template_pack = spherePacks[rock_type_index]
    # The geometry-screened clump is used at its original size. rock_scale is
    # retained in rock_poses.csv only because reconstruction consumes that
    # established column.
    rock_scale = 1.0

    shifted_spheres = [
        sphere(
            rock_scale * member.state.pos + insertion_center,
            rock_scale * member.shape.radius,
            material=matId_rock,
        )
        for member in template_pack
    ]
    member_ids = O.bodies.append(shifted_spheres)

    rock_type = rock_type_index + 1
    for member_id in member_ids:
        rockTypeByBodyId[member_id] = rock_type

    clump_start = time.perf_counter()
    clump_id = O.bodies.clump(member_ids)
    clumpingTimeSeconds += time.perf_counter() - clump_start
    clumpingCallCount += 1
    clump = O.bodies[clump_id]

    local_com = clump.state.pos - insertion_center
    localComByClumpId[clump_id] = (
        local_com[0], local_com[1], local_com[2]
    )

    q0 = clump.state.ori
    initialOriByClumpId[clump_id] = (q0[0], q0[1], q0[2], q0[3])
    rockTypeByClumpId[clump_id] = rock_type
    insertedRockTypeSequence.append(rock_type)
    rockScaleByClumpId[clump_id] = rock_scale

    rockCounter += 1
    lastInsertionIter = O.iter


insertRock()

poseFile = open(RUN_DIR / "rock_poses.csv", "w")
poseFile.write(
    "iter,clumpId,rockType,x,y,z,qx,qy,qz,qw,"
    "localComX,localComY,localComZ,q0x,q0y,q0z,q0w,rockScale\n"
)

residualFile = open(RUN_DIR / "residuals.csv", "w")
residualFile.write(
    "iter,time,unbalancedForce,kineticEnergy,kineticEnergyOverDensity,"
    "settledConsecutiveSteps,vibrationDisplacement,compressionDisplacement,"
    "wallForceMagnitude,phase\n"
)

deletedRockFile = open(RUN_DIR / "deleted_rocks.csv", "w")
deletedRockFile.write(
    "iter,time,clumpId,rockType,centerX,centerY,centerZ,"
    "lowestMemberSurfaceZ,currentBoxFloorZ,escapeThresholdZ,"
    "safetyMargin,safetyMarginFracL,reason\n"
)


def exportClumpPoses():
    for body in O.bodies:
        if not body or not body.isClump:
            continue

        q = body.state.ori
        rock_type = rockTypeByClumpId.get(body.id, -1)
        local_com = localComByClumpId[body.id]
        q0 = initialOriByClumpId[body.id]
        rock_scale = rockScaleByClumpId[body.id]

        poseFile.write(
            f"{O.iter},{body.id},{rock_type},"
            f"{body.state.pos[0]},{body.state.pos[1]},{body.state.pos[2]},"
            f"{q[0]},{q[1]},{q[2]},{q[3]},"
            f"{local_com[0]},{local_com[1]},{local_com[2]},"
            f"{q0[0]},{q0[1]},{q0[2]},{q0[3]},{rock_scale}\n"
        )

    poseFile.flush()


# ============================================================
# BOX FACET HELPERS (USED BY COMPRESSION AND FINAL BOUNDS)
# ============================================================

wallAxisSign = {}
compressWall = {}
lidIds = []


def facet_global_vertices(bid):
    body = O.bodies[bid]
    return [body.state.pos + body.state.ori * vertex for vertex in body.shape.vertices]


def classify_wall(bid):
    """Return (axis, outward sign) for an axis-aligned box facet."""
    v0, v1, v2 = facet_global_vertices(bid)
    normal = (v1 - v0).cross(v2 - v0)
    normal = normal / normal.norm()
    centroid = (v0 + v1 + v2) / 3.0

    if normal.dot(centroid - BOX_CENTER) < 0:
        normal = -normal

    axis = max(range(3), key=lambda i: abs(normal[i]))
    sign = 1 if normal[axis] > 0 else -1
    return axis, sign


for box_id in boxIds:
    wallAxisSign[box_id] = classify_wall(box_id)
    axis, sign = wallAxisSign[box_id]
    compressWall[box_id] = (
        COMPRESS_FLOOR if (axis == 2 and sign == -1) else True
    )


def plane_coordinate(axis, sign, fallback):
    values = []
    for bid in boxIds:
        if wallAxisSign.get(bid) != (axis, sign):
            continue
        vertices = facet_global_vertices(bid)
        values.extend(float(vertex[axis]) for vertex in vertices)
    return float(np.mean(values)) if values else float(fallback)


def current_box_bounds():
    return {
        "boxXMin": plane_coordinate(0, -1, initialBoxXMin),
        "boxXMax": plane_coordinate(0, 1, initialBoxXMax),
        "boxYMin": plane_coordinate(1, -1, initialBoxYMin),
        "boxYMax": plane_coordinate(1, 1, initialBoxYMax),
        "boxFloorZ": plane_coordinate(2, -1, initialBoxFloorZ),
        "boxTopZ": plane_coordinate(2, 1, initialBoxTopZ),
    }


ESCAPE_BELOW_FLOOR_MARGIN = ESCAPE_BELOW_FLOOR_MARGIN_FRAC_L * L


def eraseClumpsBelowFloorEnvelope():
    """Erase complete clumps whose member geometry escaped below the floor."""
    global deletedRockCount

    floor_z = current_box_bounds()["boxFloorZ"]
    threshold_z = floor_z - ESCAPE_BELOW_FLOOR_MARGIN

    minimum_surface_z = {}
    member_ids_by_clump = {}

    # Test actual member-sphere surfaces rather than only the clump centre.
    # Collect IDs first; bodies are erased only after this traversal finishes.
    for body in O.bodies:
        if not body or not isinstance(body.shape, Sphere):
            continue

        clump_id = int(body.clumpId)
        if clump_id < 0 or clump_id not in rockTypeByClumpId:
            continue

        surface_z = float(body.state.pos[2] - body.shape.radius)
        previous = minimum_surface_z.get(clump_id)
        if previous is None or surface_z < previous:
            minimum_surface_z[clump_id] = surface_z
        member_ids_by_clump.setdefault(clump_id, []).append(body.id)

    escaped_clump_ids = sorted(
        clump_id
        for clump_id, surface_z in minimum_surface_z.items()
        if surface_z < threshold_z
    )

    for clump_id in escaped_clump_ids:
        clump = O.bodies[clump_id]
        if not clump:
            continue

        rock_type = rockTypeByClumpId.get(clump_id, -1)
        center = Vector3(clump.state.pos)
        lowest_z = minimum_surface_z[clump_id]

        # YADE requires True here to erase the clump body AND every member.
        erased = O.bodies.erase(clump_id, True)
        if not erased:
            print("WARNING: YADE could not erase escaped clump", clump_id)
            continue

        deletedRockFile.write(
            f"{O.iter},{O.time},{clump_id},{rock_type},"
            f"{center[0]},{center[1]},{center[2]},"
            f"{lowest_z},{floor_z},{threshold_z},"
            f"{ESCAPE_BELOW_FLOOR_MARGIN},"
            f"{ESCAPE_BELOW_FLOOR_MARGIN_FRAC_L},below_floor_envelope\n"
        )
        deletedRockFile.flush()

        for member_id in member_ids_by_clump.get(clump_id, []):
            rockTypeByBodyId.pop(member_id, None)

        rockTypeByClumpId.pop(clump_id, None)
        localComByClumpId.pop(clump_id, None)
        initialOriByClumpId.pop(clump_id, None)
        rockScaleByClumpId.pop(clump_id, None)
        deletedRockIds.append(clump_id)
        deletedRockCount += 1

        print(
            "ERASED ESCAPED ROCK | clumpId =", clump_id,
            "| rockType =", rock_type,
            "| lowest surface z =", lowest_z,
            "| floor envelope z =", threshold_z,
            "| retained rocks =", len(rockTypeByClumpId),
        )


# ============================================================
# VTK EXPORT
# ============================================================

for pattern in ("scene*.vtk", "scene*.vtu"):
    for stale_vtk_file in RUN_DIR.glob(pattern):
        stale_vtk_file.unlink()

vtkExporter = export.VTKExporter(str(RUN_DIR / "scene"))


def exportVTK():
    # Give spheres and facets the same physical iteration label so ParaView
    # cannot pair the box with the wrong rock frame.
    vtkExporter.exportSpheres(
        what={"rockType": "rockTypeByBodyId.get(b.id,0)"},
        numLabel=O.iter,
    )
    vtkExporter.exportFacets(numLabel=O.iter)


# ============================================================
# ENGINES
# ============================================================

engines = [
    ForceResetter(),
    InsertionSortCollider([
        Bo1_Sphere_Aabb(),
        Bo1_Facet_Aabb(),
    ]),
    InteractionLoop(
        [
            Ig2_Sphere_Sphere_ScGeom(),
            Ig2_Facet_Sphere_ScGeom(),
        ],
        contact_ip2,
        contact_law2,
    ),
    NewtonIntegrator(
        gravity=(0, 0, -9.81),
        damping=NEWTON_DAMPING,
        exactAsphericalRot=True,
    ),
    PyRunner(command="insertRock()", virtPeriod=INSERTION_PERIOD_SECONDS),
    PyRunner(
        command="eraseClumpsBelowFloorEnvelope()",
        iterPeriod=ESCAPE_CHECK_PERIOD,
    ),
]

if VTK_EXPORT_PERIOD > 0:
    engines.append(PyRunner(command="exportVTK()", iterPeriod=VTK_EXPORT_PERIOD))

engines.extend([
    PyRunner(command="exportClumpPoses()", iterPeriod=POSE_EXPORT_PERIOD),
    PyRunner(command="addPlotData()", iterPeriod=RESIDUAL_EXPORT_PERIOD),
    PyRunner(command="controlSimulation()", iterPeriod=1),
])

O.engines = engines
O.dt = TIMESTEP_SAFETY * PWaveTimeStep()

print(
    "Maximum iterations after final insertion =",
    MAXIMUM_ITERATIONS_AFTER_LAST_INSERTION,
)


# ============================================================
# OPTIONAL VIBRATION
# ============================================================

axisIndex = {"x": 0, "y": 1, "z": 2}[VIBRATION_AXIS]
boxSizeAlongVibrationAxis = [
    INITIAL_BOX_WIDTH_X,
    INITIAL_BOX_WIDTH_Y,
    INITIAL_BOX_HEIGHT,
][axisIndex]
VIB_AMPLITUDE = VIB_AMPLITUDE_FRAC * boxSizeAlongVibrationAxis

vibrationStartIter = None
currentVibrationDisplacement = 0.0


def vibration_envelope(k):
    ramp_steps = min(VIBRATION_RAMP_STEPS, VIBRATION_STEPS // 2)
    if ramp_steps <= 0:
        return 1.0
    if k < ramp_steps:
        return k / ramp_steps
    if k > VIBRATION_STEPS - ramp_steps:
        return max(0.0, (VIBRATION_STEPS - k) / ramp_steps)
    return 1.0


def moveBoxSinusoidal():
    global currentVibrationDisplacement

    k = O.iter - vibrationStartIter
    omega_iter = 2.0 * np.pi / VIB_PERIOD_STEPS
    displacement = (
        VIB_AMPLITUDE
        * vibration_envelope(k)
        * np.sin(omega_iter * k)
    )
    velocity = (displacement - currentVibrationDisplacement) / O.dt
    currentVibrationDisplacement = displacement

    displacement_vector = Vector3(0, 0, 0)
    velocity_vector = Vector3(0, 0, 0)
    displacement_vector[axisIndex] = displacement
    velocity_vector[axisIndex] = velocity

    for bid in boxIds:
        body = O.bodies[bid]
        body.state.pos = boxInitialPos[bid] + displacement_vector
        body.state.vel = velocity_vector
        body.state.angVel = Vector3(0, 0, 0)


def stopBoxMotion():
    global currentVibrationDisplacement
    currentVibrationDisplacement = 0.0

    for bid in boxIds:
        body = O.bodies[bid]
        body.state.pos = Vector3(boxInitialPos[bid])
        body.state.vel = Vector3(0, 0, 0)
        body.state.angVel = Vector3(0, 0, 0)


def vibration_peak_acceleration():
    if not ENABLE_VIBRATION:
        return 0.0
    omega_physical = 2.0 * np.pi / (VIB_PERIOD_STEPS * O.dt)
    return VIB_AMPLITUDE * omega_physical ** 2


# ============================================================
# OPTIONAL FIXED-DISPLACEMENT COMPRESSION
# ============================================================

# Per-side-wall displacement. A value of 0.45 therefore reduces the total
# width by 45%, with each opposing wall moving inward by 22.5% of width.
COMPRESSION_TOTAL_DISPLACEMENT = (
    0.5
    * COMPRESSION_WIDTH_REDUCTION_FRAC
    * min(INITIAL_BOX_WIDTH_X, INITIAL_BOX_WIDTH_Y)
)

compressionStartIter = None
previousCompressionFraction = 0.0
currentCompressionDisplacement = 0.0


def pile_max_z():
    max_z = -float("inf")
    for body in O.bodies:
        if body and isinstance(body.shape, Sphere):
            max_z = max(max_z, body.state.pos[2] + body.shape.radius)
    if not np.isfinite(max_z):
        raise RuntimeError("Cannot place compression lid: no rock spheres exist.")
    return max_z


def add_lid_wall():
    global lidIds

    if lidIds:
        return lidIds

    pile_top = pile_max_z()

    if LID_HEIGHT_MODE == "above_pile":
        z_lid = pile_top + LID_CLEARANCE_FRAC_L * L
    else:
        z_lid = (
            initialBoxFloorZ
            + LID_HEIGHT_FRACTION * (initialBoxTopZ - initialBoxFloorZ)
        )
        if z_lid < pile_top:
            print(
                f"WARNING: fixed lid z={z_lid:.4g} is below pile top "
                f"z={pile_top:.4g}; the lid begins overlapped with rocks."
            )

    tri1 = [
        Vector3(initialBoxXMin, initialBoxYMin, z_lid),
        Vector3(initialBoxXMax, initialBoxYMin, z_lid),
        Vector3(initialBoxXMax, initialBoxYMax, z_lid),
    ]
    tri2 = [
        Vector3(initialBoxXMin, initialBoxYMin, z_lid),
        Vector3(initialBoxXMax, initialBoxYMax, z_lid),
        Vector3(initialBoxXMin, initialBoxYMax, z_lid),
    ]

    lidIds = O.bodies.append([
        utils.facet(tri1, material=matId_box),
        utils.facet(tri2, material=matId_box),
    ])

    for bid in lidIds:
        boxIds.append(bid)
        boxInitialPos[bid] = Vector3(O.bodies[bid].state.pos)
        wallAxisSign[bid] = (2, 1)
        compressWall[bid] = COMPRESS_TOP

    print(
        f"Added compression lid at z={z_lid:.4g} "
        f"(mode={LID_HEIGHT_MODE}, pile top={pile_top:.4g})."
    )
    return lidIds


def totalWallForceMagnitude():
    return sum(O.forces.f(bid).norm() for bid in boxIds)


def smoothstep(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def applyCompressionStep():
    global previousCompressionFraction, currentCompressionDisplacement

    k = O.iter - compressionStartIter
    fraction = smoothstep(k / COMPRESSION_STEPS)
    delta_fraction = fraction - previousCompressionFraction
    previousCompressionFraction = fraction

    step_displacement = COMPRESSION_TOTAL_DISPLACEMENT * delta_fraction
    currentCompressionDisplacement = COMPRESSION_TOTAL_DISPLACEMENT * fraction

    for bid in boxIds:
        if not compressWall.get(bid, False):
            continue

        axis, sign = wallAxisSign[bid]
        body = O.bodies[bid]
        position = Vector3(boxInitialPos[bid])
        position[axis] -= sign * currentCompressionDisplacement

        velocity = Vector3(0, 0, 0)
        velocity[axis] = -sign * step_displacement / O.dt

        body.state.pos = position
        body.state.vel = velocity
        body.state.angVel = Vector3(0, 0, 0)


def holdWallsAtFinalCompression():
    global currentCompressionDisplacement
    currentCompressionDisplacement = COMPRESSION_TOTAL_DISPLACEMENT

    for bid in boxIds:
        if not compressWall.get(bid, False):
            continue

        axis, sign = wallAxisSign[bid]
        body = O.bodies[bid]
        position = Vector3(boxInitialPos[bid])
        position[axis] -= sign * COMPRESSION_TOTAL_DISPLACEMENT

        body.state.pos = position
        body.state.vel = Vector3(0, 0, 0)
        body.state.angVel = Vector3(0, 0, 0)


# ============================================================
# FINAL DIAGNOSTICS AND METRICS
# ============================================================


def finalOverlapCheck():
    """Evaluate the current sphere-sphere overlap once at the end of the run."""
    # YADE's utility considers ScGeom interactions between two spheres; facet
    # contacts are therefore not included. For radii r1 and r2 it normalizes
    # penetration uN by 2*r1*r2/(r1+r2), which equals r for equal radii.
    sphere_sphere_ratio = float(utils.maxOverlapRatio())
    criterion_satisfied = (
        sphere_sphere_ratio <= MAXIMUM_ALLOWED_OVERLAP_FRACTION
    )

    print("\nFINAL SETTLED SPHERE-SPHERE OVERLAP CHECK")
    print(
        "YADE maximum overlap ratio =", sphere_sphere_ratio,
        "=", 100.0 * sphere_sphere_ratio, "%",
        "| limit =", 100.0 * MAXIMUM_ALLOWED_OVERLAP_FRACTION,
        "% | criterion satisfied =", criterion_satisfied,
    )

    return {
        "finalSphereSphereMaxOverlapRatio": sphere_sphere_ratio,
        "finalSphereSphereMaxOverlapPercent": 100.0 * sphere_sphere_ratio,
        "overlapMeasurementIteration": O.iter,
        "overlapMeasurementTime": O.time,
        "maximumAllowedOverlapFraction": MAXIMUM_ALLOWED_OVERLAP_FRACTION,
        "maximumAllowedOverlapPercent": (
            100.0 * MAXIMUM_ALLOWED_OVERLAP_FRACTION
        ),
        "overlapCriterionSatisfied": criterion_satisfied,
    }


def count_final_bodies():
    n_clumps = 0
    n_spheres = 0
    n_facets = 0

    for body in O.bodies:
        if not body:
            continue
        if body.isClump:
            n_clumps += 1
        elif isinstance(body.shape, Sphere):
            n_spheres += 1
        elif isinstance(body.shape, Facet):
            n_facets += 1

    return n_clumps, n_spheres, n_facets


def simulation_mode_name():
    if ENABLE_VIBRATION and ENABLE_COMPRESSION:
        return "vibration_then_compression"
    if ENABLE_VIBRATION:
        return "vibration"
    if ENABLE_COMPRESSION:
        return "compression"
    return "basic_random_gravity_deposition"


def engine_display_name(engine, index):
    """Human-readable unique name, including PyRunner command when available."""
    class_name = engine.__class__.__name__
    command = getattr(engine, "command", "")
    if command:
        return f"{index:02d}_{class_name}[{command}]"
    return f"{index:02d}_{class_name}"


def write_engine_timing():
    """Write cumulative YADE engine timing for this simulation run."""
    rows = []
    for index, engine in enumerate(O.engines):
        exec_time_ns = float(getattr(engine, "execTime", 0.0))
        exec_count = int(getattr(engine, "execCount", 0))
        total_seconds = exec_time_ns / 1.0e9
        average_ms = (exec_time_ns / exec_count / 1.0e6) if exec_count else np.nan

        rows.append({
            "run_id": RUN_ID,
            "engineIndex": index,
            "engine": engine_display_name(engine, index),
            "engineClass": engine.__class__.__name__,
            "execCount": exec_count,
            "totalTimeSeconds": total_seconds,
            "averageTimePerCallMs": average_ms,
        })

    total_engine_seconds = sum(row["totalTimeSeconds"] for row in rows)
    for row in rows:
        row["percentOfMeasuredEngineTime"] = (
            100.0 * row["totalTimeSeconds"] / total_engine_seconds
            if total_engine_seconds > 0 else np.nan
        )

    timing_df = pd.DataFrame(rows)
    timing_path = RUN_DIR / "engine_timing.csv"
    timing_df.to_csv(timing_path, index=False)

    print("\n========== YADE ENGINE TIMING ==========")
    for row in rows:
        print(
            f"{row['engine']}: "
            f"total={row['totalTimeSeconds']:.6f} s, "
            f"calls={row['execCount']}, "
            f"avg={row['averageTimePerCallMs']:.6f} ms/call, "
            f"share={row['percentOfMeasuredEngineTime']:.2f}%"
        )
    print(f"Measured engine total = {total_engine_seconds:.6f} s")
    print("Saved:", timing_path)

    return total_engine_seconds


def write_yade_metrics(
    final_unbalanced_force,
    overlap_metrics,
    finish_reason,
    equilibrium_reached,
):
    wall_clock_time = time.time() - wall_time_start
    n_clumps, n_spheres, n_facets = count_final_bodies()
    iterations_per_second = O.iter / wall_clock_time if wall_clock_time > 0 else np.nan

    final_kinetic_energy = utils.kineticEnergy()
    final_kinetic_energy_over_density = final_kinetic_energy / ROCK_DENSITY
    final_bounds = current_box_bounds()

    metrics = pd.DataFrame([{
        "run_id": RUN_ID,
        "description": str(table.description),
        "spawnSeed": SPAWN_SEED,
        "rockTypeSeed": ROCK_TYPE_SEED,
        "simulationMode": simulation_mode_name(),
        "spawnMode": SPAWN_MODE,
        "rockSelectionMode": ROCK_SELECTION_MODE,
        "contactModel": "Hertz-Mindlin",
        "wallClockTimeSeconds": wall_clock_time,
        "finalIteration": O.iter,
        "finalSimulationTime": O.time,
        "finishReason": finish_reason,
        "equilibriumReached": equilibrium_reached,
        "lastInsertionIteration": lastInsertionIter,
        "iterationsSinceLastInsertion": (
            O.iter - lastInsertionIter if lastInsertionIter is not None else np.nan
        ),
        "iterationsPerSecond": iterations_per_second,
        "finalUnbalancedForce": final_unbalanced_force,
        "finalKineticEnergy": final_kinetic_energy,
        "rockDensity": ROCK_DENSITY,
        "rockYoung": float(table.rockYoung),
        "rockPoisson": float(table.rockPoisson),
        "rockFrictionCoefficient": ROCK_FRICTION_COEFFICIENT,
        "rockFrictionAngleRadians": rockMat.frictionAngle,
        "rockFrictionAngleDegrees": math.degrees(rockMat.frictionAngle),
        "boxDensity": float(table.boxDensity),
        "boxYoung": float(table.boxYoung),
        "boxPoisson": float(table.boxPoisson),
        "boxFrictionCoefficient": BOX_FRICTION_COEFFICIENT,
        "boxFrictionAngleRadians": boxMat.frictionAngle,
        "boxFrictionAngleDegrees": math.degrees(boxMat.frictionAngle),
        "finalKineticEnergyOverDensity": final_kinetic_energy_over_density,
        "settleUbThreshold": SETTLE_UB_THRESHOLD,
        "finalUbThreshold": FINAL_UB_THRESHOLD,
        "settleHoldSteps": SETTLE_HOLD_STEPS,
        "minimumSettlingSteps": MINIMUM_SETTLING_STEPS,
        "maximumIterationsAfterLastInsertion": (
            MAXIMUM_ITERATIONS_AFTER_LAST_INSERTION
        ),
        "escapeBelowFloorMarginFracL": ESCAPE_BELOW_FLOOR_MARGIN_FRAC_L,
        "escapeBelowFloorMargin": ESCAPE_BELOW_FLOOR_MARGIN,
        "escapeCheckPeriod": ESCAPE_CHECK_PERIOD,
        "newtonDamping": NEWTON_DAMPING,
        "timestepSafety": TIMESTEP_SAFETY,
        "timeStep": O.dt,
        "insertionPeriodSeconds": INSERTION_PERIOD_SECONDS,
        "nRocksTarget": nRocks,
        "nRocksInserted": rockCounter,
        "nRocksErasedBelowFloor": deletedRockCount,
        "nRocksRetainedInYade": len(rockTypeByClumpId),
        "erasedClumpIds": ";".join(str(clump_id) for clump_id in deletedRockIds),
        "insertedRockTypeSequence": ";".join(
            str(rock_type) for rock_type in insertedRockTypeSequence
        ),
        "nClumps": n_clumps,
        "nSpheres": n_spheres,
        "nFacets": n_facets,
        "L": L,
        "boxHalfExtentXYFracL": BOX_HALF_EXTENT_XY_FRAC_L,
        "boxHalfExtentZFracL": BOX_HALF_EXTENT_Z_FRAC_L,
        "boxHalfExtentXY": BOX_HALF_EXTENT_XY,
        "boxHalfExtentZ": BOX_HALF_EXTENT_Z,
        "clumpingTimeSeconds": clumpingTimeSeconds,
        "clumpingCallCount": clumpingCallCount,
        "averageClumpingTimeSeconds": (
            clumpingTimeSeconds / clumpingCallCount if clumpingCallCount else np.nan
        ),
        "porosityBoundaryMarginFracL": POROSITY_BOUNDARY_MARGIN_FRAC_L,

        **overlap_metrics,

        "totalTemplateSphereCount": sum(len(pack) for pack in spherePacks),
        "meanTemplateSphereCount": float(np.mean([len(pack) for pack in spherePacks])),
        "rock1ResolutionDivisor": SELECTED_CLUMP_CONFIGURATIONS[1]["divisor"],
        "rock1OverlapFraction": SELECTED_CLUMP_CONFIGURATIONS[1]["overlapFraction"],
        "rock1LatticeRotation": SELECTED_CLUMP_CONFIGURATIONS[1]["rotation"],
        "rock1TemplateSphereCount": len(spherePacks[0]),
        "rock2ResolutionDivisor": SELECTED_CLUMP_CONFIGURATIONS[2]["divisor"],
        "rock2OverlapFraction": SELECTED_CLUMP_CONFIGURATIONS[2]["overlapFraction"],
        "rock2LatticeRotation": SELECTED_CLUMP_CONFIGURATIONS[2]["rotation"],
        "rock2TemplateSphereCount": len(spherePacks[1]),
        "rock3ResolutionDivisor": SELECTED_CLUMP_CONFIGURATIONS[3]["divisor"],
        "rock3OverlapFraction": SELECTED_CLUMP_CONFIGURATIONS[3]["overlapFraction"],
        "rock3LatticeRotation": SELECTED_CLUMP_CONFIGURATIONS[3]["rotation"],
        "rock3TemplateSphereCount": len(spherePacks[2]),
        "rock4ResolutionDivisor": SELECTED_CLUMP_CONFIGURATIONS[4]["divisor"],
        "rock4OverlapFraction": SELECTED_CLUMP_CONFIGURATIONS[4]["overlapFraction"],
        "rock4LatticeRotation": SELECTED_CLUMP_CONFIGURATIONS[4]["rotation"],
        "rock4TemplateSphereCount": len(spherePacks[3]),

        "enableVibration": ENABLE_VIBRATION,
        "vibrationAxis": VIBRATION_AXIS,
        "vibAmplitudeFrac": VIB_AMPLITUDE_FRAC,
        "vibAmplitude": VIB_AMPLITUDE,
        "vibPeriodSteps": VIB_PERIOD_STEPS,
        "vibrationSteps": VIBRATION_STEPS,
        "vibrationPeakAcceleration": vibration_peak_acceleration(),

        "enableCompression": ENABLE_COMPRESSION,
        "compressionWidthReductionFrac": COMPRESSION_WIDTH_REDUCTION_FRAC,
        "compressionTotalDisplacementPerWall": COMPRESSION_TOTAL_DISPLACEMENT,
        "compressionDisplacement": currentCompressionDisplacement,
        "compressionSteps": COMPRESSION_STEPS,
        "lidHeightMode": LID_HEIGHT_MODE,
        "compressFloor": COMPRESS_FLOOR,
        "compressTop": COMPRESS_TOP,
        "finalWallForceMagnitude": totalWallForceMagnitude(),

        "initialBoxXMin": initialBoxXMin,
        "initialBoxXMax": initialBoxXMax,
        "initialBoxYMin": initialBoxYMin,
        "initialBoxYMax": initialBoxYMax,
        "initialBoxFloorZ": initialBoxFloorZ,
        "initialBoxTopZ": initialBoxTopZ,
        # These unprefixed fields are deliberately FINAL bounds because the
        # STL contact/porosity post-processors consume them.
        **final_bounds,
    }])

    metrics.to_csv(RUN_DIR / "yade_metrics.csv", index=False)
    print("Saved: yade_metrics.csv")
    print("Final box bounds used by post-processing:", final_bounds)


# ============================================================
# COMBINED PHASE CONTROLLER
# ============================================================

simPhase = "initial_settling"
finished = False
settledConsecutiveSteps = 0
settlingPhaseStartIter = None


def resetSettlingCheck(phase_start_iter):
    global settledConsecutiveSteps, settlingPhaseStartIter
    settledConsecutiveSteps = 0
    settlingPhaseStartIter = phase_start_iter


def externalContactedClumps():
    """Return clump IDs currently touching another rock or a box facet."""
    contacted = set()

    for interaction in O.interactions:
        if not interaction.isReal:
            continue

        body1 = O.bodies[interaction.id1]
        body2 = O.bodies[interaction.id2]
        if not body1 or not body2:
            continue
        body1_is_sphere = isinstance(body1.shape, Sphere)
        body2_is_sphere = isinstance(body2.shape, Sphere)
        clump1 = body1.clumpId if body1_is_sphere else -1
        clump2 = body2.clumpId if body2_is_sphere else -1

        if clump1 >= 0:
            if isinstance(body2.shape, Facet) or (clump2 >= 0 and clump2 != clump1):
                contacted.add(clump1)
        if clump2 >= 0:
            if isinstance(body1.shape, Facet) or (clump1 >= 0 and clump1 != clump2):
                contacted.add(clump2)

    return contacted


def allRocksHaveExternalContact():
    inserted_clumps = set(rockTypeByClumpId.keys())
    return bool(inserted_clumps) and inserted_clumps.issubset(externalContactedClumps())


def unbalancedForceHasSettled(unbalanced_force, threshold):
    """Require minimum duration, current contacts, and sustained low residual."""
    global settledConsecutiveSteps

    if settlingPhaseStartIter is None:
        return False

    if O.iter - settlingPhaseStartIter < MINIMUM_SETTLING_STEPS:
        settledConsecutiveSteps = 0
        return False

    if not np.isfinite(unbalanced_force) or unbalanced_force > threshold:
        settledConsecutiveSteps = 0
        return False

    settledConsecutiveSteps += 1
    if settledConsecutiveSteps < SETTLE_HOLD_STEPS:
        return False

    # Check the more expensive contact condition only when the residual has
    # already remained low long enough to request a phase transition.
    if not allRocksHaveExternalContact():
        settledConsecutiveSteps = 0
        return False

    return True


def startVibration(unbalanced_force, kinetic_energy):
    global simPhase, vibrationStartIter, currentVibrationDisplacement
    simPhase = "vibrating"
    vibrationStartIter = O.iter
    currentVibrationDisplacement = 0.0

    print("\nSTARTING VIBRATION")
    print("Axis =", VIBRATION_AXIS)
    print("Amplitude =", VIB_AMPLITUDE)
    print("Peak acceleration estimate =", vibration_peak_acceleration(), "m/s^2")
    print("Start iteration =", O.iter)
    print("Unbalanced force before vibration =", unbalanced_force)
    print("Kinetic energy before vibration =", kinetic_energy, "J")


def startCompression(unbalanced_force, kinetic_energy):
    global simPhase, compressionStartIter, previousCompressionFraction

    stopBoxMotion()
    add_lid_wall()
    simPhase = "compressing"
    compressionStartIter = O.iter
    previousCompressionFraction = 0.0

    print("\nSTARTING COMPRESSION")
    print("Start iteration =", O.iter)
    print("Unbalanced force before compression =", unbalanced_force)
    print("Kinetic energy before compression =", kinetic_energy, "J")
    print("Total horizontal width reduction fraction =", COMPRESSION_WIDTH_REDUCTION_FRAC)
    print("Displacement per active wall =", COMPRESSION_TOTAL_DISPLACEMENT)
    print("Compressing floor =", COMPRESS_FLOOR)
    print("Compressing top =", COMPRESS_TOP)


def finishSimulation(
    final_unbalanced_force,
    finish_reason="equilibrium",
    equilibrium_reached=True,
):
    global finished

    if finished:
        return
    finished = True

    if equilibrium_reached:
        print("\nFINISHED: final equilibrium reached")
    else:
        print("\nFINISHED: post-insertion iteration allowance reached")
    print("Finish reason =", finish_reason)
    print("Simulation mode =", simulation_mode_name())
    print("Final iteration =", O.iter)
    print("Final simulation time =", O.time)
    print("Final unbalanced force =", final_unbalanced_force)
    print("Final kinetic energy =", utils.kineticEnergy(), "J")
    print("Final compression displacement per active wall =", currentCompressionDisplacement)

    overlap_metrics = finalOverlapCheck()
    exportClumpPoses()
    exportVTK()

    # Capture engine timing BEFORE launching external post-processing, so the
    # engine numbers represent YADE simulation work only.
    measured_engine_total = write_engine_timing()
    write_yade_metrics(
        final_unbalanced_force,
        overlap_metrics,
        finish_reason,
        equilibrium_reached,
    )

    poseFile.close()
    residualFile.close()
    deletedRockFile.close()

    reconstruct_seconds = np.nan
    porosity_seconds = np.nan
    post_processing_total_seconds = 0.0
    post_processing_succeeded = np.nan if ENABLE_POST_PROCESSING else True

    if ENABLE_POST_PROCESSING and DEFER_POST_PROCESSING_TO_RUNNER:
        print(
            "Post-processing deferred to the host yadepy stage for",
            RUN_DIR.name,
        )
    elif ENABLE_POST_PROCESSING:
        post_total_start = time.perf_counter()
        try:
            reconstruct_start = time.perf_counter()
            subprocess.run(
                ["python3", str(RECONSTRUCT_SCRIPT)],
                cwd=RUN_DIR,
                check=True,
            )
            reconstruct_seconds = time.perf_counter() - reconstruct_start

            porosity_start = time.perf_counter()
            subprocess.run(
                ["python3", str(POROSITY_SCRIPT)],
                cwd=RUN_DIR,
                check=True,
            )
            porosity_seconds = time.perf_counter() - porosity_start
            post_processing_succeeded = True
            print(f"Post-processing complete for {RUN_DIR.name}")
        except subprocess.CalledProcessError as error:
            print(f"POST-PROCESSING FAILED for {RUN_DIR.name}: {error}")
        finally:
            post_processing_total_seconds = time.perf_counter() - post_total_start

    timing_summary = pd.DataFrame([{
        "run_id": RUN_ID,
        "measuredEngineTotalSeconds": measured_engine_total,
        "clumpingTimeSeconds": clumpingTimeSeconds,
        "clumpingCallCount": clumpingCallCount,
        "averageClumpingTimeSeconds": (
            clumpingTimeSeconds / clumpingCallCount if clumpingCallCount else np.nan
        ),
        "reconstructSTLSeconds": reconstruct_seconds,
        "porositySeconds": porosity_seconds,
        "postProcessingTotalSeconds": post_processing_total_seconds,
        "postProcessingEnabled": ENABLE_POST_PROCESSING,
        "postProcessingDeferred": (
            ENABLE_POST_PROCESSING and DEFER_POST_PROCESSING_TO_RUNNER
        ),
        "postProcessingSucceeded": post_processing_succeeded,
    }])
    timing_summary_path = RUN_DIR / "timing_summary.csv"
    timing_summary.to_csv(timing_summary_path, index=False)

    print("\n========== RUN TIMING SUMMARY ==========")
    print(f"Clumping total = {clumpingTimeSeconds:.6f} s over {clumpingCallCount} clumps")
    print(f"Reconstruction = {reconstruct_seconds:.6f} s")
    print(f"Porosity = {porosity_seconds:.6f} s")
    print(f"Post-processing total = {post_processing_total_seconds:.6f} s")
    print("Saved:", timing_summary_path)

    O.pause()
    sys.exit(0)


def controlSimulation():
    global simPhase

    if finished:
        return

    unbalanced_force = unbalancedForce()
    kinetic_energy = utils.kineticEnergy()
    all_rocks_inserted = rockCounter >= nRocks

    # This fallback clock starts only when insertRock() records the final
    # insertion. It never interrupts the insertion schedule. Reaching it is a
    # completed run—not a failed run—and therefore follows the same export and
    # post-processing path as equilibrium-based completion.
    if (
        all_rocks_inserted
        and lastInsertionIter is not None
        and O.iter - lastInsertionIter >= MAXIMUM_ITERATIONS_AFTER_LAST_INSERTION
    ):
        finishSimulation(
            unbalanced_force,
            finish_reason="post_insertion_iteration_limit",
            equilibrium_reached=False,
        )
        return

    if simPhase == "initial_settling":
        if not all_rocks_inserted:
            resetSettlingCheck(None)
            return

        if settlingPhaseStartIter is None:
            resetSettlingCheck(lastInsertionIter)

        if not unbalancedForceHasSettled(unbalanced_force, SETTLE_UB_THRESHOLD):
            return

        print("\nINITIAL PACKING SETTLED")
        print("Iteration =", O.iter)
        print("Unbalanced force =", unbalanced_force)
        print("Kinetic energy (diagnostic only) =", kinetic_energy, "J")
        print("Required consecutive low-residual steps =", SETTLE_HOLD_STEPS)

        if ENABLE_VIBRATION:
            startVibration(unbalanced_force, kinetic_energy)
        elif ENABLE_COMPRESSION:
            startCompression(unbalanced_force, kinetic_energy)
        else:
            finishSimulation(unbalanced_force)
        return

    if simPhase == "vibrating":
        moveBoxSinusoidal()

        if O.iter - vibrationStartIter >= VIBRATION_STEPS:
            stopBoxMotion()
            simPhase = "settling_after_vibration"
            resetSettlingCheck(O.iter)
            print("\nSTOPPED VIBRATION")
            print("Stop iteration =", O.iter)
            print("Waiting for post-vibration settling...")
        return

    if simPhase == "settling_after_vibration":
        stopBoxMotion()

        target_threshold = (
            SETTLE_UB_THRESHOLD if ENABLE_COMPRESSION else FINAL_UB_THRESHOLD
        )
        if not unbalancedForceHasSettled(unbalanced_force, target_threshold):
            return

        print("\nPACKING SETTLED AFTER VIBRATION")
        print("Iteration =", O.iter)
        print("Unbalanced force =", unbalanced_force)
        print("Kinetic energy (diagnostic only) =", kinetic_energy, "J")
        print("Required consecutive low-residual steps =", SETTLE_HOLD_STEPS)

        if ENABLE_COMPRESSION:
            startCompression(unbalanced_force, kinetic_energy)
        else:
            finishSimulation(unbalanced_force)
        return

    if simPhase == "compressing":
        applyCompressionStep()

        if O.iter - compressionStartIter >= COMPRESSION_STEPS:
            holdWallsAtFinalCompression()
            simPhase = "final_settling"
            resetSettlingCheck(O.iter)
            print("\nCOMPRESSION RAMP COMPLETE")
            print("Stop iteration =", O.iter)
            print("Waiting for final settling...")
        return

    if simPhase == "final_settling":
        holdWallsAtFinalCompression()
        if unbalancedForceHasSettled(unbalanced_force, FINAL_UB_THRESHOLD):
            print("\nPACKING SETTLED AFTER COMPRESSION")
            print("Iteration =", O.iter)
            print("Unbalanced force =", unbalanced_force)
            print("Kinetic energy (diagnostic only) =", kinetic_energy, "J")
            finishSimulation(unbalanced_force)
        return

    raise RuntimeError(f"Unknown simulation phase: {simPhase}")


def addPlotData():
    unbalanced_force = unbalancedForce()
    kinetic_energy = utils.kineticEnergy()
    kinetic_energy_over_density = kinetic_energy / ROCK_DENSITY
    wall_force = totalWallForceMagnitude()

    if simPhase in {"initial_settling", "settling_after_vibration", "final_settling"}:
        if simPhase == "initial_settling" or ENABLE_COMPRESSION:
            active_threshold = SETTLE_UB_THRESHOLD
        else:
            active_threshold = FINAL_UB_THRESHOLD
        if simPhase == "final_settling":
            active_threshold = FINAL_UB_THRESHOLD

        elapsed = (
            O.iter - settlingPhaseStartIter
            if settlingPhaseStartIter is not None else 0
        )
        print(
            "SETTLING STATUS | phase =", simPhase,
            "| iteration =", O.iter,
            "| phase steps =", elapsed,
            "| unbalanced force =", unbalanced_force,
            "| threshold =", active_threshold,
            "| kinetic energy =", kinetic_energy, "J",
            "| contacted rocks =", len(externalContactedClumps()), "/",
            len(rockTypeByClumpId),
            "| inserted =", rockCounter,
            "| erased =", deletedRockCount,
            "| consecutive low-residual steps =",
            f"{settledConsecutiveSteps}/{SETTLE_HOLD_STEPS}",
        )

    residualFile.write(
        f"{O.iter},{O.time},{unbalanced_force},{kinetic_energy},"
        f"{kinetic_energy_over_density},{settledConsecutiveSteps},"
        f"{currentVibrationDisplacement},{currentCompressionDisplacement},"
        f"{wall_force},{simPhase}\n"
    )
    residualFile.flush()


O.run(wait=True)

print("YADE run finished.", flush=True)
sys.exit(0)
