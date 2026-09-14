"""YADE worker for one square-bed sphere case.

Do not launch this file directly.  ``run_parameter_study.py`` supplies one
case through environment variables and starts YADE in the correct run folder.
This worker intentionally imports neither pandas nor matplotlib.
"""

from __future__ import print_function

import csv
import json
import math
import os
import time

from yade import geom, pack


def env_text(name):
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError("Missing environment setting: " + name)
    return value


PACKING_TYPE = env_text("SPHERE_PACKING_TYPE").strip().lower()
NP_RATIO = float(env_text("SPHERE_NP_RATIO"))
REALIZATION = int(env_text("SPHERE_REALIZATION"))
RANDOM_SEED = int(env_text("SPHERE_RANDOM_SEED"))
SIDE = float(env_text("SPHERE_SQUARE_SIDE"))
TARGET_HEIGHT_D = float(env_text("SPHERE_TARGET_HEIGHT_D"))
MAX_SPHERES = int(env_text("SPHERE_MAX_COUNT"))

SPHERE_DENSITY = float(env_text("SPHERE_DENSITY"))
SPHERE_YOUNG = float(env_text("SPHERE_YOUNG"))
SPHERE_POISSON = float(env_text("SPHERE_POISSON"))
SPHERE_FRICTION = float(env_text("SPHERE_FRICTION"))
WALL_YOUNG = float(env_text("SPHERE_WALL_YOUNG"))
WALL_POISSON = float(env_text("SPHERE_WALL_POISSON"))
WALL_FRICTION = float(env_text("SPHERE_WALL_FRICTION"))
NEWTON_DAMPING = float(env_text("SPHERE_NEWTON_DAMPING"))
TIMESTEP_SAFETY = float(env_text("SPHERE_TIMESTEP_SAFETY"))
SETTLE_UB = float(env_text("SPHERE_SETTLE_UB"))
SETTLE_HOLD_CHECKS = int(env_text("SPHERE_SETTLE_HOLD_CHECKS"))
SETTLE_CHECK_PERIOD = int(env_text("SPHERE_SETTLE_CHECK_PERIOD"))
MINIMUM_STEPS = int(env_text("SPHERE_MIN_STEPS"))
MAXIMUM_STEPS = int(env_text("SPHERE_MAX_STEPS"))

if PACKING_TYPE not in ("hexa", "ortho", "random"):
    raise ValueError("Unknown SPHERE_PACKING_TYPE: " + PACKING_TYPE)
if NP_RATIO < 2.0:
    raise ValueError("Np must be at least 2")
if MAX_SPHERES < 1:
    raise ValueError("SPHERE_MAX_COUNT must be at least 1")

DIAMETER = SIDE / NP_RATIO
RADIUS = 0.5 * DIAMETER
TARGET_HEIGHT = TARGET_HEIGHT_D * DIAMETER
SPHERE_VOLUME = (4.0 / 3.0) * math.pi * RADIUS ** 3
STARTED_AT = time.time()

O.reset()
O.timingEnabled = True


def write_particles():
    rows = []
    for body in O.bodies:
        if body is None or not isinstance(body.shape, Sphere):
            continue
        rows.append((
            int(body.id),
            float(body.state.pos[0]),
            float(body.state.pos[1]),
            float(body.state.pos[2]),
            float(body.shape.radius),
        ))
    rows.sort(key=lambda row: row[0])
    with open("particles.csv", "w") as handle:
        writer = csv.writer(handle)
        writer.writerow(("particleId", "x", "y", "z", "radius"))
        writer.writerows(rows)
    return rows


def write_summary(status, settled, extra=None):
    rows = write_particles()
    top = max((row[3] + row[4] for row in rows), default=float("nan"))
    bottom = min((row[3] - row[4] for row in rows), default=float("nan"))
    payload = {
        "packingType": PACKING_TYPE,
        "npRatio": NP_RATIO,
        "realization": REALIZATION,
        "randomSeed": RANDOM_SEED,
        "squareSideLength": SIDE,
        "sphereDiameter": DIAMETER,
        "sphereRadius": RADIUS,
        "targetBedHeightDiameters": TARGET_HEIGHT_D,
        "targetBedHeight": TARGET_HEIGHT,
        "maximumSpheresPerCase": MAX_SPHERES,
        "numberOfSpheres": len(rows),
        "bedBottom": bottom,
        "bedTop": top,
        "iteration": int(O.iter),
        "simulationTime": float(O.time),
        "wallClockSeconds": time.time() - STARTED_AT,
        "status": status,
        "settled": bool(settled),
        "material": {
            "sphereDensity": SPHERE_DENSITY,
            "sphereYoung": SPHERE_YOUNG,
            "spherePoisson": SPHERE_POISSON,
            "sphereFrictionCoefficient": SPHERE_FRICTION,
            "wallYoung": WALL_YOUNG,
            "wallPoisson": WALL_POISSON,
            "wallFrictionCoefficient": WALL_FRICTION,
            "contactLaw": "Hertz-Mindlin",
        },
    }
    if extra:
        payload.update(extra)
    with open("simulation_summary.json", "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if PACKING_TYPE in ("hexa", "ortho"):
    # These are exact, stress-free geometric constructions.  Repeating them
    # with multiple seeds would only duplicate identical coordinates.
    predicate = pack.inAlignedBox(
        (0.0, 0.0, 0.0),
        (SIDE, SIDE, TARGET_HEIGHT),
    )
    if PACKING_TYPE == "hexa":
        bodies = pack.regularHexa(predicate, radius=RADIUS, gap=0.0)
    else:
        bodies = pack.regularOrtho(predicate, radius=RADIUS, gap=0.0)

    # Retain complete horizontal layers from the floor upward.  This gives a
    # flat deterministic bed while guaranteeing the same hard count limit used
    # for random cases.  The first layer is always much smaller than the cap for
    # the configured 3 <= Np <= 10 range.
    layers = []
    layer_tolerance = 1.0e-8 * DIAMETER
    for body in sorted(bodies, key=lambda item: float(item.state.pos[2])):
        z_position = float(body.state.pos[2])
        if not layers or abs(z_position - layers[-1][0]) > layer_tolerance:
            layers.append((z_position, [body]))
        else:
            layers[-1][1].append(body)
    selected_bodies = []
    selected_layer_count = 0
    for _, layer in layers:
        if len(selected_bodies) + len(layer) > MAX_SPHERES:
            break
        selected_bodies.extend(layer)
        selected_layer_count += 1
    if not selected_bodies:
        raise RuntimeError(
            "Sphere cap %d is smaller than the first regular layer (%d spheres)" %
            (MAX_SPHERES, len(layers[0][1]))
        )
    O.bodies.append(selected_bodies)
    write_summary(
        "deterministic_geometry",
        True,
        {
            "gravityUsed": False,
            "construction": (
                "yade.pack.regularHexa" if PACKING_TYPE == "hexa"
                else "yade.pack.regularOrtho"
            ),
            "finalUnbalancedForce": 0.0,
            "finalMaximumSpeed": 0.0,
            "uncappedNumberOfSpheres": len(bodies),
            "capApplied": len(selected_bodies) < len(bodies),
            "numberOfCompleteLayers": selected_layer_count,
        },
    )
    print(
        "Created", PACKING_TYPE, "packing at Np=", NP_RATIO,
        "with", len([b for b in O.bodies if isinstance(b.shape, Sphere)]),
        "spheres"
    )
else:
    sphere_material = FrictMat(
        young=SPHERE_YOUNG,
        poisson=SPHERE_POISSON,
        frictionAngle=math.atan(SPHERE_FRICTION),
        density=SPHERE_DENSITY,
        label="sphereMaterial",
    )
    wall_material = FrictMat(
        young=WALL_YOUNG,
        poisson=WALL_POISSON,
        frictionAngle=math.atan(WALL_FRICTION),
        density=SPHERE_DENSITY,
        label="wallMaterial",
    )
    O.materials.append(sphere_material)
    O.materials.append(wall_material)

    # Dixon is used only to estimate how many spheres are needed for a bed of
    # roughly TARGET_HEIGHT_D diameters.  Measured porosity is never forced to
    # equal the correlation.
    d_over_w = 1.0 / NP_RATIO
    dixon_porosity = 0.4 + 0.05 * d_over_w + 0.412 * d_over_w ** 2
    target_solid_volume = (
        (1.0 - dixon_porosity) * SIDE ** 2 * TARGET_HEIGHT
    )
    uncapped_target_number = int(
        math.ceil(1.08 * target_solid_volume / SPHERE_VOLUME)
    )
    target_number = min(uncapped_target_number, MAX_SPHERES)
    effective_target_height = min(
        TARGET_HEIGHT,
        target_number * SPHERE_VOLUME /
        (1.08 * (1.0 - dixon_porosity) * SIDE ** 2),
    )

    # An initially loose, non-overlapping monodisperse cloud is released under
    # gravity.  The cloud begins at the floor and is taller than the final bed,
    # matching a dumped/deposited random packing without sequential insertion.
    initial_cloud_solid_fraction = 0.22
    base_cloud_height = max(
        1.35 * effective_target_height,
        target_number * SPHERE_VOLUME /
        (initial_cloud_solid_fraction * SIDE ** 2),
    )

    # Narrow beds, especially Np=3, can occasionally reach makeCloud's
    # insertion-attempt limit for an unlucky seed.  Retry deterministically in
    # a progressively taller box instead of failing the complete study.
    maximum_cloud_attempts = 5
    cloud = None
    created = 0
    cloud_height = base_cloud_height
    cloud_attempts_used = 0
    for cloud_attempt in range(maximum_cloud_attempts):
        cloud_attempts_used = cloud_attempt + 1
        cloud_height = base_cloud_height * (1.25 ** cloud_attempt)
        candidate_cloud = pack.SpherePack()
        created = candidate_cloud.makeCloud(
            (0.0, 0.0, 0.0),
            (SIDE, SIDE, cloud_height),
            rMean=RADIUS,
            rRelFuzz=0.0,
            num=target_number,
            periodic=False,
            seed=RANDOM_SEED,
        )
        if created == target_number:
            cloud = candidate_cloud
            break
        print(
            "makeCloud retry:",
            "attempt=", cloud_attempts_used,
            "created=", created,
            "requested=", target_number,
            "nextHeightFactor=", 1.25 ** (cloud_attempt + 1),
        )
    if cloud is None:
        raise RuntimeError(
            "makeCloud created only %d of %d requested spheres after %d attempts" %
            (created, target_number, maximum_cloud_attempts)
        )

    wall_height = cloud_height + 4.0 * DIAMETER
    walls = geom.facetBox(
        (0.5 * SIDE, 0.5 * SIDE, 0.5 * wall_height),
        (0.5 * SIDE, 0.5 * SIDE, 0.5 * wall_height),
        wallMask=31,
        material=wall_material,
    )
    O.bodies.append(walls)
    cloud.toSimulation(material=sphere_material)

    O.engines = [
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
            [Ip2_FrictMat_FrictMat_MindlinPhys()],
            [Law2_ScGeom_MindlinPhys_Mindlin()],
        ),
        NewtonIntegrator(
            gravity=(0.0, 0.0, -9.81),
            damping=NEWTON_DAMPING,
        ),
        PyRunner(command="check_settling()", iterPeriod=SETTLE_CHECK_PERIOD),
    ]
    O.dt = TIMESTEP_SAFETY * PWaveTimeStep()

    consecutive_settled_checks = 0


    def maximum_sphere_speed():
        maximum = 0.0
        for body in O.bodies:
            if body is None or not isinstance(body.shape, Sphere):
                continue
            speed = body.state.vel.norm()
            if speed > maximum:
                maximum = speed
        return maximum


    def finish_random(status, settled, ub, maximum_speed):
        write_summary(
            status,
            settled,
            {
                "gravityUsed": True,
                "construction": "monodisperse makeCloud followed by gravity settling",
                "requestedNumberOfSpheres": target_number,
                "uncappedTargetNumberOfSpheres": uncapped_target_number,
                "capApplied": target_number < uncapped_target_number,
                "effectiveTargetBedHeight": effective_target_height,
                "effectiveTargetBedHeightDiameters": (
                    effective_target_height / DIAMETER
                ),
                "initialCloudHeight": cloud_height,
                "cloudInsertionAttempts": cloud_attempts_used,
                "wallHeight": wall_height,
                "finalUnbalancedForce": ub,
                "finalMaximumSpeed": maximum_speed,
                "settleUnbalancedForceThreshold": SETTLE_UB,
                "settleHoldChecks": SETTLE_HOLD_CHECKS,
                "settleCheckPeriod": SETTLE_CHECK_PERIOD,
            },
        )
        print(
            "Finished random packing:", status, "iteration=", O.iter,
            "unbalancedForce=", ub, "maxSpeed=", maximum_speed
        )
        O.pause()


    def check_settling():
        global consecutive_settled_checks
        ub = unbalancedForce()
        maximum_speed = maximum_sphere_speed()
        velocity_scale = math.sqrt(9.81 * DIAMETER)
        speed_threshold = 1.0e-4 * velocity_scale

        if O.iter % (20 * SETTLE_CHECK_PERIOD) == 0:
            print(
                "progress iter=", O.iter, "time=", O.time,
                "ub=", ub, "maxSpeed=", maximum_speed,
                "hold=", consecutive_settled_checks,
            )

        if O.iter >= MAXIMUM_STEPS:
            finish_random("maximum_steps_reached", False, ub, maximum_speed)
            return
        if O.iter < MINIMUM_STEPS:
            consecutive_settled_checks = 0
            return
        if (
            math.isfinite(ub) and ub <= SETTLE_UB and
            maximum_speed <= speed_threshold
        ):
            consecutive_settled_checks += 1
        else:
            consecutive_settled_checks = 0
        if consecutive_settled_checks >= SETTLE_HOLD_CHECKS:
            finish_random("settled", True, ub, maximum_speed)


    print(
        "Starting random square bed:",
        "Np=", NP_RATIO,
        "seed=", RANDOM_SEED,
        "spheres=", target_number,
        "cloudHeight=", cloud_height,
        "dt=", O.dt,
    )
    O.run(wait=True)
