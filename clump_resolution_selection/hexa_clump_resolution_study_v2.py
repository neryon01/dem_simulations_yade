"""
UPDATED SCRIPT — 2026-09-07 — DETERMINISTIC HEXAGONAL FIDELITY STUDY

Each GTS rock is filled with regular-hexagonal arrangements of overlapping
spheres for every combination of resolution divisor, overlap fraction and
fixed lattice orientation. The study evaluates spatial IoU, GTS coverage,
clump containment, independent volume agreement, and member-sphere count.
Every divisor-overlap-rotation combination is generated exactly once; there
are no random packing realisations.

The following descriptors are calculated only for the original reference mesh
(the GTS representation of the STL), never for the sphere clumps:
    sphericity  psi = (36*pi*V**2)**(1/3) / A
    convexity   c   = V_particle / V_convex_hull    (Suhr and Six, 2020)

The GTS predicate uses centre-based clipping, rather than requiring each sphere
to remain fully inside the surface. This avoids erosion of the represented rock
by one member radius. Candidates must satisfy both the sphere-count limit and
the independent-volume-ratio interval. Eligible candidates are ranked by
spatial IoU, with member-sphere count and volume error used as tie-breakers.
Consequently, different rocks may receive different divisors, overlaps and
orientations.

Run with:
    yade hexa_clump_resolution_study_v2.py
"""

from __future__ import print_function

import csv
import math
from collections import defaultdict, deque

import gts
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from yade import Vector3, export, pack


# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------

GTS_FILES = [
    "rock_1.gts",
    "rock_2.gts",
    "rock_3.gts",
    "rock_4.gts",
]

# Same expanded range used in the hexagonal/orthogonal comparison study.
RESOLUTION_DIVISORS = list(range(6, 19))
OVERLAP_DEPTH_FRACTIONS_OF_RADIUS = [0.10, 0.30, 0.50, 0.70]

# The GTS surface is rotated relative to YADE's fixed regularHexa lattice.
# Generated sphere centres are subsequently rotated back into the original
# GTS coordinate system. Angles are specified in degrees for readability.
LATTICE_ROTATIONS = [
    ("identity", (1.0, 0.0, 0.0), 0.0),
    ("x_30deg", (1.0, 0.0, 0.0), 30.0),
    ("y_30deg", (0.0, 1.0, 0.0), 30.0),
    ("z_30deg", (0.0, 0.0, 1.0), 30.0),
]

MAX_SPHERES_PER_ROCK = 150
MIN_VOLUME_COVERAGE_RATIO = 0.90
MAX_VOLUME_COVERAGE_RATIO = 1.10
TOP_CANDIDATES_TO_REPORT = 3

# Example: 0.10 means overlap depth = 0.10*r = 0.05*sphere diameter.
DISPLAY_SPACING = 2.0

OUTPUT_SCENE = "center_clipped_selected_clumps.yade.gz"
VTK_OUTPUT_PREFIX = "center_clipped_selected_clumps"
OUTPUT_CSV = "center_clipped_clump_results.csv"
OUTPUT_RANKING_CSV = "center_clipped_clump_rankings.csv"
OUTPUT_SELECTION_CSV = "center_clipped_clump_selections.csv"
OUTPUT_SELECTION_HEATMAP = "center_clipped_selection_heatmaps.png"
OUTPUT_SELECTION_SUMMARY = "center_clipped_selected_candidate_summary.png"
OUTPUT_COUNT_VS_DIVISOR_PLOT = "center_clipped_sphere_count_vs_divisor.png"
OUTPUT_VOLUME_PLOT = "center_clipped_independent_volume_ratio.png"

# Used to evaluate the geometric union volume of overlapping member spheres.
OVERLAP_VOLUME_SAMPLES = 2 ** 18
SPHERE_OVERLAP_REL_TOL = 1.0e-8

# One fixed point set is shared by every candidate of the same rock. This is
# numerical integration for spatial metrics, not repeated clump generation.
SPATIAL_METRIC_SAMPLES = 2 ** 15


# -----------------------------------------------------------------------------
# GTS REFERENCE GEOMETRY
# -----------------------------------------------------------------------------

def read_gts_mesh(path):
    """Return vertices and triangular faces from a GTS file."""
    with open(path, "r") as handle:
        lines = [
            line.strip() for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]

    n_vertices, n_edges, n_faces = map(int, lines[0].split()[:3])
    vertices = np.asarray([
        list(map(float, lines[1 + index].split()[:3]))
        for index in range(n_vertices)
    ])

    edge_start = 1 + n_vertices
    edges = []
    for index in range(n_edges):
        first, second = map(int, lines[edge_start + index].split()[:2])
        edges.append((first - 1, second - 1))

    face_start = edge_start + n_edges
    faces = []
    for index in range(n_faces):
        edge_ids = [
            int(value) - 1
            for value in lines[face_start + index].split()[:3]
        ]
        vertex_ids = set()
        for edge_id in edge_ids:
            vertex_ids.update(edges[edge_id])
        if len(vertex_ids) != 3:
            raise ValueError("Invalid triangular face in {}.".format(path))
        faces.append(tuple(sorted(vertex_ids)))

    return vertices, np.asarray(faces, dtype=int)


def orient_closed_faces(vertices, faces):
    """Give every connected triangle component a consistent outward direction."""
    edge_uses = defaultdict(list)
    for face_index, face in enumerate(faces):
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = (min(first, second), max(first, second))
            direction = 1 if (first, second) == edge else -1
            edge_uses[edge].append((face_index, direction))

    if any(len(uses) != 2 for uses in edge_uses.values()):
        raise ValueError("The GTS reference surface is not closed and manifold.")

    adjacency = defaultdict(list)
    for uses in edge_uses.values():
        (face_a, direction_a), (face_b, direction_b) = uses
        same_direction = direction_a == direction_b
        adjacency[face_a].append((face_b, same_direction))
        adjacency[face_b].append((face_a, same_direction))

    visited = np.zeros(len(faces), dtype=bool)
    flip = np.zeros(len(faces), dtype=bool)
    components = []

    for start in range(len(faces)):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor, same_direction in adjacency[current]:
                required = (not flip[current]) if same_direction else flip[current]
                if not visited[neighbor]:
                    visited[neighbor] = True
                    flip[neighbor] = required
                    queue.append(neighbor)
                elif flip[neighbor] != required:
                    raise ValueError("Inconsistent GTS surface orientation.")
        components.append(component)

    oriented = faces.copy()
    oriented[flip, 1], oriented[flip, 2] = (
        oriented[flip, 2].copy(), oriented[flip, 1].copy()
    )

    for component in components:
        component = np.asarray(component, dtype=int)
        triangles = vertices[oriented[component]]
        signed_volume = np.sum(
            np.einsum(
                "ij,ij->i",
                triangles[:, 0],
                np.cross(triangles[:, 1], triangles[:, 2]),
            )
        ) / 6.0
        if signed_volume < 0.0:
            oriented[component, 1], oriented[component, 2] = (
                oriented[component, 2].copy(),
                oriented[component, 1].copy(),
            )

    return oriented


# -----------------------------------------------------------------------------
# REFERENCE-MESH SPHERICITY AND CONVEXITY
# -----------------------------------------------------------------------------

def descriptor_values(volume, area, hull_volume):
    """Calculate descriptors for the original reference mesh only."""
    return {
        "volume": volume,
        "surfaceArea": area,
        "convexHullVolume": hull_volume,
        "sphericity": (36.0 * math.pi * volume ** 2) ** (1.0 / 3.0) / area,
        "convexity": volume / hull_volume,
    }


def mesh_metrics(vertices, faces):
    faces = orient_closed_faces(vertices, faces)
    triangles = vertices[faces]
    cross_products = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    area = 0.5 * float(np.sum(np.linalg.norm(cross_products, axis=1)))
    volume = float(np.sum(
        np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2]),
        )
    ) / 6.0)
    hull_volume = float(ConvexHull(vertices).volume)
    return descriptor_values(volume, area, hull_volume)


# -----------------------------------------------------------------------------
# CLUMP UNION GEOMETRY
# -----------------------------------------------------------------------------

def overlapping_union_volume(centers, radius):
    lower = np.min(centers, axis=0) - radius
    upper = np.max(centers, axis=0) + radius
    rng = np.random.RandomState(0)
    unit_points = rng.random_sample((OVERLAP_VOLUME_SAMPLES, 3))
    points = lower + unit_points * (upper - lower)
    distances, _ = cKDTree(centers).query(points, k=1)
    return float(np.prod(upper - lower)) * np.mean(distances <= radius)


def clump_metrics(member_bodies):
    centers = np.asarray([
        [
            float(member.state.pos[0]),
            float(member.state.pos[1]),
            float(member.state.pos[2]),
        ]
        for member in member_bodies
    ])
    radii = np.asarray([
        float(member.shape.radius) for member in member_bodies
    ])

    if not np.allclose(radii, radii[0], rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("Shape analysis currently requires equal member radii.")

    radius = float(radii[0])
    overlap_distance = 2.0 * radius * (1.0 - SPHERE_OVERLAP_REL_TOL)
    has_overlap = bool(cKDTree(centers).query_pairs(overlap_distance))

    if has_overlap:
        volume = overlapping_union_volume(centers, radius)
        union_method = "deterministic_numerical_overlap_union"
    else:
        volume = len(centers) * (4.0 / 3.0) * math.pi * radius ** 3
        union_method = "exact_nonoverlapping_sphere_union"

    return {
        "volume": volume,
        "hasSphereOverlap": int(has_overlap),
        "unionMethod": union_method,
    }


def relative_error_percent(approximation, reference):
    return 100.0 * (approximation - reference) / reference


def make_shared_spatial_sample(rock, seed):
    """Create one reproducible integration point set for a GTS rock."""
    maximum_radius = rock["dx"] / float(min(RESOLUTION_DIVISORS))
    lower = rock["lower"] - maximum_radius
    upper = rock["upper"] + maximum_radius
    rng = np.random.RandomState(seed)
    points = lower + rng.random_sample((SPATIAL_METRIC_SAMPLES, 3)) * (
        upper - lower
    )

    with open(rock["gtsFile"], "r") as handle:
        surface = gts.read(handle)
    predicate = pack.inGtsSurface(surface, True)
    inside_gts = np.fromiter(
        (predicate(Vector3(*point), 0.0) for point in points),
        dtype=bool,
        count=len(points),
    )
    return {
        "points": points,
        "insideGts": inside_gts,
        "boxVolume": float(np.prod(upper - lower)),
    }


def spatial_overlap_metrics(sample, centers, radius):
    """Estimate IoU, GTS coverage and clump containment on shared points."""
    distances, _ = cKDTree(centers).query(sample["points"], k=1)
    inside_clump = distances <= radius
    inside_gts = sample["insideGts"]
    intersection = inside_gts & inside_clump
    union = inside_gts | inside_clump

    n_gts = int(np.count_nonzero(inside_gts))
    n_clump = int(np.count_nonzero(inside_clump))
    n_intersection = int(np.count_nonzero(intersection))
    n_union = int(np.count_nonzero(union))
    if min(n_gts, n_clump, n_union) == 0:
        raise RuntimeError("Spatial sample did not resolve both geometries.")

    point_volume = sample["boxVolume"] / float(len(sample["points"]))
    return {
        "spatialIoU": n_intersection / float(n_union),
        "gtsCoverage": n_intersection / float(n_gts),
        "clumpContainment": n_intersection / float(n_clump),
        "spatialGtsVolumeEstimate": n_gts * point_volume,
        "spatialClumpVolumeEstimate": n_clump * point_volume,
        "spatialIntersectionVolumeEstimate": n_intersection * point_volume,
    }


# -----------------------------------------------------------------------------
# GENERATE CLUMPS AND CALCULATE RESULTS
# -----------------------------------------------------------------------------

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


def generate_lattice_members(
    rock,
    resolution_divisor,
    overlap_fraction,
    rotation_setting,
    material_id,
    color,
):
    """Fill a rotated GTS and map the resulting lattice back to the GTS frame."""
    rotation_label, rotation_axis, angle_degrees = rotation_setting
    with open(rock["gtsFile"], "r") as handle:
        surface = gts.read(handle)

    center = rock["center"]
    if angle_degrees:
        surface.translate(-center[0], -center[1], -center[2])
        surface.rotate(
            rotation_axis[0],
            rotation_axis[1],
            rotation_axis[2],
            math.radians(angle_degrees),
        )
        surface.translate(center[0], center[1], center[2])

    # regularHexa passes radius as predicate padding. With the default GTS
    # predicate this keeps every sphere fully inside the mesh and erodes the
    # represented particle by approximately one radius. noPad=True instead
    # admits lattice centres inside the GTS; boundary spheres may protrude, and
    # the resulting over/under-coverage is measured explicitly below.
    predicate = pack.inGtsSurface(surface, True)  # True means noPad
    member_radius = rock["dx"] / float(resolution_divisor)
    member_gap = -overlap_fraction * member_radius
    member_bodies = pack.regularHexa(
        predicate,
        radius=member_radius,
        gap=member_gap,
        material=material_id,
        color=color,
    )

    # Rotating the surface changes the lattice-to-rock orientation. Mapping the
    # centres back restores the original GTS orientation used by the main model.
    if angle_degrees and member_bodies:
        inverse_rotation = rotation_matrix(
            rotation_axis, angle_degrees
        ).T
        for member in member_bodies:
            position = np.asarray([
                float(member.state.pos[0]),
                float(member.state.pos[1]),
                float(member.state.pos[2]),
            ])
            mapped = center + inverse_rotation.dot(position - center)
            member.state.pos = Vector3(*mapped)

    return member_bodies, member_radius, member_gap

O.reset()

material_id = O.materials.append(
    FrictMat(
        young=1.0e7,
        poisson=0.25,
        frictionAngle=0.0,
        density=2600.0,
    )
)

colors = [
    (0.85, 0.25, 0.25),
    (0.25, 0.65, 0.90),
    (0.35, 0.80, 0.35),
    (0.90, 0.70, 0.20),
]

result_rows = []
rock_data = []

print("PER-ROCK RESOLUTION–OVERLAP–ROTATION STUDY")
print("Resolution divisors:", RESOLUTION_DIVISORS)
print(
    "Overlap-depth fractions of radius:",
    OVERLAP_DEPTH_FRACTIONS_OF_RADIUS,
)
print("Maximum spheres per rock:", MAX_SPHERES_PER_ROCK)
print(
    "Admissible volume coverage: {:.0f}% to {:.0f}%".format(
        100.0 * MIN_VOLUME_COVERAGE_RATIO,
        100.0 * MAX_VOLUME_COVERAGE_RATIO,
    )
)
print("GTS clipping: sphere centres inside (noPad=True)")
print(
    "Lattice rotations:",
    [rotation[0] for rotation in LATTICE_ROTATIONS],
)

# Read each GTS surface and calculate its reference values only once.
for rock_index, gts_file in enumerate(GTS_FILES):
    vertices, faces = read_gts_mesh(gts_file)
    lower = np.min(vertices, axis=0)
    upper = np.max(vertices, axis=0)
    dx = float(upper[0] - lower[0])
    reference = mesh_metrics(vertices, faces)
    rock = {
        "rockType": rock_index + 1,
        "gtsFile": gts_file,
        "dx": dx,
        "lower": lower,
        "upper": upper,
        "center": 0.5 * (lower + upper),
        "reference": reference,
    }
    rock["spatialSample"] = make_shared_spatial_sample(
        rock, 91000 + rock_index
    )
    rock_data.append(rock)

# Evaluate every divisor–overlap–rotation candidate separately for each rock.
for rock_index, rock in enumerate(rock_data):
    print("\n" + rock["gtsFile"])
    for resolution_divisor in RESOLUTION_DIVISORS:
        for overlap_fraction in OVERLAP_DEPTH_FRACTIONS_OF_RADIUS:
            for rotation_setting in LATTICE_ROTATIONS:
                rotation_label, rotation_axis, angle_degrees = rotation_setting
                member_bodies, member_radius, member_gap = (
                    generate_lattice_members(
                        rock,
                        resolution_divisor,
                        overlap_fraction,
                        rotation_setting,
                        material_id,
                        colors[rock_index],
                    )
                )

                if len(member_bodies) == 0:
                    print(
                        "  skipped empty candidate: divisor={}, overlap="
                        "{:.2f}r, rotation={}".format(
                            resolution_divisor,
                            overlap_fraction,
                            rotation_label,
                        )
                    )
                    continue

                clump = clump_metrics(member_bodies)
                centers = np.asarray([
                    [
                        float(member.state.pos[0]),
                        float(member.state.pos[1]),
                        float(member.state.pos[2]),
                    ]
                    for member in member_bodies
                ])
                spatial = spatial_overlap_metrics(
                    rock["spatialSample"], centers, member_radius
                )
                reference = rock["reference"]
                volume_coverage = clump["volume"] / reference["volume"]
                within_sphere_limit = (
                    len(member_bodies) <= MAX_SPHERES_PER_ROCK
                )
                within_volume_interval = (
                    MIN_VOLUME_COVERAGE_RATIO
                    <= volume_coverage
                    <= MAX_VOLUME_COVERAGE_RATIO
                )
                row = {
                    "rockType": rock["rockType"],
                    "gtsFile": rock["gtsFile"],
                    "resolutionDivisor": resolution_divisor,
                    "overlapDepthFractionOfRadius": overlap_fraction,
                    "rotationLabel": rotation_label,
                    "rotationAxisX": rotation_axis[0],
                    "rotationAxisY": rotation_axis[1],
                    "rotationAxisZ": rotation_axis[2],
                    "rotationAngleDegrees": angle_degrees,
                    "memberRadius": member_radius,
                    "gap": member_gap,
                    "overlapDepth": -member_gap,
                    "nSpheres": len(member_bodies),
                    "withinSphereLimit": int(within_sphere_limit),
                    "hasSphereOverlap": clump["hasSphereOverlap"],
                    "unionMethod": clump["unionMethod"],
                    "reference_volume": reference["volume"],
                    "clump_volume": clump["volume"],
                    "volumeCoverageRatio": volume_coverage,
                    "volumeRelativeErrorPercent": relative_error_percent(
                        clump["volume"], reference["volume"]
                    ),
                    "volumeAbsoluteRelativeErrorPercent": abs(
                        relative_error_percent(
                            clump["volume"], reference["volume"]
                        )
                    ),
                    "withinVolumeCoverageInterval": int(
                        within_volume_interval
                    ),
                    "isEligible": int(
                        within_sphere_limit and within_volume_interval
                    ),
                    "reference_surfaceArea": reference["surfaceArea"],
                    "reference_convexHullVolume": reference["convexHullVolume"],
                    "reference_sphericity": reference["sphericity"],
                    "reference_convexity": reference["convexity"],
                }
                row.update(spatial)
                result_rows.append(row)
                print(
                    "  d={}, overlap={:.2f}r, rotation={}: spheres={}, "
                    "coverage={:.1f}%, IoU={:.1f}%, GTS coverage={:.1f}%, "
                    "containment={:.1f}%, eligible={}".format(
                        resolution_divisor,
                        overlap_fraction,
                        rotation_label,
                        len(member_bodies),
                        100.0 * volume_coverage,
                        100.0 * row["spatialIoU"],
                        100.0 * row["gtsCoverage"],
                        100.0 * row["clumpContainment"],
                        row["isEligible"],
                    )
                )


# Rank eligible candidates separately for each rock. Spatial agreement is the
# primary selection criterion. Sphere count and independent-volume error are
# deterministic tie-breakers.
best_by_rock = {}
for rock in rock_data:
    candidates = [
        row for row in result_rows
        if row["rockType"] == rock["rockType"]
        and row["isEligible"]
    ]
    candidates.sort(key=lambda row: (
        -row["spatialIoU"],
        row["nSpheres"],
        row["volumeAbsoluteRelativeErrorPercent"],
    ))
    if not candidates:
        print(
            "WARNING: no geometrically admissible candidate for {}.".format(
                rock["gtsFile"]
            )
        )
        continue
    for rank, row in enumerate(candidates, start=1):
        row["eligibleRankForRock"] = rank
    best_by_rock[rock["rockType"]] = candidates[0]

for row in result_rows:
    if "eligibleRankForRock" not in row:
        row["eligibleRankForRock"] = ""

ranking_rows = sorted(result_rows, key=lambda row: (
    row["rockType"],
    not bool(row["isEligible"]),
    -row["spatialIoU"],
    row["nSpheres"],
    row["volumeAbsoluteRelativeErrorPercent"],
))
selection_rows = [
    best_by_rock[rock["rockType"]]
    for rock in rock_data if rock["rockType"] in best_by_rock
]

with open(OUTPUT_CSV, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(result_rows[0].keys()))
    writer.writeheader()
    writer.writerows(result_rows)

with open(OUTPUT_RANKING_CSV, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(ranking_rows[0].keys()))
    writer.writeheader()
    writer.writerows(ranking_rows)

with open(OUTPUT_SELECTION_CSV, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(result_rows[0].keys()))
    writer.writeheader()
    writer.writerows(selection_rows)

print("\nTOP ELIGIBLE CANDIDATES FOR EACH ROCK")
for rock in rock_data:
    candidates = [
        row for row in ranking_rows
        if row["rockType"] == rock["rockType"]
        and row["isEligible"]
    ]
    print("  " + rock["gtsFile"])
    if not candidates:
        print("    none")
    for row in candidates[:TOP_CANDIDATES_TO_REPORT]:
        print(
            "    {}. d={}, overlap={:.2f}r, rotation={}, "
            "volume ratio={:.1f}%, IoU={:.1f}%, GTS coverage={:.1f}%, "
            "containment={:.1f}%, "
            "spheres={}".format(
                row["eligibleRankForRock"],
                row["resolutionDivisor"],
                row["overlapDepthFractionOfRadius"],
                row["rotationLabel"],
                100.0 * row["volumeCoverageRatio"],
                100.0 * row["spatialIoU"],
                100.0 * row["gtsCoverage"],
                100.0 * row["clumpContainment"],
                row["nSpheres"],
            )
        )


# Regenerate each rock's own selected candidate in the YADE display scene.
rockTypeByBodyId = {}
for rock_index, rock in enumerate(rock_data):
    if rock["rockType"] not in best_by_rock:
        continue
    selected = best_by_rock[rock["rockType"]]
    rotation_setting = next(
        rotation for rotation in LATTICE_ROTATIONS
        if rotation[0] == selected["rotationLabel"]
    )
    member_bodies, _, _ = generate_lattice_members(
        rock,
        selected["resolutionDivisor"],
        selected["overlapDepthFractionOfRadius"],
        rotation_setting,
        material_id,
        colors[rock_index],
    )

    original_centers = np.asarray([
        [
            float(member.state.pos[0]),
            float(member.state.pos[1]),
            float(member.state.pos[2]),
        ]
        for member in member_bodies
    ])
    packing_center = 0.5 * (
        np.min(original_centers, axis=0)
        + np.max(original_centers, axis=0)
    )
    target_center = Vector3(rock_index * DISPLAY_SPACING, 0.0, 0.0)
    translation = target_center - Vector3(*packing_center)
    for member in member_bodies:
        member.state.pos += translation

    member_ids = O.bodies.append(member_bodies)
    O.bodies.clump(member_ids)
    for member_id in member_ids:
        rockTypeByBodyId[member_id] = rock["rockType"]


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    divisors = list(RESOLUTION_DIVISORS)
    overlaps = list(OVERLAP_DEPTH_FRACTIONS_OF_RADIUS)
    rotation_abbreviations = {
        "identity": "I",
        "x_30deg": "X",
        "y_30deg": "Y",
        "z_30deg": "Z",
    }

    def best_cell_candidate(rock_type, divisor, overlap):
        candidates = [
            row for row in result_rows
            if row["rockType"] == rock_type
            and row["resolutionDivisor"] == divisor
            and row["overlapDepthFractionOfRadius"] == overlap
            and row["isEligible"]
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda row: (
            -row["spatialIoU"],
            row["nSpheres"],
            row["volumeAbsoluteRelativeErrorPercent"],
        ))

    eligible_iou_percent = [
        100.0 * row["spatialIoU"]
        for row in result_rows if row["isEligible"]
    ]
    iou_color_min = 5.0 * math.floor(min(eligible_iou_percent) / 5.0)
    iou_color_max = 5.0 * math.ceil(max(eligible_iou_percent) / 5.0)
    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad(color="lightgray")

    # Core selection figure. For each divisor-overlap cell, the rotation with
    # the highest IoU is shown. Cell text gives IoU in percent followed by the
    # rotation abbreviation. The black border marks
    # the final candidate selected across the whole panel.
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for rock, axis in zip(rock_data, axes.flat):
        iou_matrix = np.full(
            (len(overlaps), len(divisors)), np.nan, dtype=float
        )
        rotation_matrix_labels = np.empty(iou_matrix.shape, dtype=object)
        for overlap_index, overlap in enumerate(overlaps):
            for divisor_index, divisor in enumerate(divisors):
                candidate = best_cell_candidate(
                    rock["rockType"], divisor, overlap
                )
                if candidate is None:
                    rotation_matrix_labels[overlap_index, divisor_index] = ""
                    continue
                iou_matrix[overlap_index, divisor_index] = (
                    100.0 * candidate["spatialIoU"]
                )
                rotation_matrix_labels[overlap_index, divisor_index] = (
                    rotation_abbreviations[candidate["rotationLabel"]]
                )

        image = axis.imshow(
            np.ma.masked_invalid(iou_matrix),
            aspect="auto",
            origin="lower",
            cmap=color_map,
            vmin=iou_color_min,
            vmax=iou_color_max,
        )
        axis.set_xticks(range(len(divisors)))
        axis.set_xticklabels(divisors)
        axis.set_yticks(range(len(overlaps)))
        axis.set_yticklabels(["{:.2f}".format(value) for value in overlaps])
        axis.set_xlabel("Resolution divisor")
        axis.set_ylabel("Overlap depth / radius")
        axis.set_title("Rock {}".format(rock["rockType"]))

        for overlap_index in range(len(overlaps)):
            for divisor_index in range(len(divisors)):
                iou_value = iou_matrix[overlap_index, divisor_index]
                if np.isnan(iou_value):
                    continue
                text_color = (
                    "white"
                    if iou_value < 0.5 * (iou_color_min + iou_color_max)
                    else "black"
                )
                axis.text(
                    divisor_index,
                    overlap_index,
                    "{:.0f}\n{}".format(
                        iou_value,
                        rotation_matrix_labels[
                            overlap_index, divisor_index
                        ],
                    ),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )

        selected = best_by_rock.get(rock["rockType"])
        if selected is not None:
            selected_column = divisors.index(selected["resolutionDivisor"])
            selected_row = overlaps.index(
                selected["overlapDepthFractionOfRadius"]
            )
            axis.add_patch(Rectangle(
                (selected_column - 0.5, selected_row - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="black",
                linewidth=2.5,
            ))

    colorbar_axis = figure.add_axes([0.915, 0.16, 0.015, 0.64])
    figure.colorbar(image, cax=colorbar_axis, label="Spatial IoU [%]")
    figure.suptitle(
        "Selection of hexagonal clump geometry",
        y=0.985,
        fontsize=16,
    )
    figure.legend(
        handles=[
            Patch(
                facecolor="lightgray",
                edgecolor="gray",
                label="Excluded: sphere limit or independent-volume ratio",
            ),
            Patch(
                facecolor="white",
                edgecolor="black",
                linewidth=2.0,
                label="Final selected candidate",
            ),
            Patch(
                facecolor="white",
                edgecolor="white",
                label="Cell: IoU [%]; I, X, Y, Z = rotation",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=True,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.89,
        bottom=0.08,
        top=0.84,
        hspace=0.34,
        wspace=0.23,
    )
    figure.savefig(OUTPUT_SELECTION_HEATMAP, dpi=220)
    plt.close(figure)

    # Compact summary of only the four selected candidates.
    selection_labels = [
        "Rock {}".format(row["rockType"]) for row in selection_rows
    ]
    x_positions = np.arange(len(selection_rows), dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    bar_width = 0.24
    for offset, (field_name, label) in zip(
        (-bar_width, 0.0, bar_width),
        (
            ("spatialIoU", "Spatial IoU"),
            ("gtsCoverage", "GTS coverage"),
            ("clumpContainment", "Clump containment"),
        ),
    ):
        axes[0].bar(
            x_positions + offset,
            [100.0 * row[field_name] for row in selection_rows],
            width=bar_width,
            label=label,
        )
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(selection_labels)
    axes[0].set_ylim(0.0, 100.0)
    axes[0].set_ylabel("Spatial metric [%]")
    axes[0].set_title("Spatial fidelity")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", alpha=0.25)

    counts = [row["nSpheres"] for row in selection_rows]
    selected_colors = [
        colors[row["rockType"] - 1] for row in selection_rows
    ]
    bars = axes[1].bar(selection_labels, counts, color=selected_colors)
    axes[1].axhline(
        MAX_SPHERES_PER_ROCK,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Sphere-count limit",
    )
    axes[1].set_ylabel("Number of member spheres")
    axes[1].set_title("Computational representation")
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, selection_rows):
        axes[1].text(
            bar.get_x() + 0.5 * bar.get_width(),
            bar.get_height() + 0.02 * MAX_SPHERES_PER_ROCK,
            "d={}\n{:.2f}r, {}".format(
                row["resolutionDivisor"],
                row["overlapDepthFractionOfRadius"],
                rotation_abbreviations[row["rotationLabel"]],
            ),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_ylim(0.0, 1.18 * MAX_SPHERES_PER_ROCK)
    figure.suptitle("Final selected clump candidates", y=0.98)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(OUTPUT_SELECTION_SUMMARY, dpi=220)
    plt.close(figure)

    # Sphere count as a direct function of the resolution divisor.  Each line
    # is the mean across lattice rotations for one overlap setting; the shaded
    # band gives the corresponding minimum-to-maximum range.
    figure, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharex=True)
    for rock, axis in zip(rock_data, axes.flat):
        for overlap in overlaps:
            means = []
            minimums = []
            maximums = []
            for divisor in divisors:
                values = np.asarray([
                    row["nSpheres"]
                    for row in result_rows
                    if row["rockType"] == rock["rockType"]
                    and row["resolutionDivisor"] == divisor
                    and row["overlapDepthFractionOfRadius"] == overlap
                ], dtype=float)
                means.append(float(np.mean(values)))
                minimums.append(float(np.min(values)))
                maximums.append(float(np.max(values)))
            line = axis.plot(
                divisors,
                means,
                marker="o",
                linewidth=1.5,
                label="Overlap = {:.2f}$r$".format(overlap),
            )[0]
            axis.fill_between(
                divisors,
                minimums,
                maximums,
                color=line.get_color(),
                alpha=0.12,
            )
        axis.axhline(
            MAX_SPHERES_PER_ROCK,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Sphere-count limit",
        )
        axis.set_title("Rock {}".format(rock["rockType"]))
        axis.set_xlabel("Resolution divisor")
        axis.set_ylabel("Number of member spheres")
        axis.set_xticks(divisors)
        axis.grid(alpha=0.25)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    handles.append(Patch(
        facecolor="gray",
        alpha=0.18,
        label="Shading: minimum–maximum across lattice rotations",
    ))
    legend_labels.append(
        "Shading: minimum–maximum across lattice rotations"
    )
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=True,
    )
    figure.suptitle(
        "Member-sphere count (lines: rotational mean; "
        "shading: rotational range)",
        y=0.985,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    figure.savefig(OUTPUT_COUNT_VS_DIVISOR_PLOT, dpi=220)
    plt.close(figure)

    # The independent volume ratio is a hard geometric check, not a spatial
    # coverage metric. Lines show rotational means; shading shows their range.
    figure, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharex=True)
    for rock, axis in zip(rock_data, axes.flat):
        for overlap in overlaps:
            means = []
            minimums = []
            maximums = []
            for divisor in divisors:
                values = np.asarray([
                    row["volumeCoverageRatio"]
                    for row in result_rows
                    if row["rockType"] == rock["rockType"]
                    and row["resolutionDivisor"] == divisor
                    and row["overlapDepthFractionOfRadius"] == overlap
                ])
                means.append(float(np.mean(values)))
                minimums.append(float(np.min(values)))
                maximums.append(float(np.max(values)))
            line = axis.plot(
                divisors,
                means,
                marker="o",
                label="overlap={:.2f}r".format(overlap),
            )[0]
            axis.fill_between(
                divisors,
                minimums,
                maximums,
                color=line.get_color(),
                alpha=0.12,
            )
        axis.axhspan(
            MIN_VOLUME_COVERAGE_RATIO,
            MAX_VOLUME_COVERAGE_RATIO,
            color="gray",
            alpha=0.15,
                label="Admissible independent-volume interval",
        )
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_title("Rock {}".format(rock["rockType"]))
        axis.set_xlabel("Resolution divisor")
        axis.set_ylabel("Independent clump volume / GTS volume")
        axis.grid(alpha=0.25)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    handles.append(Patch(
        facecolor="gray",
        alpha=0.35,
        label="Colored shading: minimum–maximum across lattice rotations",
    ))
    legend_labels.append(
        "Colored shading: minimum–maximum across lattice rotations"
    )
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=True,
    )
    figure.suptitle(
        "Independent volume ratio (lines: rotational mean; "
        "colored shading: rotational range)",
        y=0.985,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    figure.savefig(OUTPUT_VOLUME_PLOT, dpi=220)
    plt.close(figure)

    print("Saved plot:", OUTPUT_SELECTION_HEATMAP)
    print("Saved plot:", OUTPUT_SELECTION_SUMMARY)
    print("Saved plot:", OUTPUT_COUNT_VS_DIVISOR_PLOT)
    print("Saved plot:", OUTPUT_VOLUME_PLOT)
except ImportError:
    print("Matplotlib is unavailable; CSV results were still saved.")


O.save(OUTPUT_SCENE)

# Export only the final selected member spheres. The temporary candidates used
# during the parameter sweep were never appended to O.bodies and are therefore
# absent from this file. In ParaView, use a Glyph filter with "radius" as the
# scale array if the reader displays the spheres as points.
if rockTypeByBodyId:
    vtk_exporter = export.VTKExporter(VTK_OUTPUT_PREFIX)
    vtk_exporter.exportSpheres(
        what={"rockType": "rockTypeByBodyId.get(b.id,0)"},
        numLabel=0,
    )
    print("Saved VTK sphere export with prefix:", VTK_OUTPUT_PREFIX)
else:
    print("Skipped VTK export: no geometrically admissible clumps.")

print("Saved CSV:", OUTPUT_CSV)
print("Saved per-rock ranking CSV:", OUTPUT_RANKING_CSV)
print("Saved per-rock selection CSV:", OUTPUT_SELECTION_CSV)
print("Saved selected-candidate YADE scene:", OUTPUT_SCENE)
