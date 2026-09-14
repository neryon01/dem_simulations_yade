"""
UPDATED SCRIPT — 2026-09-07 — INTEGRATED HEXAGONAL/ORTHOGONAL STUDY + PLOTS

Purpose
-------
Test, rather than assume, whether YADE's regularHexa or regularOrtho lattice
represents each input GTS rock more faithfully.  For every deterministic
divisor/overlap/rotation candidate the script reports:

  * true spatial intersection-over-union (IoU);
  * GTS coverage = intersection / GTS volume;
  * clump containment = intersection / clump volume;
  * one-sided GTS-to-clump surface-distance mean, RMS, 95th percentile and max,
    all also normalized by the rock x-extent dx.

The old quantity V_clump/V_GTS is retained under the unambiguous name
``independentVolumeRatio``.  It is NOT called coverage, because two displaced
objects can have equal volumes and zero spatial overlap.

The two lattice methods use identical radii, overlaps and rotations.  All
Monte-Carlo samples are seeded and shared between candidates of the same rock,
so the study is reproducible and the comparison is paired and fair.

Sphericity and convexity are calculated only for the original reference mesh
(the GTS representation of the STL), not for any sphere clump.

The spatially selected candidate for every rock/method is exported in its
original GTS coordinate system to a separate VTK file.  No display translation
is applied, so the corresponding GTS can be overlaid directly in ParaView.

Run with:
    yade generate_hexa_ortho_fidelity_comparison_v1.py
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

GTS_FILES = ["rock_1.gts", "rock_2.gts", "rock_3.gts", "rock_4.gts"]

RESOLUTION_DIVISORS = list(range(6, 19))
OVERLAP_DEPTH_FRACTIONS_OF_RADIUS = [0.10, 0.30, 0.50, 0.70]
LATTICE_ROTATIONS = [
    ("identity", (1.0, 0.0, 0.0), 0.0),
    ("x_30deg", (1.0, 0.0, 0.0), 30.0),
    ("y_30deg", (0.0, 1.0, 0.0), 30.0),
    ("z_30deg", (0.0, 0.0, 1.0), 30.0),
]

MAX_SPHERES_PER_ROCK = 60

# This interval is a hard admissibility requirement before spatial ranking.
MIN_INDEPENDENT_VOLUME_RATIO = 0.90
MAX_INDEPENDENT_VOLUME_RATIO = 1.10

# Accuracy/runtime controls.  Increase for final publication numbers if a
# convergence check shows that the reported values still change materially.
UNION_VOLUME_SAMPLES = 2 ** 17
SPATIAL_OVERLAP_SAMPLES = 2 ** 17
GTS_SURFACE_SAMPLES = 8192
CLUMP_SURFACE_DIRECTIONS_PER_SPHERE = 256

OUTPUT_ALL_CSV = "hexa_ortho_all_candidates.csv"
OUTPUT_SELECTION_CSV = "hexa_ortho_spatially_selected.csv"
OUTPUT_SPATIAL_PLOT = "hexa_ortho_selected_spatial_fidelity.png"
OUTPUT_DISTANCE_PLOT = "hexa_ortho_selected_surface_distance.png"
OUTPUT_SCATTER_PLOT = "hexa_ortho_iou_vs_surface_distance.png"
VTK_PREFIX = "hexa_ortho_selected"


# -----------------------------------------------------------------------------
# GTS MESH AND REFERENCE METRICS
# -----------------------------------------------------------------------------

def read_gts_mesh(path):
    with open(path, "r") as handle:
        lines = [
            line.strip() for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]

    n_vertices, n_edges, n_faces = map(int, lines[0].split()[:3])
    vertices = np.asarray([
        list(map(float, lines[1 + i].split()[:3]))
        for i in range(n_vertices)
    ])
    edge_start = 1 + n_vertices
    edges = []
    for i in range(n_edges):
        first, second = map(int, lines[edge_start + i].split()[:2])
        edges.append((first - 1, second - 1))
    face_start = edge_start + n_edges
    faces = []
    for i in range(n_faces):
        edge_ids = [int(v) - 1 for v in lines[face_start + i].split()[:3]]
        vertex_ids = set()
        for edge_id in edge_ids:
            vertex_ids.update(edges[edge_id])
        if len(vertex_ids) != 3:
            raise ValueError("Invalid triangular face in {}".format(path))
        faces.append(tuple(sorted(vertex_ids)))
    return vertices, np.asarray(faces, dtype=int)


def orient_closed_faces(vertices, faces):
    edge_uses = defaultdict(list)
    for face_index, face in enumerate(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]),
                              (face[2], face[0])):
            edge = (min(first, second), max(first, second))
            direction = 1 if (first, second) == edge else -1
            edge_uses[edge].append((face_index, direction))
    if any(len(uses) != 2 for uses in edge_uses.values()):
        raise ValueError("GTS surface is not closed and manifold")

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
                    raise ValueError("Inconsistent GTS face orientation")
        components.append(component)

    oriented = faces.copy()
    oriented[flip, 1], oriented[flip, 2] = (
        oriented[flip, 2].copy(), oriented[flip, 1].copy()
    )
    for component in components:
        indices = np.asarray(component, dtype=int)
        triangles = vertices[oriented[indices]]
        signed_volume = np.sum(np.einsum(
            "ij,ij->i", triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2])
        )) / 6.0
        if signed_volume < 0.0:
            oriented[indices, 1], oriented[indices, 2] = (
                oriented[indices, 2].copy(), oriented[indices, 1].copy()
            )
    return oriented


def descriptor_values(volume, area, hull_volume):
    """Calculate descriptors for the original reference mesh only."""
    return {
        "volume": float(volume),
        "surfaceArea": float(area),
        "convexHullVolume": float(hull_volume),
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
    volume = abs(float(np.sum(np.einsum(
        "ij,ij->i", triangles[:, 0],
        np.cross(triangles[:, 1], triangles[:, 2])
    )) / 6.0))
    return descriptor_values(volume, area, float(ConvexHull(vertices).volume))


def sample_mesh_surface(vertices, faces, count, seed):
    """Area-uniform deterministic samples on the triangular GTS surface."""
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    ), axis=1)
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(triangles), size=count, p=areas / np.sum(areas))
    selected = triangles[chosen]
    u = rng.random_sample(count)
    v = rng.random_sample(count)
    reflected = (u + v) > 1.0
    u[reflected] = 1.0 - u[reflected]
    v[reflected] = 1.0 - v[reflected]
    return (
        selected[:, 0]
        + u[:, None] * (selected[:, 1] - selected[:, 0])
        + v[:, None] * (selected[:, 2] - selected[:, 0])
    )


# -----------------------------------------------------------------------------
# EQUAL-RADIUS SPHERE-UNION METRICS
# -----------------------------------------------------------------------------

def fibonacci_directions(count):
    indices = np.arange(count, dtype=float)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radial = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
    angle = golden_angle * indices
    return np.column_stack((radial * np.cos(angle), radial * np.sin(angle), z))


def sphere_union_volume(centers, radius, seed):
    lower = np.min(centers, axis=0) - radius
    upper = np.max(centers, axis=0) + radius
    rng = np.random.RandomState(seed)
    points = lower + rng.random_sample((UNION_VOLUME_SAMPLES, 3)) * (upper - lower)
    distances, _ = cKDTree(centers).query(points, k=1)
    return float(np.prod(upper - lower)) * float(np.mean(distances <= radius))


def exposed_sphere_surface_points(centers, radius, direction_count):
    directions = fibonacci_directions(direction_count)
    tree = cKDTree(centers)
    exposed_sets = []
    for index, center in enumerate(centers):
        points = center + radius * directions
        covered = np.zeros(len(points), dtype=bool)
        for neighbor in tree.query_ball_point(center, 2.0 * radius):
            if neighbor == index:
                continue
            delta = points - centers[neighbor]
            covered |= np.einsum("ij,ij->i", delta, delta) < radius ** 2
        exposed_sets.append(points[~covered])
    return np.vstack(exposed_sets)


def clump_metrics(centers, radius, seed):
    return {"volume": sphere_union_volume(centers, radius, seed)}


# -----------------------------------------------------------------------------
# TRUE SPATIAL OVERLAP AND SURFACE DISTANCE
# -----------------------------------------------------------------------------

def make_shared_spatial_sample(rock, seed):
    """One fixed evaluation cloud used for every candidate of this rock."""
    max_radius = rock["dx"] / float(min(RESOLUTION_DIVISORS))
    lower = rock["lower"] - max_radius
    upper = rock["upper"] + max_radius
    rng = np.random.RandomState(seed)
    points = lower + rng.random_sample((SPATIAL_OVERLAP_SAMPLES, 3)) * (upper - lower)

    with open(rock["gtsFile"], "r") as handle:
        surface = gts.read(handle)
    predicate = pack.inGtsSurface(surface, True)
    inside_gts = np.fromiter(
        # YADE 2022.01a exposes Predicate.__call__, but not the newer-looking
        # Python convenience method Predicate.containsPoint.
        (predicate(Vector3(*point), 0.0) for point in points),
        dtype=bool,
        count=len(points),
    )
    if not np.any(inside_gts):
        raise RuntimeError("Spatial sample found no point inside {}".format(
            rock["gtsFile"]
        ))
    return points, inside_gts, float(np.prod(upper - lower))


def spatial_overlap_metrics(points, inside_gts, box_volume, centers, radius):
    distances, _ = cKDTree(centers).query(points, k=1)
    inside_clump = distances <= radius
    intersection = inside_gts & inside_clump
    union = inside_gts | inside_clump
    n_gts = int(np.sum(inside_gts))
    n_clump = int(np.sum(inside_clump))
    n_intersection = int(np.sum(intersection))
    return {
        "spatialIoU": float(n_intersection) / float(np.sum(union)),
        "gtsCoverage": float(n_intersection) / n_gts,
        "clumpContainment": (
            float(n_intersection) / n_clump if n_clump else 0.0
        ),
        "spatialGtsVolumeEstimate": box_volume * n_gts / len(points),
        "spatialClumpVolumeEstimate": box_volume * n_clump / len(points),
        "spatialIntersectionVolumeEstimate": (
            box_volume * n_intersection / len(points)
        ),
    }


def surface_distance_metrics(gts_surface_points, centers, radius, dx):
    """One-sided distance from the reference GTS surface to clump boundary."""
    clump_surface_points = exposed_sphere_surface_points(
        centers, radius, CLUMP_SURFACE_DIRECTIONS_PER_SPHERE
    )
    distances, _ = cKDTree(clump_surface_points).query(gts_surface_points, k=1)
    mean = float(np.mean(distances))
    rms = float(np.sqrt(np.mean(distances ** 2)))
    p95 = float(np.percentile(distances, 95.0))
    maximum = float(np.max(distances))
    return {
        "gtsToClumpSurfaceMean": mean,
        "gtsToClumpSurfaceRms": rms,
        "gtsToClumpSurfaceP95": p95,
        "gtsToClumpSurfaceMax": maximum,
        "gtsToClumpSurfaceMeanOverDx": mean / dx,
        "gtsToClumpSurfaceRmsOverDx": rms / dx,
        "gtsToClumpSurfaceP95OverDx": p95 / dx,
        "gtsToClumpSurfaceMaxOverDx": maximum / dx,
    }


# -----------------------------------------------------------------------------
# LATTICE GENERATION
# -----------------------------------------------------------------------------

def rotation_matrix(axis, angle_degrees):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    angle = math.radians(angle_degrees)
    c, s, q = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.asarray([
        [c + x*x*q, x*y*q - z*s, x*z*q + y*s],
        [y*x*q + z*s, c + y*y*q, y*z*q - x*s],
        [z*x*q - y*s, z*y*q + x*s, c + z*z*q],
    ])


def generate_lattice_centers(rock, method, divisor, overlap_fraction, rotation,
                             material_id, color):
    label, axis, angle_degrees = rotation
    with open(rock["gtsFile"], "r") as handle:
        surface = gts.read(handle)
    center = rock["center"]
    if angle_degrees:
        surface.translate(-center[0], -center[1], -center[2])
        surface.rotate(axis[0], axis[1], axis[2], math.radians(angle_degrees))
        surface.translate(center[0], center[1], center[2])

    predicate = pack.inGtsSurface(surface, True)
    radius = rock["dx"] / float(divisor)
    gap = -overlap_fraction * radius
    generator = pack.regularHexa if method == "hexa" else pack.regularOrtho
    bodies = generator(
        predicate, radius=radius, gap=gap, material=material_id, color=color
    )
    centers = np.asarray([
        [float(body.state.pos[0]), float(body.state.pos[1]),
         float(body.state.pos[2])] for body in bodies
    ])
    if angle_degrees and len(centers):
        inverse = rotation_matrix(axis, angle_degrees).T
        centers = center + np.dot(centers - center, inverse.T)
    return centers, radius, gap


def bodies_from_centers(centers, radius, material_id, color):
    return [sphere(Vector3(*center), radius, material=material_id, color=color)
            for center in centers]


# -----------------------------------------------------------------------------
# STUDY
# -----------------------------------------------------------------------------

O.reset()
material_id = O.materials.append(FrictMat(
    young=1.0e7, poisson=0.25, frictionAngle=0.0, density=2600.0
))
colors = {"hexa": (0.15, 0.55, 0.95), "ortho": (0.95, 0.40, 0.15)}
methods = ("hexa", "ortho")
method_labels = {
    "hexa": "Hexagonal packing",
    "ortho": "Orthogonal packing",
}

print("HEXA–ORTHO GTS GEOMETRIC-FIDELITY COMPARISON")
print("Divisors:", RESOLUTION_DIVISORS)
print("Overlap-depth fractions / radius:", OVERLAP_DEPTH_FRACTIONS_OF_RADIUS)
print("Sphere-count limit:", MAX_SPHERES_PER_ROCK)
print("Spatial samples per rock:", SPATIAL_OVERLAP_SAMPLES)

rocks = []
for rock_index, gts_file in enumerate(GTS_FILES):
    vertices, faces = read_gts_mesh(gts_file)
    faces = orient_closed_faces(vertices, faces)
    lower, upper = np.min(vertices, axis=0), np.max(vertices, axis=0)
    rock = {
        "rockType": rock_index + 1,
        "gtsFile": gts_file,
        "vertices": vertices,
        "faces": faces,
        "lower": lower,
        "upper": upper,
        "center": 0.5 * (lower + upper),
        "dx": float(upper[0] - lower[0]),
        "reference": mesh_metrics(vertices, faces),
    }
    rock["spatialPoints"], rock["insideGts"], rock["spatialBoxVolume"] = (
        make_shared_spatial_sample(rock, 71000 + rock_index)
    )
    rock["surfacePoints"] = sample_mesh_surface(
        vertices, faces, GTS_SURFACE_SAMPLES, 72000 + rock_index
    )
    rocks.append(rock)

rows = []
candidate_centers = {}
for rock_index, rock in enumerate(rocks):
    print("\n" + rock["gtsFile"])
    for method in methods:
        for divisor in RESOLUTION_DIVISORS:
            for overlap_fraction in OVERLAP_DEPTH_FRACTIONS_OF_RADIUS:
                for rotation in LATTICE_ROTATIONS:
                    rotation_label, rotation_axis, angle_degrees = rotation
                    centers, radius, gap = generate_lattice_centers(
                        rock, method, divisor, overlap_fraction, rotation,
                        material_id, colors[method]
                    )
                    if len(centers) == 0:
                        continue

                    key = (rock["rockType"], method, divisor,
                           overlap_fraction, rotation_label)
                    candidate_centers[key] = centers
                    seed = (73000 + rock_index * 10000
                            + (0 if method == "hexa" else 5000)
                            + divisor * 100
                            + int(round(overlap_fraction * 100))
                            + int(round(angle_degrees)))
                    clump = clump_metrics(centers, radius, seed)
                    spatial = spatial_overlap_metrics(
                        rock["spatialPoints"], rock["insideGts"],
                        rock["spatialBoxVolume"], centers, radius
                    )
                    distance = surface_distance_metrics(
                        rock["surfacePoints"], centers, radius, rock["dx"]
                    )
                    reference = rock["reference"]
                    volume_ratio = clump["volume"] / reference["volume"]
                    within_count = len(centers) <= MAX_SPHERES_PER_ROCK
                    within_ratio = (MIN_INDEPENDENT_VOLUME_RATIO <= volume_ratio
                                    <= MAX_INDEPENDENT_VOLUME_RATIO)
                    row = {
                        "rockType": rock["rockType"],
                        "gtsFile": rock["gtsFile"],
                        "latticeMethod": method,
                        "resolutionDivisor": divisor,
                        "overlapDepthFractionOfRadius": overlap_fraction,
                        "rotationLabel": rotation_label,
                        "rotationAxisX": rotation_axis[0],
                        "rotationAxisY": rotation_axis[1],
                        "rotationAxisZ": rotation_axis[2],
                        "rotationAngleDegrees": angle_degrees,
                        "memberRadius": radius,
                        "gap": gap,
                        "nSpheres": len(centers),
                        "withinSphereLimit": int(within_count),
                        "referenceVolume": reference["volume"],
                        "clumpUnionVolume": clump["volume"],
                        "independentVolumeRatio": volume_ratio,
                        "withinIndependentVolumeRatioInterval": int(within_ratio),
                        "isEligible": int(within_count and within_ratio),
                        "referenceSphericity": reference["sphericity"],
                        "referenceConvexity": reference["convexity"],
                        "referenceSurfaceArea": reference["surfaceArea"],
                        "referenceConvexHullVolume": reference["convexHullVolume"],
                    }
                    row.update(spatial)
                    row.update(distance)
                    rows.append(row)
                    print(
                        "  {:5s} d={:2d}, overlap={:.2f}r, {:8s}: "
                        "N={:3d}, IoU={:5.1f}%, cover={:5.1f}%, "
                        "contain={:5.1f}%, p95/dx={:.4f}".format(
                            method, divisor, overlap_fraction, rotation_label,
                            len(centers), 100.0 * spatial["spatialIoU"],
                            100.0 * spatial["gtsCoverage"],
                            100.0 * spatial["clumpContainment"],
                            distance["gtsToClumpSurfaceP95OverDx"],
                        )
                    )


# Select separately for every rock and lattice. Spatial IoU is primary;
# surface distance and member-sphere count provide deterministic tie-breakers.
selected_rows = []
for rock in rocks:
    for method in methods:
        candidates = [
            row for row in rows
            if row["rockType"] == rock["rockType"]
            and row["latticeMethod"] == method
            and row["isEligible"]
        ]
        if not candidates:
            candidates = [
                row for row in rows
                if row["rockType"] == rock["rockType"]
                and row["latticeMethod"] == method
                and row["withinSphereLimit"]
            ]
            status = "fallback_no_volume_eligible_candidate"
        else:
            status = "spatially_selected"
        if not candidates:
            print("WARNING: no <= {} sphere {} candidate for {}".format(
                MAX_SPHERES_PER_ROCK, method, rock["gtsFile"]
            ))
            continue
        candidates.sort(key=lambda row: (
            -row["spatialIoU"],
            row["gtsToClumpSurfaceP95OverDx"],
            row["nSpheres"],
        ))
        selected = candidates[0]
        selected["selectionStatus"] = status
        selected_rows.append(selected)

for row in rows:
    row["selectionStatus"] = row.get("selectionStatus", "not_selected")

with open(OUTPUT_ALL_CSV, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
with open(OUTPUT_SELECTION_CSV, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(selected_rows)

print("\nSPATIALLY SELECTED CANDIDATES")
for row in selected_rows:
    print(
        "  rock {}, {:5s}: d={}, overlap={:.2f}r, rotation={}, N={}, "
        "IoU={:.1f}%, coverage={:.1f}%, "
        "containment={:.1f}%, p95/dx={:.4f}".format(
            row["rockType"], row["latticeMethod"], row["resolutionDivisor"],
            row["overlapDepthFractionOfRadius"], row["rotationLabel"],
            row["nSpheres"], 100.0 * row["spatialIoU"],
            100.0 * row["gtsCoverage"],
            100.0 * row["clumpContainment"],
            row["gtsToClumpSurfaceP95OverDx"],
        )
    )


# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rock_labels = ["Rock {}".format(rock["rockType"]) for rock in rocks]
    x = np.arange(len(rocks), dtype=float)
    width = 0.36

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    spatial_fields = [
        ("spatialIoU", "Spatial intersection over union"),
        ("gtsCoverage", "GTS coverage"),
        ("clumpContainment", "Clump containment"),
    ]
    for axis, (field, title) in zip(axes, spatial_fields):
        for method_index, method in enumerate(methods):
            values = [next(row[field] for row in selected_rows
                           if row["rockType"] == rock["rockType"]
                           and row["latticeMethod"] == method)
                      for rock in rocks]
            axis.bar(x + (method_index - 0.5) * width, values, width,
                     label=method_labels[method], color=colors[method])
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(rock_labels, rotation=25)
        axis.set_ylim(0.0, 1.05)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Fraction")
    axes[-1].legend()
    figure.suptitle("Spatial fidelity of selected clumps")
    figure.tight_layout()
    figure.savefig(OUTPUT_SPATIAL_PLOT, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    for method_index, method in enumerate(methods):
        values = [next(row["gtsToClumpSurfaceP95OverDx"] for row in selected_rows
                       if row["rockType"] == rock["rockType"]
                       and row["latticeMethod"] == method)
                  for rock in rocks]
        axis.bar(x + (method_index - 0.5) * width, values, width,
                 label=method_labels[method], color=colors[method])
    axis.set_xticks(x)
    axis.set_xticklabels(rock_labels)
    axis.set_ylabel("95th-percentile GTS-to-clump distance / $d_x$")
    axis.set_title("Surface discrepancy of selected clumps")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DISTANCE_PLOT, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for rock, axis in zip(rocks, axes.flat):
        for method in methods:
            candidates = [row for row in rows
                          if row["rockType"] == rock["rockType"]
                          and row["latticeMethod"] == method
                          and row["withinSphereLimit"]]
            axis.scatter(
                [row["spatialIoU"] for row in candidates],
                [row["gtsToClumpSurfaceP95OverDx"] for row in candidates],
                s=[18.0 + 1.55 * row["nSpheres"] for row in candidates],
                alpha=0.55,
                color=colors[method],
                edgecolors=colors[method],
                linewidths=0.7,
                label=method_labels[method],
            )
        axis.set_title("Rock {}".format(rock["rockType"]))
        axis.set_xlabel("Spatial intersection over union")
        axis.set_ylabel("95th-percentile surface distance / $d_x$")
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.42, 1.0),
        ncol=2,
        frameon=True,
    )

    sphere_counts = [
        row["nSpheres"] for row in rows if row["withinSphereLimit"]
    ]
    representative_counts = sorted(set([
        int(np.min(sphere_counts)),
        int(np.median(sphere_counts)),
        int(np.max(sphere_counts)),
    ]))
    size_handles = [
        axes.flat[0].scatter(
            [], [],
            s=18.0 + 1.55 * count,
            facecolor="none",
            edgecolor="#555555",
            linewidth=0.9,
            label=str(count),
        )
        for count in representative_counts
    ]
    figure.legend(
        handles=size_handles,
        labels=[str(count) for count in representative_counts],
        title="Member spheres",
        loc="upper center",
        bbox_to_anchor=(0.83, 1.0),
        ncol=len(size_handles),
        frameon=True,
    )
    figure.suptitle(
        "Spatial overlap and surface discrepancy of clump candidates",
        y=0.95,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.savefig(OUTPUT_SCATTER_PLOT, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print("Saved plot:", OUTPUT_SPATIAL_PLOT)
    print("Saved plot:", OUTPUT_DISTANCE_PLOT)
    print("Saved plot:", OUTPUT_SCATTER_PLOT)
except ImportError:
    print("Matplotlib unavailable; CSV results were still saved")


# -----------------------------------------------------------------------------
# ORIGINAL-COORDINATE VTK EXPORT
# -----------------------------------------------------------------------------

for selected in selected_rows:
    key = (
        selected["rockType"], selected["latticeMethod"],
        selected["resolutionDivisor"],
        selected["overlapDepthFractionOfRadius"], selected["rotationLabel"],
    )
    centers = candidate_centers[key]
    bodies = bodies_from_centers(
        centers, selected["memberRadius"], material_id,
        colors[selected["latticeMethod"]]
    )
    ids = O.bodies.append(bodies)
    file_prefix = "{}_{}_rock_{}".format(
        VTK_PREFIX, selected["latticeMethod"], selected["rockType"]
    )
    vtk_exporter = export.VTKExporter(file_prefix)
    vtk_exporter.exportSpheres(ids=ids, numLabel=0)
    for body_id in ids:
        O.bodies.erase(body_id)
    print("Saved original-coordinate VTK with prefix:", file_prefix)

print("Saved all-candidate CSV:", OUTPUT_ALL_CSV)
print("Saved selected-candidate CSV:", OUTPUT_SELECTION_CSV)
