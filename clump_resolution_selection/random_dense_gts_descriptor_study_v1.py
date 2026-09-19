"""

Outputs:
    random_dense_individual_results.csv
    random_dense_summary_mean_std.csv
    random_dense_generation_failures.csv (only if a call fails)
    random_dense_volume_mean_std.png
    random_dense_runtime_mean_std.png
    random_dense_sphere_count_vs_divisor.png
    random_dense_eligibility_heatmaps.png
    random_dense_provisional_best_clumps.vtk-* / .vtu
    random_dense_provisional_best_clumps.yade.gz

Run with:
    yade random_dense_gts_descriptor_study_v1.py
"""

from __future__ import print_function

import csv
import math
import time
from collections import defaultdict, deque

import gts
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from yade import Vector3, export, pack, utils


# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------

GTS_FILES = [
    "rock_1.gts",
    "rock_2.gts",
    "rock_3.gts",
    "rock_4.gts",
]

# Use the same divisor set as the deterministic packing comparison.  A separate
# dictionary is retained so rock-specific ranges can still be restored easily.
COMMON_RESOLUTION_DIVISORS = list(range(6, 19))
DIVISORS_BY_ROCK = {
    1: COMMON_RESOLUTION_DIVISORS,
    2: COMMON_RESOLUTION_DIVISORS,
    3: COMMON_RESOLUTION_DIVISORS,
    4: COMMON_RESOLUTION_DIVISORS,
}

RREL_FUZZ = 0.40
N_REALISATIONS = 5
BASE_SEED = 24680

MAX_SPHERES_PER_ROCK = 60
MIN_VOLUME_COVERAGE_RATIO = 0.90
MAX_VOLUME_COVERAGE_RATIO = 1.10

# Native aperiodic randomDensePack settings. spheresInCell=0 avoids the
# periodic-cell route. No memoisation database is used, so every documented
# seed is generated independently.
CROP_LAYERS = 1
USE_OBB = True
SPHERES_IN_CELL = 0

# Numerical union-volume settings. The same Monte-Carlo seed is used for every
# candidate so that sampling noise does not masquerade as realisation scatter.
UNION_VOLUME_SAMPLES = 2 ** 17
UNION_VOLUME_CHUNK_SIZE = 4096
CONTACT_RELATIVE_TOLERANCE = 2.0e-2

DISPLAY_SPACING = 2.0

OUTPUT_INDIVIDUAL_CSV = "random_dense_individual_results.csv"
OUTPUT_SUMMARY_CSV = "random_dense_summary_mean_std.csv"
OUTPUT_FAILURE_CSV = "random_dense_generation_failures.csv"
OUTPUT_VOLUME_PLOT = "random_dense_volume_mean_std.png"
OUTPUT_RUNTIME_PLOT = "random_dense_runtime_mean_std.png"
OUTPUT_COUNT_VS_DIVISOR_PLOT = "random_dense_sphere_count_vs_divisor.png"
OUTPUT_ELIGIBILITY_HEATMAP = "random_dense_eligibility_heatmaps.png"
OUTPUT_SCENE = "random_dense_provisional_best_clumps.yade.gz"
VTK_OUTPUT_PREFIX = "random_dense_provisional_best_clumps"


# -----------------------------------------------------------------------------
# GTS REFERENCE GEOMETRY
# -----------------------------------------------------------------------------

def read_gts_mesh(path):
    """Read vertices and triangular faces from a GTS file."""
    with open(path, "r") as handle:
        lines = [
            line.strip()
            for line in handle
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
    """Orient each connected triangle component consistently and outwards."""
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


def descriptor_values(volume, area, hull_volume):
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
# VARIABLE-RADIUS SPHERE-UNION GEOMETRY
# -----------------------------------------------------------------------------

def variable_radius_union_volume(centers, radii):
    """Deterministic Monte-Carlo estimate of the variable-radius union."""
    lower = np.min(centers - radii[:, None], axis=0)
    upper = np.max(centers + radii[:, None], axis=0)
    rng = np.random.RandomState(0)
    unit_points = rng.random_sample((UNION_VOLUME_SAMPLES, 3))
    points = lower + unit_points * (upper - lower)
    radii_squared = radii ** 2
    inside_count = 0

    for start in range(0, len(points), UNION_VOLUME_CHUNK_SIZE):
        chunk = points[start:start + UNION_VOLUME_CHUNK_SIZE]
        delta = chunk[:, None, :] - centers[None, :, :]
        distance_squared = np.einsum("ijk,ijk->ij", delta, delta)
        inside_count += int(np.count_nonzero(
            np.any(distance_squared <= radii_squared[None, :], axis=1)
        ))

    fraction = float(inside_count) / float(len(points))
    return float(np.prod(upper - lower)) * fraction


def contact_graph_metrics(centers, radii):
    """Return component count and largest-component fraction."""
    count = len(centers)
    if count == 1:
        return 1, 1.0

    tree = cKDTree(centers)
    candidate_pairs = tree.query_pairs(2.0 * float(np.max(radii)))
    adjacency = [[] for _ in range(count)]

    for first, second in candidate_pairs:
        distance = float(np.linalg.norm(centers[first] - centers[second]))
        tolerance = CONTACT_RELATIVE_TOLERANCE * min(
            radii[first], radii[second]
        )
        if distance <= radii[first] + radii[second] + tolerance:
            adjacency[first].append(second)
            adjacency[second].append(first)

    visited = np.zeros(count, dtype=bool)
    component_sizes = []
    for start in range(count):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        component_sizes.append(size)

    return len(component_sizes), max(component_sizes) / float(count)


def clump_metrics(centers, radii):
    volume = variable_radius_union_volume(centers, radii)
    component_count, largest_fraction = contact_graph_metrics(centers, radii)
    return {
        "volume": volume,
        "contactGraphComponents": component_count,
        "largestComponentFraction": largest_fraction,
    }


def relative_error_percent(approximation, reference):
    return 100.0 * (approximation - reference) / reference


def seed_for(rock_type, divisor, realisation):
    """Create unique, reproducible seeds without relying on Python hashing."""
    return BASE_SEED + 10000 * rock_type + 100 * divisor + realisation


def extract_sphere_pack(sphere_pack):
    centers = []
    radii = []
    for center, radius in sphere_pack:
        centers.append([
            float(center[0]), float(center[1]), float(center[2])
        ])
        radii.append(float(radius))
    return np.asarray(centers, dtype=float), np.asarray(radii, dtype=float)


def generate_random_dense_candidate(rock, divisor, seed, material_id, color):
    """Run one native aperiodic randomDensePack realisation."""
    with open(rock["gtsFile"], "r") as handle:
        surface = gts.read(handle)
    predicate = pack.inGtsSurface(surface)
    requested_mean_radius = rock["dx"] / float(divisor)

    started = time.time()
    sphere_pack = pack.randomDensePack(
        predicate,
        radius=requested_mean_radius,
        material=material_id,
        cropLayers=CROP_LAYERS,
        rRelFuzz=RREL_FUZZ,
        spheresInCell=SPHERES_IN_CELL,
        memoizeDb=None,
        useOBB=USE_OBB,
        color=color,
        returnSpherePack=True,
        seed=seed,
    )
    generation_seconds = time.time() - started
    centers, radii = extract_sphere_pack(sphere_pack)
    if len(radii) == 0:
        raise RuntimeError("randomDensePack returned no member spheres.")
    return centers, radii, requested_mean_radius, generation_seconds


# -----------------------------------------------------------------------------
# RUN STUDY
# -----------------------------------------------------------------------------

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

rock_data = []
for rock_index, gts_file in enumerate(GTS_FILES):
    vertices, faces = read_gts_mesh(gts_file)
    lower = np.min(vertices, axis=0)
    upper = np.max(vertices, axis=0)
    rock_data.append({
        "rockType": rock_index + 1,
        "gtsFile": gts_file,
        "dx": float(upper[0] - lower[0]),
        "reference": mesh_metrics(vertices, faces),
    })

print("RANDOM-DENSE GTS CLUMP FIDELITY STUDY")
print("rRelFuzz:", RREL_FUZZ)
print("Realisations per rock and divisor:", N_REALISATIONS)
print("Base seed:", BASE_SEED)
print("Divisors by rock:", DIVISORS_BY_ROCK)
print("Maximum member spheres:", MAX_SPHERES_PER_ROCK)
print(
    "Admissible volume coverage: {:.0f}% to {:.0f}%".format(
        100.0 * MIN_VOLUME_COVERAGE_RATIO,
        100.0 * MAX_VOLUME_COVERAGE_RATIO,
    )
)
print("randomDensePack: aperiodic, cropLayers={}, useOBB={}".format(
    CROP_LAYERS, USE_OBB
))

individual_rows = []
failures = []
candidate_geometry = {}

for rock_index, rock in enumerate(rock_data):
    print("\n" + rock["gtsFile"])
    for divisor in DIVISORS_BY_ROCK[rock["rockType"]]:
        for realisation in range(1, N_REALISATIONS + 1):
            seed = seed_for(rock["rockType"], divisor, realisation)
            print(
                "  generating d={}, realisation={}/{}, seed={} ...".format(
                    divisor, realisation, N_REALISATIONS, seed
                )
            )
            try:
                centers, radii, requested_radius, generation_seconds = (
                    generate_random_dense_candidate(
                        rock,
                        divisor,
                        seed,
                        material_id,
                        colors[rock_index],
                    )
                )
                metrics = clump_metrics(centers, radii)
            except Exception as error:
                failures.append({
                    "rockType": rock["rockType"],
                    "gtsFile": rock["gtsFile"],
                    "resolutionDivisor": divisor,
                    "realisation": realisation,
                    "seed": seed,
                    "error": str(error),
                })
                print("    FAILED:", error)
                continue

            reference = rock["reference"]
            volume_coverage = metrics["volume"] / reference["volume"]
            within_sphere_limit = len(radii) <= MAX_SPHERES_PER_ROCK
            within_volume_interval = (
                MIN_VOLUME_COVERAGE_RATIO
                <= volume_coverage
                <= MAX_VOLUME_COVERAGE_RATIO
            )
            connected = metrics["contactGraphComponents"] == 1

            row = {
                "rockType": rock["rockType"],
                "gtsFile": rock["gtsFile"],
                "resolutionDivisor": divisor,
                "realisation": realisation,
                "seed": seed,
                "rRelFuzz": RREL_FUZZ,
                "requestedMeanRadius": requested_radius,
                "generatedMeanRadius": float(np.mean(radii)),
                "generatedStdRadius": float(np.std(radii, ddof=1)) if len(radii) > 1 else 0.0,
                "generatedMinRadius": float(np.min(radii)),
                "generatedMaxRadius": float(np.max(radii)),
                "generatedMeanToRequestedRatio": float(np.mean(radii)) / requested_radius,
                "nSpheres": len(radii),
                "withinSphereLimit": int(within_sphere_limit),
                "generationSeconds": generation_seconds,
                "contactGraphComponents": metrics["contactGraphComponents"],
                "largestComponentFraction": metrics["largestComponentFraction"],
                "isConnected": int(connected),
                "reference_volume": reference["volume"],
                "clump_volume": metrics["volume"],
                "volumeCoverageRatio": volume_coverage,
                "volumeRelativeErrorPercent": relative_error_percent(
                    metrics["volume"], reference["volume"]
                ),
                "withinVolumeCoverageInterval": int(within_volume_interval),
                "reference_surfaceArea": reference["surfaceArea"],
                "reference_convexHullVolume": reference["convexHullVolume"],
                "reference_sphericity": reference["sphericity"],
                "reference_convexity": reference["convexity"],
                "isEligible": int(
                    within_sphere_limit
                    and within_volume_interval
                    and connected
                ),
            }
            individual_rows.append(row)
            candidate_geometry[(rock["rockType"], divisor, realisation)] = (
                centers.copy(), radii.copy()
            )

            print(
                "    spheres={}, radius mean={:.8g} (requested {:.8g}), "
                "coverage={:.1f}%, components={}, time={:.1f}s, "
                "eligible={}".format(
                    len(radii),
                    float(np.mean(radii)),
                    requested_radius,
                    100.0 * volume_coverage,
                    metrics["contactGraphComponents"],
                    generation_seconds,
                    row["isEligible"],
                )
            )

if not individual_rows:
    raise RuntimeError("Every randomDensePack call failed; no results to save.")


# -----------------------------------------------------------------------------
# AGGREGATE MEAN AND SAMPLE STANDARD DEVIATION
# -----------------------------------------------------------------------------

def mean_std(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


summary_rows = []
for rock in rock_data:
    for divisor in DIVISORS_BY_ROCK[rock["rockType"]]:
        group = [
            row for row in individual_rows
            if row["rockType"] == rock["rockType"]
            and row["resolutionDivisor"] == divisor
        ]
        if not group:
            continue

        summary = {
            "rockType": rock["rockType"],
            "gtsFile": rock["gtsFile"],
            "resolutionDivisor": divisor,
            "rRelFuzz": RREL_FUZZ,
            "nRequestedRealisations": N_REALISATIONS,
            "nSuccessfulRealisations": len(group),
            "nEligibleRealisations": sum(row["isEligible"] for row in group),
            "eligibleFraction": sum(row["isEligible"] for row in group) / float(len(group)),
            "reference_volume": group[0]["reference_volume"],
            "reference_sphericity": group[0]["reference_sphericity"],
            "reference_convexity": group[0]["reference_convexity"],
        }

        aggregate_columns = [
            "nSpheres",
            "generatedMeanRadius",
            "generatedMeanToRequestedRatio",
            "generationSeconds",
            "volumeCoverageRatio",
            "volumeRelativeErrorPercent",
            "contactGraphComponents",
            "largestComponentFraction",
        ]
        for column in aggregate_columns:
            mean, std = mean_std([row[column] for row in group])
            summary[column + "_mean"] = mean
            summary[column + "_std"] = std
        summary_rows.append(summary)


with open(OUTPUT_INDIVIDUAL_CSV, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(individual_rows[0].keys()))
    writer.writeheader()
    writer.writerows(individual_rows)

with open(OUTPUT_SUMMARY_CSV, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

if failures:
    with open(OUTPUT_FAILURE_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(failures[0].keys()))
        writer.writeheader()
        writer.writerows(failures)


print("\nMEAN +/- SAMPLE STANDARD DEVIATION")
for summary in summary_rows:
    print(
        "  rock {}, d={}: n={:.1f}+/-{:.1f}, coverage={:.3f}+/-{:.3f}, "
        "eligible={}/{}".format(
            summary["rockType"],
            summary["resolutionDivisor"],
            summary["nSpheres_mean"],
            summary["nSpheres_std"],
            summary["volumeCoverageRatio_mean"],
            summary["volumeCoverageRatio_std"],
            summary["nEligibleRealisations"],
            summary["nSuccessfulRealisations"],
        )
    )


# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    for rock, axis in zip(rock_data, axes.flat):
        rows = sorted(
            [row for row in summary_rows if row["rockType"] == rock["rockType"]],
            key=lambda row: row["resolutionDivisor"],
        )
        divisors = [row["resolutionDivisor"] for row in rows]
        means = [row["volumeCoverageRatio_mean"] for row in rows]
        stds = [row["volumeCoverageRatio_std"] for row in rows]
        axis.errorbar(divisors, means, yerr=stds, marker="o", capsize=4)
        axis.axhspan(
            MIN_VOLUME_COVERAGE_RATIO,
            MAX_VOLUME_COVERAGE_RATIO,
            color="gray",
            alpha=0.18,
            label="Admissible interval",
        )
        axis.axhline(1.0, color="black", linestyle="--")
        axis.set_title("Rock {}".format(rock["rockType"]))
        axis.set_xlabel("Resolution divisor")
        axis.set_ylabel("Clump volume / GTS volume")
        axis.set_xticks(divisors)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Random-dense clump volume coverage: mean +/- SD")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(OUTPUT_VOLUME_PLOT, dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for rock, axis in zip(rock_data, axes.flat):
        rows = sorted(
            [row for row in summary_rows if row["rockType"] == rock["rockType"]],
            key=lambda row: row["resolutionDivisor"],
        )
        divisors = [row["resolutionDivisor"] for row in rows]
        means = [row["generationSeconds_mean"] for row in rows]
        stds = [row["generationSeconds_std"] for row in rows]
        axis.errorbar(divisors, means, yerr=stds, marker="o", capsize=4)
        axis.set_title("Rock {}".format(rock["rockType"]))
        axis.set_xlabel("Resolution divisor")
        axis.set_ylabel("Generation time [s]")
        axis.set_xticks(divisors)
        axis.grid(alpha=0.25)
    figure.suptitle("randomDensePack generation time: mean +/- SD")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(OUTPUT_RUNTIME_PLOT, dpi=220)
    plt.close(figure)

    # Direct sphere-count/resolution relationship. Error bars show the sample
    # standard deviation across the independently generated realisations.
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for rock, axis in zip(rock_data, axes.flat):
        rows = sorted(
            [row for row in summary_rows if row["rockType"] == rock["rockType"]],
            key=lambda row: row["resolutionDivisor"],
        )
        divisors = [row["resolutionDivisor"] for row in rows]
        means = [row["nSpheres_mean"] for row in rows]
        stds = [row["nSpheres_std"] for row in rows]
        axis.errorbar(
            divisors,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            linewidth=1.5,
            label=r"Mean $\pm$ sample SD",
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
        axis.legend(fontsize=8)
    figure.suptitle("Random-dense member-sphere count and clump resolution")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(OUTPUT_COUNT_VS_DIVISOR_PLOT, dpi=220)
    plt.close(figure)

    # Individual-realisation eligibility heatmaps. Cell color and text show
    # clump/reference volume ratio. Excluded and failed cells remain gray.
    eligible_ratios = [
        100.0 * row["volumeCoverageRatio"]
        for row in individual_rows if row["isEligible"]
    ]
    color_min = min(eligible_ratios) if eligible_ratios else 90.0
    color_max = max(eligible_ratios) if eligible_ratios else 110.0
    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad(color="lightgray")
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    last_image = None
    for rock, axis in zip(rock_data, axes.flat):
        divisors = list(DIVISORS_BY_ROCK[rock["rockType"]])
        matrix = np.full((N_REALISATIONS, len(divisors)), np.nan, dtype=float)
        for row in individual_rows:
            if row["rockType"] != rock["rockType"] or not row["isEligible"]:
                continue
            divisor_index = divisors.index(row["resolutionDivisor"])
            realisation_index = row["realisation"] - 1
            matrix[realisation_index, divisor_index] = (
                100.0 * row["volumeCoverageRatio"]
            )
        last_image = axis.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            origin="lower",
            cmap=color_map,
            vmin=color_min,
            vmax=color_max,
        )
        axis.set_xticks(range(len(divisors)))
        axis.set_xticklabels(divisors)
        axis.set_yticks(range(N_REALISATIONS))
        axis.set_yticklabels(range(1, N_REALISATIONS + 1))
        axis.set_xlabel("Resolution divisor")
        axis.set_ylabel("Realisation")
        axis.set_title("Rock {}".format(rock["rockType"]))
        for realisation_index in range(N_REALISATIONS):
            for divisor_index in range(len(divisors)):
                value = matrix[realisation_index, divisor_index]
                if np.isnan(value):
                    continue
                axis.text(
                    divisor_index,
                    realisation_index,
                    "{:.0f}".format(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )
    if last_image is not None:
        figure.colorbar(
            last_image,
            ax=list(axes.flat),
            shrink=0.80,
            label="Clump / reference volume [%]",
        )
    figure.legend(
        handles=[Patch(
            facecolor="lightgray",
            edgecolor="gray",
            label="Excluded by admissibility criteria or generation failure",
        )],
        loc="upper center",
        frameon=True,
    )
    figure.suptitle("Eligibility of random-dense clump realisations", y=0.95)
    figure.subplots_adjust(
        left=0.07, right=0.88, bottom=0.08, top=0.88,
        wspace=0.22, hspace=0.28,
    )
    figure.savefig(OUTPUT_ELIGIBILITY_HEATMAP, dpi=220)
    plt.close(figure)

    print("Saved plot:", OUTPUT_VOLUME_PLOT)
    print("Saved plot:", OUTPUT_RUNTIME_PLOT)
    print("Saved plot:", OUTPUT_COUNT_VS_DIVISOR_PLOT)
    print("Saved plot:", OUTPUT_ELIGIBILITY_HEATMAP)
except ImportError:
    print("Matplotlib is unavailable; CSV results were still saved.")


# -----------------------------------------------------------------------------
# PROVISIONAL REPRESENTATIVE REALISATION PER ROCK: YADE + VTK
# -----------------------------------------------------------------------------

best_by_rock = {}
for rock in rock_data:
    all_candidates = [
        row for row in individual_rows
        if row["rockType"] == rock["rockType"]
    ]
    eligible_candidates = [row for row in all_candidates if row["isEligible"]]
    if eligible_candidates:
        eligible_candidates.sort(key=lambda row: (
            abs(row["volumeRelativeErrorPercent"]),
            row["nSpheres"],
            row["generationSeconds"],
        ))
        selected = eligible_candidates[0]
        selected["vtkSelectionStatus"] = "eligible"
        best_by_rock[rock["rockType"]] = selected
        continue

    # randomDensePack normally produces touching rather than overlapping
    # spheres, so its raw union volume may fall below the eligibility interval.
    # Still export one clearly labelled fallback so the failed geometry can be
    # inspected in ParaView instead of silently producing no visual output.
    under_limit = [
        row for row in all_candidates if row["withinSphereLimit"]
    ]
    fallback_candidates = under_limit if under_limit else all_candidates
    fallback_candidates.sort(key=lambda row: (
        abs(row["volumeRelativeErrorPercent"]),
        row["nSpheres"],
        row["generationSeconds"],
    ))
    if fallback_candidates:
        selected = fallback_candidates[0]
        selected["vtkSelectionStatus"] = "fallback_not_eligible"
        best_by_rock[rock["rockType"]] = selected

print("\nPROVISIONAL REPRESENTATIVE REALISATION PER ROCK")
for rock in rock_data:
    selected = best_by_rock.get(rock["rockType"])
    if selected is None:
        print("  {}: none".format(rock["gtsFile"]))
        continue
    print(
        "  {}: status={}, d={}, realisation={}, seed={}, spheres={}, "
        "coverage={:.1f}%".format(
            rock["gtsFile"],
            selected["vtkSelectionStatus"],
            selected["resolutionDivisor"],
            selected["realisation"],
            selected["seed"],
            selected["nSpheres"],
            100.0 * selected["volumeCoverageRatio"],
        )
    )

rockTypeByBodyId = {}
divisorByBodyId = {}
realisationByBodyId = {}

for rock_index, rock in enumerate(rock_data):
    selected = best_by_rock.get(rock["rockType"])
    if selected is None:
        continue
    centers, radii = candidate_geometry[
        (
            rock["rockType"],
            selected["resolutionDivisor"],
            selected["realisation"],
        )
    ]
    packing_center = 0.5 * (
        np.min(centers - radii[:, None], axis=0)
        + np.max(centers + radii[:, None], axis=0)
    )
    target_center = np.asarray([rock_index * DISPLAY_SPACING, 0.0, 0.0])
    translated_centers = centers + (target_center - packing_center)

    bodies = [
        utils.sphere(
            tuple(center),
            float(radius),
            material=material_id,
            color=colors[rock_index],
        )
        for center, radius in zip(translated_centers, radii)
    ]
    member_ids = O.bodies.append(bodies)
    O.bodies.clump(member_ids)
    for member_id in member_ids:
        rockTypeByBodyId[member_id] = rock["rockType"]
        divisorByBodyId[member_id] = selected["resolutionDivisor"]
        realisationByBodyId[member_id] = selected["realisation"]

O.save(OUTPUT_SCENE)

if rockTypeByBodyId:
    vtk_exporter = export.VTKExporter(VTK_OUTPUT_PREFIX)
    vtk_exporter.exportSpheres(
        what={
            "rockType": "rockTypeByBodyId.get(b.id,0)",
            "resolutionDivisor": "divisorByBodyId.get(b.id,0)",
            "realisation": "realisationByBodyId.get(b.id,0)",
        },
        numLabel=0,
    )
    print("Saved VTK sphere export with prefix:", VTK_OUTPUT_PREFIX)
else:
    print("Skipped VTK export: no eligible realisation was found.")

print("Saved individual results:", OUTPUT_INDIVIDUAL_CSV)
print("Saved mean/SD summary:", OUTPUT_SUMMARY_CSV)
if failures:
    print("Saved generation failures:", OUTPUT_FAILURE_CSV)
print("Saved YADE scene:", OUTPUT_SCENE)
