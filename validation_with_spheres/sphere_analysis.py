"""Per-run and aggregate analysis for square-bed sphere packings."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from scipy.special import sph_harm_y as _sph_harm_y

    def spherical_harmonic(l: int, m: int, theta, phi):
        return _sph_harm_y(l, m, theta, phi)

except ImportError:  # SciPy < 1.15
    from scipy.special import sph_harm as _legacy_sph_harm

    def spherical_harmonic(l: int, m: int, theta, phi):
        return _legacy_sph_harm(m, l, phi, theta)


CONTACT_TOLERANCE_DIAMETERS = 0.01
LATERAL_SAMPLES_PER_DIAMETER = 10
HAMZAH_DELTA_BIN_WIDTH = 0.10
HAMZAH_CORNER_DELTA_MAX = 0.110
HAMZAH_OUTER_CORE_DELTA_MIN = 0.747
HAMZAH_INNER_CORE_DELTA_MIN = 1.110
HAMZAH_INFINITE_BED_POROSITY = 0.400

Q_REFERENCE_STRUCTURES = {
    "BCC (n=8)": (0.509, 0.629),
    "BCC (n=14)": (0.036, 0.511),
    "FCC (n=12)": (0.190, 0.575),
    "HCP (n=12)": (0.097, 0.484),
    "Icosahedral (n=12)": (0.000, 0.663),
    "Simple cubic (n=6)": (0.764, 0.354),
}

PACKING_STYLE = {
    "hexa": {"label": "Hexagonal close packing", "color": "#D55E00", "marker": "^"},
    "ortho": {"label": "Simple cubic packing", "color": "#009E73", "marker": "s"},
    "random": {"label": "Random gravity packing", "color": "#0072B2", "marker": "o"},
}


def as_boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def beavers_sparrow_porosity(np_ratio):
    """Beavers & Sparrow (1973), Eq. (8), with De=W for a square."""
    np_ratio = np.asarray(np_ratio, dtype=float)
    epsilon_infinity = 0.368
    epsilon_wall = 0.476
    return epsilon_infinity * (
        1.0 + 2.0 / np_ratio * (epsilon_wall / epsilon_infinity - 1.0)
    )


def dixon_porosity(np_ratio):
    """Dixon (1988), reproduced as Hamzah Eq. (17)/Eppinger Eq. (5)."""
    np_ratio = np.asarray(np_ratio, dtype=float)
    return 0.4 + 0.05 / np_ratio + 0.412 / np_ratio ** 2


def hamzah_zone1_relation(delta, epsilon_bulk=HAMZAH_INFINITE_BED_POROSITY):
    """Hamzah et al. (2020), Eq. (18)."""
    delta = np.asarray(delta, dtype=float)
    near = 2.14 * delta ** 2 - 2.53 * delta + 1.0
    far = (
        epsilon_bulk
        + 0.29 * np.exp(-0.6 * delta) * np.cos(2.3 * np.pi * (delta - 0.16))
        + 0.15 * np.exp(-0.9 * delta)
    )
    return np.where(delta <= 0.637, near, far)


def hamzah_zone2_relation(delta, epsilon_bulk=HAMZAH_INFINITE_BED_POROSITY):
    """Hamzah et al. (2020), Eq. (19); Zone 3 is deliberately excluded."""
    delta = np.asarray(delta, dtype=float)
    near = 2.14 * delta ** 2 - 2.53 * delta + 1.0
    outer_core = (
        epsilon_bulk
        + 0.29 * np.exp(-0.6 * delta) * np.cos(2.3 * np.pi * (delta - 0.16))
        + 0.15 * np.exp(-0.9 * delta)
    )
    inner_core = (
        epsilon_bulk
        + 0.35 * np.exp(-0.6 * delta) * np.cos(1.68 * np.pi * (delta - 0.06))
        + 0.01 * np.exp(-1.6 * delta)
    )
    return np.select(
        [delta <= HAMZAH_OUTER_CORE_DELTA_MIN,
         delta <= HAMZAH_INNER_CORE_DELTA_MIN],
        [near, outer_core],
        default=inner_core,
    )


def clipped_sphere_volume(z_center, radius, z_min, z_max):
    """Exact volume of spheres between two horizontal clipping planes."""
    z_center = np.asarray(z_center, dtype=float)
    radius = np.asarray(radius, dtype=float)
    lower = np.maximum(-radius, z_min - z_center)
    upper = np.minimum(radius, z_max - z_center)
    valid = upper > lower
    primitive_upper = np.pi * (radius ** 2 * upper - upper ** 3 / 3.0)
    primitive_lower = np.pi * (radius ** 2 * lower - lower ** 3 / 3.0)
    return np.where(valid, primitive_upper - primitive_lower, 0.0)


def build_contacts(positions, radii, particle_ids, diameter):
    if len(positions) < 2:
        return np.empty((0, 2), dtype=int), np.empty(0), np.empty(0)
    maximum_radius = float(np.max(radii))
    cutoff = 2.0 * maximum_radius + CONTACT_TOLERANCE_DIAMETERS * diameter
    candidate_pairs = cKDTree(positions).query_pairs(cutoff, output_type="ndarray")
    if candidate_pairs.size == 0:
        return np.empty((0, 2), dtype=int), np.empty(0), np.empty(0)
    differences = positions[candidate_pairs[:, 1]] - positions[candidate_pairs[:, 0]]
    distances = np.linalg.norm(differences, axis=1)
    allowed = (
        radii[candidate_pairs[:, 0]] + radii[candidate_pairs[:, 1]]
        + CONTACT_TOLERANCE_DIAMETERS * diameter
    )
    keep = (distances > 0.0) & (distances <= allowed)
    pairs = candidate_pairs[keep]
    differences = differences[keep]
    distances = distances[keep]
    # A contact line is undirected, so angle from vertical lies in [0, 90].
    cos_vertical = np.clip(np.abs(differences[:, 2]) / distances, 0.0, 1.0)
    angles = np.degrees(np.arccos(cos_vertical))
    overlaps = (
        radii[pairs[:, 0]] + radii[pairs[:, 1]] - distances
    ) / diameter
    return pairs, angles, overlaps


def local_bond_order(positions, pairs, particle_ids, q_values=(4, 6)):
    neighbours = defaultdict(list)
    for first, second in pairs:
        neighbours[int(first)].append(int(second))
        neighbours[int(second)].append(int(first))

    output = {q: np.full(len(positions), np.nan, dtype=float) for q in q_values}
    for particle_index in range(len(positions)):
        local = neighbours.get(particle_index, [])
        if not local:
            continue
        vectors = positions[np.asarray(local, dtype=int)] - positions[particle_index]
        distances = np.linalg.norm(vectors, axis=1)
        valid = distances > 0.0
        vectors = vectors[valid]
        distances = distances[valid]
        if len(distances) == 0:
            continue
        theta = np.arccos(np.clip(vectors[:, 2] / distances, -1.0, 1.0))
        phi = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2.0 * np.pi)
        for q in q_values:
            qlm = np.asarray([
                np.mean(spherical_harmonic(q, m, theta, phi))
                for m in range(-q, q + 1)
            ])
            output[q][particle_index] = math.sqrt(
                4.0 * math.pi / (2 * q + 1) * float(np.sum(np.abs(qlm) ** 2))
            )
    return output


def axial_lateral_porosity_field(positions, radii, x_coordinates, y_coordinates,
                                  z_min, z_max):
    """Exact vertical-line integration sampled on a lateral grid."""
    height = z_max - z_min
    xx, yy = np.meshgrid(x_coordinates, y_coordinates, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    tree = cKDTree(positions[:, :2])
    max_radius = float(np.max(radii))
    nearby = tree.query_ball_point(points, r=max_radius * (1.0 + 1.0e-12))
    solid_lengths = np.zeros(len(points), dtype=float)
    for point_index, sphere_indices in enumerate(nearby):
        if not sphere_indices:
            continue
        indices = np.asarray(sphere_indices, dtype=int)
        delta_xy = positions[indices, :2] - points[point_index]
        radial_squared = np.einsum("ij,ij->i", delta_xy, delta_xy)
        inside = radial_squared < radii[indices] ** 2
        if not np.any(inside):
            continue
        indices = indices[inside]
        radial_squared = radial_squared[inside]
        half_chord = np.sqrt(np.maximum(radii[indices] ** 2 - radial_squared, 0.0))
        lower = np.maximum(positions[indices, 2] - half_chord, z_min)
        upper = np.minimum(positions[indices, 2] + half_chord, z_max)
        solid_lengths[point_index] = np.sum(np.maximum(upper - lower, 0.0))
    porosity = 1.0 - solid_lengths / height
    return np.clip(porosity.reshape(xx.shape), 0.0, 1.0), xx, yy


def wall_normal_slab_profiles(field, bin_edges):
    """Return finite-width wall-normal porosity profiles in both directions.

    ``field[y, x]`` contains vertical-line porosity sampled at the centre of
    every square lateral cell.  A single centreline is unsuitable for a regular
    packing: depending on lattice parity, it can pass entirely through a row of
    spheres or entirely through a void channel.  Here, each profile value
    instead represents a full-width slab between adjacent bin edges.  Midpoint
    quadrature averages all cells across the complete transverse direction.
    """
    bin_edges = np.asarray(bin_edges, dtype=float)
    field = np.asarray(field, dtype=float)
    expected_shape = (len(bin_edges) - 1, len(bin_edges) - 1)
    if field.shape != expected_shape:
        raise ValueError(
            f"Porosity field has shape {field.shape}; expected {expected_shape}"
        )
    if len(bin_edges) < 2 or np.any(np.diff(bin_edges) <= 0.0):
        raise ValueError("Lateral bin edges must be strictly increasing")

    side = float(bin_edges[-1] - bin_edges[0])
    if side <= 0.0:
        raise ValueError("The lateral domain must have positive width")

    slab_widths = np.diff(bin_edges)
    transverse_weights = slab_widths / side
    left_to_right = np.sum(field * transverse_weights[:, None], axis=0)
    bottom_to_top = np.sum(field * transverse_weights[None, :], axis=1)
    slab_midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return (
        slab_midpoints,
        slab_widths,
        np.clip(left_to_right, 0.0, 1.0),
        np.clip(bottom_to_top, 0.0, 1.0),
    )


def hamzah_zone_coordinates(xx, yy, side, diameter, np_ratio):
    distance_to_vertical_wall = np.minimum(xx, side - xx)
    distance_to_horizontal_wall = np.minimum(yy, side - yy)
    distance_from_center = np.hypot(xx - 0.5 * side, yy - 0.5 * side)
    corner_to_center = side / math.sqrt(2.0)
    radius = 0.5 * diameter
    width_scale = 0.67 if np.isclose(np_ratio, 3.0, atol=0.05, rtol=0.0) else 1.0
    diagonal_band = (
        np.abs(distance_to_vertical_wall - distance_to_horizontal_wall)
        <= width_scale * radius
    )
    delta_zone1 = np.minimum(
        distance_to_vertical_wall, distance_to_horizontal_wall
    ) / diameter
    delta_zone2 = np.maximum(
        0.0, (corner_to_center - distance_from_center) / diameter
    )
    zone = np.ones(xx.shape, dtype=np.int8)
    zone[diagonal_band] = 2
    zone[diagonal_band & (delta_zone2 <= HAMZAH_CORNER_DELTA_MAX)] = 3
    return zone, delta_zone1, delta_zone2, width_scale


def binned_porosity_profile(zone_id, zone, delta, field):
    mask = zone == zone_id
    if not np.any(mask):
        return pd.DataFrame()
    maximum = float(np.max(delta[mask]))
    number_of_bins = max(1, int(math.ceil(maximum / HAMZAH_DELTA_BIN_WIDTH)))
    edges = np.arange(number_of_bins + 1, dtype=float) * HAMZAH_DELTA_BIN_WIDTH
    index = np.digitize(delta, edges[1:-1], right=False)
    rows = []
    for bin_index in range(number_of_bins):
        selected = mask & (index == bin_index)
        count = int(np.count_nonzero(selected))
        if count == 0:
            continue
        lower = float(edges[bin_index])
        upper = float(edges[bin_index + 1])
        middle = 0.5 * (lower + upper)
        values = field[selected]
        rows.append({
            "zoneId": zone_id,
            "zoneName": "wall-adjacent" if zone_id == 1 else "diagonal",
            "deltaLower": lower,
            "deltaUpper": upper,
            "deltaMid": middle,
            "numberOfLateralSamples": count,
            "localPorosity": float(np.mean(values)),
            "lateralSampleStd": float(np.std(values, ddof=0)),
            "hamzahPorosity": float(
                hamzah_zone1_relation(middle) if zone_id == 1
                else hamzah_zone2_relation(middle)
            ),
        })
    return pd.DataFrame(rows)


def analyse_one_run(run_dir, axial_trim_diameters=2.0):
    run_dir = Path(run_dir)
    with (run_dir / "simulation_summary.json").open("r", encoding="utf-8") as handle:
        simulation = json.load(handle)
    particles = pd.read_csv(run_dir / "particles.csv")
    if particles.empty:
        raise RuntimeError("particles.csv contains no spheres")

    positions = particles[["x", "y", "z"]].to_numpy(dtype=float)
    radii = particles["radius"].to_numpy(dtype=float)
    particle_ids = particles["particleId"].to_numpy(dtype=int)
    diameter = float(simulation["sphereDiameter"])
    side = float(simulation["squareSideLength"])
    np_ratio = float(simulation["npRatio"])
    packing_type = str(simulation["packingType"])

    top_surfaces = positions[:, 2] + radii
    if packing_type == "random":
        robust_top = float(np.quantile(top_surfaces, 0.995))
    else:
        robust_top = float(np.max(top_surfaces))
    # A 500-sphere cap necessarily makes the largest-Np beds only a few layers
    # tall.  Use the requested two-diameter end trim whenever possible, and
    # reduce it symmetrically only when needed to retain at least two diameters
    # of analysable bed.
    minimum_analysis_height = 2.0 * diameter
    maximum_feasible_trim = max(
        0.0,
        0.5 * (robust_top - minimum_analysis_height) / diameter,
    )
    effective_axial_trim_diameters = min(
        axial_trim_diameters,
        0.95 * maximum_feasible_trim,
    )
    z_min = effective_axial_trim_diameters * diameter
    z_max = robust_top - effective_axial_trim_diameters * diameter
    if z_max <= z_min:
        raise RuntimeError(
            f"Analysed bed is too short after axial trimming: [{z_min}, {z_max}]"
        )

    sphere_volume_inside = clipped_sphere_volume(
        positions[:, 2], radii, z_min, z_max
    )
    domain_volume = side ** 2 * (z_max - z_min)
    global_porosity = 1.0 - float(np.sum(sphere_volume_inside)) / domain_volume

    pairs, contact_angles, overlaps = build_contacts(
        positions, radii, particle_ids, diameter
    )
    contact_count = np.zeros(len(positions), dtype=int)
    for first, second in pairs:
        contact_count[first] += 1
        contact_count[second] += 1

    wall_tolerance = CONTACT_TOLERANCE_DIAMETERS * diameter
    wall_contacts = (
        (positions[:, 0] - radii <= wall_tolerance).astype(int)
        + (side - positions[:, 0] - radii <= wall_tolerance).astype(int)
        + (positions[:, 1] - radii <= wall_tolerance).astype(int)
        + (side - positions[:, 1] - radii <= wall_tolerance).astype(int)
        + (positions[:, 2] - radii <= wall_tolerance).astype(int)
    )
    lateral_clearance = np.minimum.reduce([
        positions[:, 0] - radii,
        side - positions[:, 0] - radii,
        positions[:, 1] - radii,
        side - positions[:, 1] - radii,
    ])
    axial_clearance = np.minimum(positions[:, 2] - z_min, z_max - positions[:, 2])
    bulk_mask = (
        (lateral_clearance >= 0.75 * diameter)
        & (axial_clearance >= 0.75 * diameter)
    )

    bond_order = local_bond_order(positions, pairs, particle_ids)
    if packing_type == "ortho":
        q_selection = bulk_mask & (contact_count == 6)
    elif packing_type == "hexa":
        q_selection = bulk_mask & (contact_count == 12)
    else:
        q_selection = bulk_mask & (contact_count >= 3)
    if not np.any(q_selection):
        q_selection = (contact_count >= 3)

    contacts = pd.DataFrame({
        "particleIdA": particle_ids[pairs[:, 0]] if len(pairs) else np.array([], dtype=int),
        "particleIdB": particle_ids[pairs[:, 1]] if len(pairs) else np.array([], dtype=int),
        "contactAngleDegFromVertical": contact_angles,
        "overlapByDiameter": overlaps,
    })
    contacts.to_csv(run_dir / "contacts.csv", index=False)

    coordination = particles[["particleId", "x", "y", "z"]].copy()
    coordination["particleParticleContacts"] = contact_count
    coordination["wallAndFloorContacts"] = wall_contacts
    coordination["totalContactsIncludingWalls"] = contact_count + wall_contacts
    coordination["bulkParticle"] = bulk_mask
    coordination["includedInQMean"] = q_selection
    coordination["localQ4ContactNeighbours"] = bond_order[4]
    coordination["localQ6ContactNeighbours"] = bond_order[6]
    coordination.to_csv(run_dir / "coordination_per_particle.csv", index=False)

    intervals = max(2, int(math.ceil(np_ratio * LATERAL_SAMPLES_PER_DIAMETER)))
    lateral_bin_edges = np.linspace(0.0, side, intervals + 1)
    # Keep the original edge-sampled field for the Hamzah zone comparison so
    # this wall-profile correction does not silently change those results.
    field, xx, yy = axial_lateral_porosity_field(
        positions, radii, lateral_bin_edges, lateral_bin_edges, z_min, z_max
    )

    # Use cell centres for the finite-volume wall-normal slabs.  This avoids
    # sampling the zero-area tangent planes of an ordered lattice.
    lateral_cell_centres = 0.5 * (
        lateral_bin_edges[:-1] + lateral_bin_edges[1:]
    )
    wall_field, _, _ = axial_lateral_porosity_field(
        positions, radii, lateral_cell_centres, lateral_cell_centres, z_min, z_max
    )
    (
        slab_midpoints,
        slab_widths,
        left_to_right,
        bottom_to_top,
    ) = wall_normal_slab_profiles(wall_field, lateral_bin_edges)
    mean_across_directions = 0.5 * (left_to_right + bottom_to_top)
    wall_profile = pd.DataFrame({
        "distanceFromStartingWall": slab_midpoints,
        "distanceFromStartingWallByDiameter": slab_midpoints / diameter,
        "slabWidth": slab_widths,
        "slabWidthByDiameter": slab_widths / diameter,
        "leftToRightPorosity": left_to_right,
        "bottomToTopPorosity": bottom_to_top,
        "meanAcrossTwoDirections": mean_across_directions,
        # Retain the former column names so existing downstream scripts remain
        # compatible.  Their values now contain plane/slab averages, not
        # centreline samples.
        "meanAcrossTwoCentrelines": mean_across_directions,
        "minimumAcrossTwoCentrelines": np.minimum(
            left_to_right, bottom_to_top
        ),
        "maximumAcrossTwoCentrelines": np.maximum(
            left_to_right, bottom_to_top
        ),
        "profileMethod": "full-width finite-slab volume average",
    })
    wall_profile.to_csv(run_dir / "wall_to_wall_porosity.csv", index=False)

    zones, delta_zone1, delta_zone2, zone2_width_scale = hamzah_zone_coordinates(
        xx, yy, side, diameter, np_ratio
    )
    zone_profiles = pd.concat([
        binned_porosity_profile(1, zones, delta_zone1, field),
        binned_porosity_profile(2, zones, delta_zone2, field),
    ], ignore_index=True)
    zone_profiles.to_csv(run_dir / "hamzah_zone_profiles.csv", index=False)

    q4_values = bond_order[4][q_selection]
    q6_values = bond_order[6][q_selection]
    q4_values = q4_values[np.isfinite(q4_values)]
    q6_values = q6_values[np.isfinite(q6_values)]
    bulk_contacts = contact_count[bulk_mask]
    summary = {
        "packingType": packing_type,
        "npRatio": np_ratio,
        "realization": int(simulation["realization"]),
        "randomSeed": int(simulation["randomSeed"]),
        "simulationStatus": simulation["status"],
        "settled": bool(simulation["settled"]),
        "numberOfSpheres": len(particles),
        "sphereDiameter": diameter,
        "squareSideLength": side,
        "robustBedTop": robust_top,
        "analysisZMin": z_min,
        "analysisZMax": z_max,
        "analysisHeight": z_max - z_min,
        "requestedAxialTrimDiameters": axial_trim_diameters,
        "axialTrimDiameters": effective_axial_trim_diameters,
        "globalPorosity": global_porosity,
        "beaversSparrowPorosity": float(beavers_sparrow_porosity(np_ratio)),
        "dixonPorosity": float(dixon_porosity(np_ratio)),
        "globalMinusBeaversSparrow": global_porosity - float(beavers_sparrow_porosity(np_ratio)),
        "globalMinusDixon": global_porosity - float(dixon_porosity(np_ratio)),
        "numberOfParticleParticleContacts": len(pairs),
        "meanCoordinationNumber": float(np.mean(contact_count)),
        "meanBulkCoordinationNumber": (
            float(np.mean(bulk_contacts)) if len(bulk_contacts) else np.nan
        ),
        "meanWallAndFloorContacts": float(np.mean(wall_contacts)),
        "meanTotalContactsIncludingWalls": float(np.mean(contact_count + wall_contacts)),
        "meanContactAngleDegFromVertical": (
            float(np.mean(contact_angles)) if len(contact_angles) else np.nan
        ),
        "medianContactAngleDegFromVertical": (
            float(np.median(contact_angles)) if len(contact_angles) else np.nan
        ),
        "stdContactAngleDegFromVertical": (
            float(np.std(contact_angles, ddof=1)) if len(contact_angles) > 1 else 0.0
        ),
        "meanLocalQ4ContactNeighbours": (
            float(np.mean(q4_values)) if len(q4_values) else np.nan
        ),
        "stdLocalQ4ContactNeighbours": (
            float(np.std(q4_values, ddof=1)) if len(q4_values) > 1 else 0.0
        ),
        "meanLocalQ6ContactNeighbours": (
            float(np.mean(q6_values)) if len(q6_values) else np.nan
        ),
        "stdLocalQ6ContactNeighbours": (
            float(np.std(q6_values, ddof=1)) if len(q6_values) > 1 else 0.0
        ),
        "numberOfParticlesInQMean": int(np.count_nonzero(q_selection)),
        "contactToleranceDiameters": CONTACT_TOLERANCE_DIAMETERS,
        "zone2WidthScale": zone2_width_scale,
        "porosityMethod": "exact clipped sphere volumes and exact vertical-line chords",
        "qNeighborDefinition": "particle contacts within 0.01 sphere diameter",
    }
    pd.DataFrame([summary]).to_csv(run_dir / "analysis_summary.csv", index=False)
    return summary


def load_run_tables(tasks, filename, settled_random_only=False):
    frames = []
    for task in tasks:
        summary_path = task.run_dir / "analysis_summary.csv"
        path = task.run_dir / filename
        if not summary_path.exists() or not path.exists():
            continue
        summary = pd.read_csv(summary_path).iloc[0]
        if settled_random_only and task.packing_type == "random":
            if not as_boolean(summary["settled"]):
                continue
        frame = pd.read_csv(path)
        frame.insert(0, "packingType", task.packing_type)
        frame.insert(1, "npRatio", task.np_ratio)
        frame.insert(2, "realization", task.realization)
        frame.insert(3, "randomSeed", task.seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def random_group_statistics(results):
    random = results[
        (results["packingType"] == "random") & results["settled"].map(as_boolean)
    ].copy()
    # List-style aggregation works with both the older pandas shipped by
    # Ubuntu/YADE 2022.01a and current pandas releases.  Older SeriesGroupBy
    # versions reject keyword named aggregation (mean="mean", ...).
    grouped = (
        random.groupby("npRatio")["globalPorosity"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci95"] = 1.96 * grouped["sem"]
    grouped["beaversSparrowPorosity"] = beavers_sparrow_porosity(grouped["npRatio"])
    grouped["dixonPorosity"] = dixon_porosity(grouped["npRatio"])
    return grouped


def plot_literature_porosity(grouped, output_dir, relation, relation_label, filename, title):
    x_curve = np.linspace(2.0, 10.25, 500)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(x_curve, relation(x_curve), color="black", linewidth=2.0, label=relation_label)
    ax.errorbar(
        grouped["npRatio"], grouped["mean"], yerr=grouped["std"],
        fmt="o", markersize=5.5, capsize=3, color=PACKING_STYLE["random"]["color"],
        markeredgecolor="black", markeredgewidth=0.5,
        label="Random DEM: mean $\\pm$ SD (50 requested)", zorder=4,
    )
    ax.set_xlabel(r"Square bed-size ratio $N_p=W/d$ [-]")
    ax.set_ylabel(r"Global porosity $\varepsilon$ [-]")
    ax.set_xlim(2.8, 10.2)
    ax.set_ylim(0.34, 0.52)
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_beavers_dixon_comparison(grouped, output_dir):
    """Plot random DEM porosity with both literature correlations."""
    x_data = grouped["npRatio"].to_numpy(dtype=float)
    mean = grouped["mean"].to_numpy(dtype=float)
    sample_std = grouped["std"].fillna(0.0).to_numpy(dtype=float)

    padding = max(0.2, 0.03 * (x_data.max() - x_data.min()))
    x_curve = np.linspace(
        x_data.min() - padding,
        x_data.max() + padding,
        600,
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.plot(
        x_curve,
        beavers_sparrow_porosity(x_curve),
        color="black",
        linewidth=2.0,
        label="Beavers-Sparrow Eq. (8)",
    )
    ax.plot(
        x_curve,
        dixon_porosity(x_curve),
        color="#D55E00",
        linewidth=2.0,
        linestyle="--",
        label="Dixon Eq. (17)",
    )
    ax.errorbar(
        x_data,
        mean,
        yerr=sample_std,
        fmt="o",
        markersize=6.0,
        capsize=3.0,
        color=PACKING_STYLE["random"]["color"],
        markeredgecolor="black",
        markeredgewidth=0.6,
        label="Random DEM: mean $\\pm$ sample SD",
        zorder=4,
    )

    ax.set_xlabel(r"Square bed-size ratio $N_p=W/d$ [-]")
    ax.set_ylabel(r"Global porosity $\varepsilon$ [-]")
    ax.set_title("Random square-bed porosity and literature correlations")
    ax.set_xlim(x_curve.min(), x_curve.max())

    all_y = np.concatenate((
        mean - sample_std,
        mean + sample_std,
        beavers_sparrow_porosity(x_curve),
        dixon_porosity(x_curve),
    ))
    y_padding = max(0.01, 0.08 * (all_y.max() - all_y.min()))
    ax.set_ylim(
        max(0.0, all_y.min() - y_padding),
        min(1.0, all_y.max() + y_padding),
    )
    ax.grid(True, alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "beavers_dixon_random_porosity.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_all_packings(results, output_dir):
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for packing_type in ("random", "hexa", "ortho"):
        style = PACKING_STYLE[packing_type]
        subset = results[results["packingType"] == packing_type].copy()
        if packing_type == "random":
            subset = subset[subset["settled"].map(as_boolean)]
        grouped = (
            subset.groupby("npRatio")["globalPorosity"]
            .agg(["mean", "std"])
            .reset_index()
        )
        error = grouped["std"].fillna(0.0)
        ax.errorbar(
            grouped["npRatio"], grouped["mean"], yerr=error,
            marker=style["marker"], color=style["color"], linewidth=1.5,
            capsize=2.5, label=style["label"],
        )
    ax.axhline(1.0 - math.pi / (3.0 * math.sqrt(2.0)), color="#D55E00",
               linestyle=":", linewidth=1.1, label="Infinite HCP porosity 0.25952")
    ax.axhline(1.0 - math.pi / 6.0, color="#009E73",
               linestyle=":", linewidth=1.1, label="Infinite SC porosity 0.47640")
    ax.set_xlabel(r"Square bed-size ratio $N_p=W/d$ [-]")
    ax.set_ylabel(r"Measured global porosity $\varepsilon$ [-]")
    ax.set_title("Square-bed porosity: regular and random sphere packings")
    ax.set_ylim(0.20, 0.55)
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "global_porosity_all_packings.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_hamzah_zone_panels(zone_profiles, output_dir, zone_id):
    random = zone_profiles[
        (zone_profiles["packingType"] == "random")
        & (zone_profiles["zoneId"] == zone_id)
    ].copy()
    sizes = sorted(random["npRatio"].unique())
    fig, axes = plt.subplots(5, 3, figsize=(13.0, 17.0), sharey=True)
    axes = axes.ravel()
    for axis, np_ratio in zip(axes, sizes):
        subset = random[np.isclose(random["npRatio"], np_ratio)]
        grouped = (
            subset.groupby("deltaMid")["localPorosity"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        if grouped.empty:
            axis.set_visible(False)
            continue
        x = grouped["deltaMid"].to_numpy(dtype=float)
        mean = grouped["mean"].to_numpy(dtype=float)
        std = grouped["std"].fillna(0.0).to_numpy(dtype=float)
        relation_x_min = 0.0 if zone_id == 1 else HAMZAH_CORNER_DELTA_MAX
        relation_x = np.linspace(relation_x_min, max(float(x.max()), 1.2), 350)
        relation_y = (
            hamzah_zone1_relation(relation_x) if zone_id == 1
            else hamzah_zone2_relation(relation_x)
        )
        axis.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                          color="#56B4E9", alpha=0.22)
        axis.plot(x, mean, "o-", color="#0072B2", markersize=3.0,
                  linewidth=1.2, label="Random DEM mean $\\pm$ SD")
        axis.plot(relation_x, relation_y, "--", color="#D55E00", linewidth=1.5,
                  label=f"Hamzah Eq. ({18 if zone_id == 1 else 19})")
        axis.set_title(rf"$N_p={np_ratio:g}$")
        axis.set_xlim(relation_x_min, max(1.25, float(x.max()) + 0.05))
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, alpha=0.18)
        if np.isclose(np_ratio, 3.0):
            axis.text(0.98, 0.03, "published-fit caveat at $N_p=3$",
                      transform=axis.transAxes, ha="right", va="bottom", fontsize=7)
    for axis in axes[len(sizes):]:
        axis.set_visible(False)
    for row in range(5):
        axes[row * 3].set_ylabel("Local porosity [-]")
    for axis in axes[-3:]:
        if axis.get_visible():
            axis.set_xlabel(r"Relative lateral distance $\delta$ [-]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    zone_name = "wall-adjacent" if zone_id == 1 else "diagonal"
    fig.suptitle(
        f"Hamzah Zone {zone_id} ({zone_name}): 50-realization random-pack comparison",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.975))
    fig.savefig(output_dir / f"hamzah_zone{zone_id}_comparison.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_wall_profiles(wall_profiles, output_dir):
    sizes = sorted(wall_profiles["npRatio"].unique())
    fig, axes = plt.subplots(5, 3, figsize=(13.0, 17.0), sharey=True)
    axes = axes.ravel()
    for axis, np_ratio in zip(axes, sizes):
        size_data = wall_profiles[np.isclose(wall_profiles["npRatio"], np_ratio)]
        for packing_type in ("random", "hexa", "ortho"):
            style = PACKING_STYLE[packing_type]
            subset = size_data[size_data["packingType"] == packing_type]
            grouped = (
                subset.groupby("distanceFromStartingWallByDiameter")
                ["meanAcrossTwoDirections"]
                .agg(["mean", "std"])
                .reset_index()
            )
            if grouped.empty:
                continue
            x = grouped["distanceFromStartingWallByDiameter"].to_numpy(dtype=float)
            mean = grouped["mean"].to_numpy(dtype=float)
            axis.plot(x, mean, color=style["color"], linewidth=1.35,
                      label=style["label"])
            if packing_type == "random":
                std = grouped["std"].fillna(0.0).to_numpy(dtype=float)
                axis.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                                  color=style["color"], alpha=0.18)
        axis.set_title(rf"$N_p={np_ratio:g}$")
        axis.set_xlim(0.0, np_ratio)
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, alpha=0.18)
    for row in range(5):
        axes[row * 3].set_ylabel("Finite-slab local porosity [-]")
    for axis in axes[-3:]:
        axis.set_xlabel("Distance from starting wall / sphere diameter [-]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Complete wall-to-wall plane-averaged porosity profiles", fontsize=14
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.975))
    fig.savefig(output_dir / "wall_to_wall_porosity_all_packings.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def covariance_ellipse(axis, x, y):
    if len(x) < 3:
        return
    from matplotlib.patches import Ellipse
    covariance = np.cov(np.vstack((x, y)))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if np.any(eigenvalues < 0.0):
        return
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ellipse = Ellipse(
        (float(np.mean(x)), float(np.mean(y))),
        width=4.0 * math.sqrt(eigenvalues[0]),
        height=4.0 * math.sqrt(eigenvalues[1]),
        angle=angle, facecolor="0.5", edgecolor="0.25", alpha=0.13,
    )
    axis.add_patch(ellipse)


def plot_q4_q6(results, output_dir):
    q4_column = "meanLocalQ4ContactNeighbours"
    q6_column = "meanLocalQ6ContactNeighbours"
    valid = results.dropna(subset=[q4_column, q6_column]).copy()
    valid = valid[(valid["packingType"] != "random") | valid["settled"].map(as_boolean)]
    fig, (absolute_axis, vector_axis) = plt.subplots(1, 2, figsize=(14.2, 6.5))

    # Panel A contains exactly the three requested packing types.
    for packing_type in ("hexa", "ortho", "random"):
        subset = valid[valid["packingType"] == packing_type]
        style = PACKING_STYLE[packing_type]
        absolute_axis.scatter(
            subset[q4_column], subset[q6_column],
            color=style["color"], marker=style["marker"],
            s=28 if packing_type == "random" else 52,
            alpha=0.22 if packing_type == "random" else 0.85,
            linewidths=0.0, label=style["label"],
        )
    absolute_axis.set_xlim(-0.035, 0.82)
    absolute_axis.set_ylim(-0.035, 0.72)
    absolute_axis.set_aspect("equal", adjustable="box")
    absolute_axis.set_xlabel(r"Mean local $q_4$ [-]")
    absolute_axis.set_ylabel(r"Mean local $q_6$ [-]")
    absolute_axis.set_title("A  Hexa, ortho, and random packings")
    absolute_axis.grid(True, alpha=0.20)
    absolute_axis.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13))

    # Panel B uses only random data and translates the random ensemble mean to
    # the origin, retaining the vector logic of the rock-study figure.
    random = valid[valid["packingType"] == "random"]
    random_q4 = random[q4_column].to_numpy(dtype=float)
    random_q6 = random[q6_column].to_numpy(dtype=float)
    mean_q4 = float(np.mean(random_q4))
    mean_q6 = float(np.mean(random_q6))
    centered_q4 = random_q4 - mean_q4
    centered_q6 = random_q6 - mean_q6
    vector_axis.scatter(centered_q4, centered_q6, s=19, color="0.4", alpha=0.22,
                        label="Random realization means")
    covariance_ellipse(vector_axis, centered_q4, centered_q6)
    vector_axis.scatter(0.0, 0.0, marker="*", s=170, color="black",
                        label="Random ensemble mean", zorder=6)

    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(Q_REFERENCE_STRUCTURES)))
    distances = []
    for index, (name, (reference_q4, reference_q6)) in enumerate(Q_REFERENCE_STRUCTURES.items()):
        delta_q4 = reference_q4 - mean_q4
        delta_q6 = reference_q6 - mean_q6
        distance = float(np.hypot(delta_q4, delta_q6))
        distances.append((name, reference_q4, reference_q6, delta_q4, delta_q6,
                          distance, colors[index]))
    distances.sort(key=lambda row: row[5])
    distance_rows = []
    ranking_lines = ["Distance ranking:"]
    for rank, row in enumerate(distances, start=1):
        name, reference_q4, reference_q6, delta_q4, delta_q6, distance, color = row
        vector_axis.annotate("", xy=(delta_q4, delta_q6), xytext=(0.0, 0.0),
                             arrowprops={"arrowstyle": "-|>", "color": color,
                                         "linewidth": 1.7, "mutation_scale": 12})
        vector_axis.scatter(delta_q4, delta_q6, marker="D", s=55,
                            facecolors="white", edgecolors=[color], linewidths=1.4)
        vector_axis.annotate(str(rank), (delta_q4, delta_q6),
                             xytext=(6 if delta_q4 >= 0 else -6, 5),
                             textcoords="offset points",
                             ha="left" if delta_q4 >= 0 else "right",
                             color=color, fontsize=9, fontweight="bold")
        ranking_lines.append(f"{rank}. {name}: d={distance:.3f}")
        distance_rows.append({
            "randomMeanQ4": mean_q4,
            "randomMeanQ6": mean_q6,
            "referenceStructure": name,
            "referenceQ4": reference_q4,
            "referenceQ6": reference_q6,
            "deltaQ4ReferenceMinusRandom": delta_q4,
            "deltaQ6ReferenceMinusRandom": delta_q6,
            "euclideanDistance": distance,
            "distanceRank": rank,
            "numberOfRandomRealizations": len(random),
        })
    vector_axis.text(0.02, 0.03, "\n".join(ranking_lines), transform=vector_axis.transAxes,
                     ha="left", va="bottom", fontsize=8,
                     bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                           "edgecolor": "0.75", "alpha": 0.9})
    extent = max(0.08, 1.22 * max(abs(value) for row in distances for value in row[3:5]))
    vector_axis.axhline(0.0, color="0.55", linestyle="--", linewidth=0.9)
    vector_axis.axvline(0.0, color="0.55", linestyle="--", linewidth=0.9)
    vector_axis.set_xlim(-extent, extent)
    vector_axis.set_ylim(-extent, extent)
    vector_axis.set_aspect("equal", adjustable="box")
    vector_axis.set_xlabel(r"$\Delta q_4=q_{4,ref}-\overline{q}_{4,random}$ [-]")
    vector_axis.set_ylabel(r"$\Delta q_6=q_{6,ref}-\overline{q}_{6,random}$ [-]")
    vector_axis.set_title("B  Literature references relative to random mean")
    vector_axis.grid(True, alpha=0.16)
    vector_axis.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Contact-neighbour bond-orientational order", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
    fig.savefig(output_dir / "q4_q6_reference_comparison.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(distance_rows).to_csv(output_dir / "q4_q6_reference_distances.csv",
                                       index=False)


def contact_angle_distribution(tasks):
    rows = []
    edges = np.linspace(0.0, 90.0, 19)
    for task in tasks:
        contact_path = task.run_dir / "contacts.csv"
        summary_path = task.run_dir / "analysis_summary.csv"
        if not contact_path.exists() or not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path).iloc[0]
        if task.packing_type == "random" and not as_boolean(summary["settled"]):
            continue
        contacts = pd.read_csv(contact_path)
        values = pd.to_numeric(
            contacts.get("contactAngleDegFromVertical", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        counts, _ = np.histogram(values, bins=edges)
        for index, count in enumerate(counts):
            rows.append({
                "packingType": task.packing_type,
                "npRatio": task.np_ratio,
                "realization": task.realization,
                "angleLowerDeg": edges[index],
                "angleUpperDeg": edges[index + 1],
                "angleMidDeg": 0.5 * (edges[index] + edges[index + 1]),
                "contactCount": int(count),
                "contactFraction": float(count / len(values)) if len(values) else np.nan,
            })
    return pd.DataFrame(rows)


def build_aggregate_outputs(root, tasks):
    root = Path(root)
    output_dir = root / "statistics_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for task in tasks:
        path = task.run_dir / "analysis_summary.csv"
        if path.exists():
            summaries.append(pd.read_csv(path))
    if len(summaries) != len(tasks):
        raise RuntimeError(
            f"Expected {len(tasks)} analysis summaries; found {len(summaries)}"
        )
    results = pd.concat(summaries, ignore_index=True)
    results["settled"] = results["settled"].map(as_boolean)
    results.sort_values(["packingType", "npRatio", "realization"], inplace=True)
    results.to_csv(output_dir / "study_results.csv", index=False)

    contact_columns = [
        "packingType", "npRatio", "realization", "randomSeed", "settled",
        "numberOfSpheres", "numberOfParticleParticleContacts",
        "meanCoordinationNumber", "meanBulkCoordinationNumber",
        "meanWallAndFloorContacts", "meanTotalContactsIncludingWalls",
        "meanContactAngleDegFromVertical", "medianContactAngleDegFromVertical",
        "stdContactAngleDegFromVertical",
    ]
    results[contact_columns].to_csv(
        output_dir / "coordination_contact_angle_summary.csv", index=False
    )

    random_stats = random_group_statistics(results)
    random_stats.to_csv(output_dir / "random_porosity_by_bed_size.csv", index=False)
    plot_literature_porosity(
        random_stats, output_dir, beavers_sparrow_porosity,
        r"Beavers-Sparrow Eq. (8): $\varepsilon_\infty=0.368$, $\varepsilon_w=0.476$",
        "beavers_sparrow_random_porosity.png",
        "Random square beds versus Beavers-Sparrow bed-size relation",
    )
    plot_literature_porosity(
        random_stats, output_dir, dixon_porosity,
        r"Dixon Eq. (17): $0.4+0.05/N_p+0.412/N_p^2$",
        "dixon_random_porosity.png",
        "Random square beds versus Dixon correlation",
    )
    plot_beavers_dixon_comparison(random_stats, output_dir)
    plot_all_packings(results, output_dir)

    wall_profiles = load_run_tables(
        tasks, "wall_to_wall_porosity.csv", settled_random_only=True
    )
    wall_profiles.to_csv(output_dir / "wall_to_wall_profiles_all_runs.csv", index=False)
    plot_wall_profiles(wall_profiles, output_dir)

    zone_profiles = load_run_tables(
        tasks, "hamzah_zone_profiles.csv", settled_random_only=True
    )
    zone_profiles.to_csv(output_dir / "hamzah_zone_profiles_all_runs.csv", index=False)
    plot_hamzah_zone_panels(zone_profiles, output_dir, 1)
    plot_hamzah_zone_panels(zone_profiles, output_dir, 2)

    plot_q4_q6(results, output_dir)
    angle_distribution = contact_angle_distribution(tasks)
    angle_distribution.to_csv(output_dir / "contact_angle_distribution.csv", index=False)

    expected_random = len({task.np_ratio for task in tasks}) * max(
        task.realization for task in tasks if task.packing_type == "random"
    )
    completed_random = results[results["packingType"] == "random"]
    settled_random = completed_random[completed_random["settled"]]
    quality = {
        "expectedTotalRuns": len(tasks),
        "completedTotalRuns": len(results),
        "expectedRandomRuns": expected_random,
        "completedRandomRuns": len(completed_random),
        "settledRandomRunsIncludedInPlots": len(settled_random),
        "unsettledRandomRunsExcludedFromPlots": int((~completed_random["settled"]).sum()),
        "allRequestedRandomRunsSettled": bool(len(settled_random) == expected_random),
    }
    (output_dir / "quality_summary.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )
    print("Aggregate outputs written to:", output_dir)
