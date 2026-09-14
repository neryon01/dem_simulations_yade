# Clump-Divisor Sensitivity Study

This study evaluates whether the DEM packing results depend strongly on the
resolution of the sphere clumps used to represent the four irregular rocks.
One common clump-resolution divisor is applied to all four rock types.

The tested divisors are:

d = 9, 10, 11, 12

Each divisor uses 15 realizations, giving 60 simulations in total. The same 15
spawn seeds are repeated for every divisor so that results can be compared as
matched realizations. Rock friction is fixed at mu = 0.45, wall friction is
fixed at mu = 0.50, and all other DEM parameters remain unchanged.

The divisor controls the member-sphere radius according to

member-sphere radius = rock x extent / divisor

A larger divisor therefore produces smaller member spheres and generally a
larger number of spheres in each clump. The divisor itself is not the number of
member spheres.

# Main files

**run_parameter_study.py**

Controls the 4 x 15 study. It creates the matched parameter table, launches
the YADE simulations, manages parallel execution, performs post-processing,
combines the results, and creates the nearest-neighbour surface-distance
distribution.

**rock_packing_vibrating_box_random_spawn.py**

Runs one YADE simulation. It applies the divisor supplied by the parameter
table to all four clump templates, inserts the rocks at randomized positions,
simulates gravity deposition, and exports the final poses and YADE metrics.

**reconstruct_stl_w_distance.py**

Reconstructs the original STL rocks at their final YADE positions. It
calculates signed surface gaps, contacts, contact angles, coordination
numbers, pile height, and local Steinhardt q4 and q6 parameters.

**rock_porosity.py**

Calculates global, interior, and spatial porosity from the reconstructed STL
geometries using voxelization. It also produces per-run porosity profiles and
contour plots.

**analyze_parameter_study.py**

Compares scalar results across the four divisors. It plots matched
realizations, group means, and sample standard deviations for porosity, pile
height, total coordination number, maximum overlap, and other recorded
metrics.

**analyze_radial_porosity.py**

Compares spatial porosity profiles across the four divisors, including
wall-to-wall, vertical, diagonal, and lateral-zone profiles.

**run_divisor_sensitivity_cm4_tiny.sh**

Launches the complete workflow on the CoolMUC SLURM cluster. It runs the YADE
stage inside the configured Apptainer image and the post-processing stage in
the yadepy Conda environment.

# Running on CoolMUC

Run the allocation-free check first:

bash run_divisor_sensitivity_cm4_tiny.sh --preflight

Submit the complete study:

sbatch run_divisor_sensitivity_cm4_tiny.sh

If the YADE simulations completed but host post-processing was interrupted:

sbatch run_divisor_sensitivity_cm4_tiny.sh --postprocess-only

# Running locally

Run the YADE simulations:

**python3 run_parameter_study.py dispatch**

Then run STL reconstruction, porosity calculation, aggregation, and plotting:

**python3 run_parameter_study.py postprocess**


The number of concurrent local workers can be set temporarily, for example:

**SLURM_CPUS_PER_TASK=2 python3 run_parameter_study.py dispatch**

**SLURM_CPUS_PER_TASK=2 python3 run_parameter_study.py postprocess**
