# Box Size Sensitivity

This folder contains the scripts used for the square-box size sensitivity
study.

The full box widths are:

W/L = 3.5, 4.0, 4.5, 5.0, 5.5

YADE uses half-extents, so the values given to the simulation are:

boxHalfExtentXYFracL = 1.75, 2.00, 2.25, 2.50, 2.75

There are 20 runs for each width and 100 runs in total. The same 20 spawn
seeds are used for every width. The target is 80 rocks per run. Only the x and
y dimensions of the box are changed. The z dimensions, rock properties, wall
properties, and clump definitions remain fixed. The baseline width is
W/L = 4.0.

# Files

**run_parameter_study.py**

Creates the parameter table, runs the 100 cases, starts post-processing, and
collects the results into the study-level CSV files.

**rock_packing_vibrating_box_random_spawn.py**

Runs one YADE case. It creates the box with the width supplied in the parameter
table, generates the clump templates, inserts the rocks, lets them settle under
gravity, and saves the final poses and YADE metrics.

**reconstruct_stl_w_distance.py**

Reconstructs the STL rocks from the final YADE poses. It calculates the pile
height, contacts, contact angles, coordination numbers, surface distances, and
the local q4 and q6 values.

**rock_porosity.py**

Calculates porosity from the reconstructed STL geometry. The actual box bounds
from each YADE run are used. Global porosity, interior porosity, spatial
profiles, and contour plots are saved.

**analyze_parameter_study.py**

Compares the scalar results for the five box widths. It plots individual
matched runs together with the mean and sample standard deviation.

**analyze_radial_porosity.py**

Compares the spatial porosity profiles for the five box widths, including the
wall-to-wall, vertical, diagonal, and lateral-zone profiles.

**run_bed_size_sensitivity_cm4_tiny.sh**

SLURM launcher for CoolMUC. It runs YADE in the Apptainer container and runs
the STL and porosity post-processing in the yadepy environment.

# Run on CoolMUC

Preflight:

**bash run_bed_size_sensitivity_cm4_tiny.sh --preflight**

Submit the study:

**sbatch run_bed_size_sensitivity_cm4_tiny.sh**

Post-processing only:

**sbatch run_bed_size_sensitivity_cm4_tiny.sh --postprocess-only**

# Run locally

Run YADE:

**python3 run_parameter_study.py dispatch**

Run reconstruction, porosity analysis, and plotting:

**python3 run_parameter_study.py postprocess**

To use two simultaneous workers locally:

**SLURM_CPUS_PER_TASK=2 python3 run_parameter_study.py dispatch**

**SLURM_CPUS_PER_TASK=2 python3 run_parameter_study.py postprocess**
