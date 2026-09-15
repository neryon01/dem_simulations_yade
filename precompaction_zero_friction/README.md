# Zero-friction gravity packing

This study generates 100 gravity packings with frictionless rock–rock contacts. Rock–wall contacts retain a friction coefficient of 0.50. Each realization has a target of 80 rocks, inserted at random horizontal positions in a repeating sequence of the four rock types.

The packings are generated from scratch, not restarted from the normal-gravity beds. Vibration and compression are disabled. The normal-gravity results are used afterward for the paired porosity comparison.

# Main files

**run_parameter_study.py**: creates the realization table, launches YADE through yade-batch, manages post-processing and combines the results from all runs.

**rock_packing_vibrating_box_random_spawn.py**: generates the rock clumps, inserts them under gravity and saves the final packing, overlap diagnostics and timing data. Explicit material-pair settings make only the rock–rock contacts frictionless.

**reconstruct_stl_w_distance.py**: reconstructs the final STL packing and calculates surface distances, contacts, coordination statistics and local q4–q6 values using SANN, adaptive and Voronoi neighbours.

**rock_porosity.py**: calculates global confined-bed, interior and full-domain porosity. Saves spatial profiles, contour plots and lateral-zone results for each realization.

**analyze_parameter_study.py**: creates scalar distributions, the q4–q6 reference comparison, porosity–structure relationships and the paired normal-gravity-versus-zero-friction porosity comparison.

**analyze_radial_porosity.py**: combines the spatial data into aggregate wall-to-wall, diagonal and vertical profiles and lateral-zone averages.

# Inputs and dependencies

For the paired comparison, provide the combined normal-gravity results CSV containing the corresponding realization IDs. The analyzer can search nearby study folders automatically, or use an explicit path through GRAVITY_RESULTS_CSV. Keep matching run IDs between the two studies; the comparison pairs by run_id.

# Settings

Physical settings are defined in rock_packing_vibrating_box_random_spawn.py:

nRocks = 80

rockFrictionCoefficient = 0.0

boxFrictionCoefficient = 0.50

enableVibration = 0

enableCompression = 0

The contact law is Hertz–Mindlin. Rock Young's modulus is 20 GPa, Poisson's ratio is 0.20 and density is 2660 kg/m³. Numerical damping is 0.4 and timestep safety is 0.25.

The settling threshold is 1e-3, with a minimum settling duration of 5000 iterations and 2000 consecutive low-residual iterations. Every rock must have an external contact. Runs can also finish at the iteration limit; check the exported equilibrium flag when interpreting results.

The runner uses 100 realizations, spawn seeds 24680 through 24779 and a fixed rock-type seed of 13579.

# Run locally

Run from the study folder. Use a Python interpreter with pandas for dispatch and ensure yade-batch is available on the command path. Post-processing must use the Python environment described above; Conda is not required if those packages are already available locally.

**CPU count**

By default, MAX_SIMULTANEOUS_JOBS reads SLURM_CPUS_PER_TASK and falls back to one job on a local machine. To set the local concurrency inside run_parameter_study.py, replace its current assignment with, for example:

MAX_SIMULTANEOUS_JOBS = 4
JOB_THREADS = 1

This runs up to four single-threaded simulations concurrently and also controls the number of parallel post-processing jobs. No --jobs argument is needed. A fixed assignment also applies on the cluster, so keep it consistent with the CPU allocation there.

**Commands**

Check the post-processing environment:

**python3 run_parameter_study.py preflight**

If you want a comparison with normal gravity = Set the normal-gravity comparison input, replacing the example path with the actual results CSV:

**export GRAVITY_RESULTS_CSV=/path/to/gravity/statistics_outputs/parameter_study_results.csv**

Run the simulations, then post-process them after dispatch finishes:

**python3 run_parameter_study.py dispatch**
**python3 run_parameter_study.py postprocess**

To resume interrupted post-processing without repeating the simulations:

**python3 run_parameter_study.py postprocess**

Completed post-processing is skipped. To rebuild only the combined CSV files and aggregate plots from existing per-run outputs:

**python3 run_parameter_study.py collect**

If you chose to do a comparison with normal gravity = The gravity comparison input must remain available during postprocess and collect.

# Run on CoolMUC-4

Check the Apptainer image path and yadepy environment name in run_zhao_cm4_tiny.sh. The launcher requests 20 CPUs, 4 GB per CPU and a 20-hour time limit. With the original runner assignment, concurrency follows the allocated CPU count.

**bash run_zhao_cm4_tiny.sh --preflight**
**sbatch run_zhao_cm4_tiny.sh**

To resume only host-side post-processing:

**sbatch run_zhao_cm4_tiny.sh --postprocess-only**

If you chose to do a comparison with normal gravity = Ensure the normal-gravity results are discoverable in the nearby study folders. If an explicit comparison path is needed, set GRAVITY_RESULTS_CSV inside the launcher before its post-processing command; the launcher's --export=NONE does not inherit it automatically from the submission terminal.
