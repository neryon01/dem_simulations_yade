# Zero friction with horizontal vibration

This study generates 100 gravity packings with frictionless rock–rock contacts, then applies horizontal vibration and lets the rocks settle again. Rock–wall contacts retain a friction coefficient of 0.50. Each realization has a target of 80 rocks, inserted at random horizontal positions in a repeating sequence of the four rock types.

The simulations start from scratch. The zero-friction-only results are used afterward as a comparison baseline, not as restart geometry. The comparison pairs realizations by spawnSeed and reports the porosity change caused by adding vibration.

# Main files

**run_parameter_study.py**: creates the realization table, dispatches YADE simulations through yade-batch, manages post-processing and combines the results.

**rock_packing_vibrating_box_random_spawn.py**: generates the rock clumps, inserts them under gravity, waits for initial settling, applies X-direction vibration and waits for settling again. Saves the final poses, YADE metrics and timing information.

**reconstruct_stl_w_distance.py**: reconstructs the final STL packing and calculates surface distances, contacts, coordination statistics and local q4–q6 values using SANN, adaptive and Voronoi neighbours.

**rock_porosity.py**: calculates global confined-bed, interior and full-domain porosity. Creates per-run spatial profiles, contour plots and lateral-zone results.

**analyze_parameter_study.py**: creates scalar distributions, the q4–q6 reference comparison, porosity–structure relationships and the paired zero-friction-only-versus-vibration comparison.

**analyze_radial_porosity.py**: combines the spatial results into aggregate wall-to-wall, diagonal and vertical profiles and lateral-zone averages.

# Settings

The active physical settings are defined in rock_packing_vibrating_box_random_spawn.py:

nRocks = 80

rockFrictionCoefficient = 0.0

boxFrictionCoefficient = 0.50

enableVibration = 1

vibrationAxis = "x"

vibAmplitudeFrac = 0.008

vibPeriodSteps = 3000

vibrationSteps = 30000

vibrationRampSteps = 3000

enableCompression = 0

The vibration amplitude is 0.008 times the box width in X. The motion ramps up and down over 3000 iterations at each end. After vibration, the box returns to its original position and the bed settles again. Rock–rock friction remains zero throughout; there is no friction-restoration stage.

The contact law is Hertz–Mindlin. Rock Young's modulus is 20 GPa, Poisson's ratio is 0.20 and density is 2660 kg/m³. Numerical damping is 0.4 and timestep safety is 0.25.

The initial and final settling thresholds are 1e-3, with a minimum settling duration of 5000 iterations and 2000 consecutive low-residual iterations. Every rock must have an external contact. Runs can also finish at the iteration limit; check the exported equilibrium flag when interpreting results.

The runner uses 100 realizations, spawn seeds 24680 through 24779 and a fixed rock-type seed of 13579.

# Zero-friction-only baseline

By default, the analyzer reads a folder named zhao_runs inside the current study folder. This can be a directory or symbolic link to the zero-friction-only results. It accepts an aggregate results CSV or individual run_XXX folders, either directly below zhao_runs/ or below zhao_runs/runs/.

For individual baseline runs, yade_metrics.csv supplies spawnSeed and porosity_result.csv supplies porosity. An aggregate baseline CSV must contain spawnSeed and porosity.

Alternatively, set ZERO_FRICTION_RESULTS_CSV to the baseline CSV or directory. Pairing uses matching spawn seeds, not folder numbering. Unmatched seeds are excluded and reported.

# Run locally

Run from the study folder. Use a Python interpreter with pandas for dispatch and ensure yade-batch is available on the command path. Post-processing must use the Python environment described above. Conda is not required if the necessary packages are already available locally.

CPU count

The runner reads SLURM_CPUS_PER_TASK and defaults to one concurrent job when it is unset. To choose the CPU count inside run_parameter_study.py, replace the current MAX_SIMULTANEOUS_JOBS assignment with, for example:

MAX_SIMULTANEOUS_JOBS = 4
JOB_THREADS = 1

This allows four single-threaded simulations at once and also controls parallel post-processing. No --jobs argument is needed. A fixed assignment also applies on the cluster, so keep it consistent with the allocated CPUs.

**Commands**

Check the post-processing environment:

**python3 run_parameter_study.py preflight**

If the baseline is not available through zhao_runs, set its actual path:

**export ZERO_FRICTION_RESULTS_CSV=/path/to/zero_friction/statistics_outputs/zhao_frictionless_results.csv**

Run the simulations, then post-process them after dispatch finishes:

**python3 run_parameter_study.py dispatch**
**python3 run_parameter_study.py postprocess**


To resume interrupted post-processing without repeating YADE:

**python3 run_parameter_study.py postprocess**

Completed post-processing is skipped. To rebuild only the combined CSV files and aggregate plots from existing per-run outputs:

**python3 run_parameter_study.py collect**

The zero-friction-only baseline must remain available during postprocess and collect.

# Run on CoolMUC-4

Check the Apptainer image path and yadepy environment name in run_zhao_cm4_tiny.sh. The uploaded launcher requests 30 CPUs, 4 GB per CPU and a 24-hour time limit. With the original runner assignment, concurrency follows the allocated CPU count.

**bash run_zhao_cm4_tiny.sh --preflight**
**sbatch run_zhao_cm4_tiny.sh**

To resume only host-side post-processing:

**sbatch run_zhao_cm4_tiny.sh --postprocess-only**
