# Selecting Number of Rocks

This folder contains the scripts used to check how the prescribed number of rocks affects the gravity-deposited bed. The tested rock counts are 20, 40, 60, 80 and 100. Four realizations are run for each rock count, giving 20 simulations in total.

The same spawn seed is used for corresponding realizations at every rock count. This means that each comparison uses the same initial part of the rock-insertion sequence.

# Main files

**run_parameter_study.py**

Controls the complete number-of-rocks study. It creates the YADE parameter table, launches the simulations, runs the host-side post-processing and combines the results.

It also creates the convergence summary and the plot comparing porosity, particle-particle coordination number, simulation time and final maximum overlap against the prescribed number of rocks.

**rock_packing_vibrating_box_random_spawn.py**

Runs one gravity-deposition simulation in YADE. It creates the clump templates from the GTS rock surfaces, inserts the requested number of rocks, settles the bed and saves the final rock poses, YADE metrics and timing information.

**reconstruct_stl_w_distance.py**

Reconstructs the final rocks from the saved YADE poses and the original STL files. It calculates surface distances, contacts, coordination statistics and Steinhardt order parameters for the reconstructed bed.

**rock_porosity.py**

Calculates the porosity of each reconstructed packing using a voxel representation. It saves the global and interior porosity results together with the spatial porosity profiles.

**analyze_parameter_study.py**

Reads the combined results and creates the statistical plots for the scalar output quantities. It also creates the q4-q6 reference-comparison plots for the different rock counts.

**analyze_radial_porosity.py**

Reads the combined spatial porosity data and creates the aggregate wall, vertical, diagonal and zone-based porosity-profile plots.

# Run locally

Run the YADE simulations with two simultaneous jobs:

**SLURM_CPUS_PER_TASK=2 python3 run_parameter_study.py dispatch**

After the simulations finish, run reconstruction, porosity analysis and result collection:

**SLURM_CPUS_PER_TASK=2 python3 run_parameter_study.py postprocess**

The environment variable controls the number of simultaneous simulations or post-processing workers. If it is omitted outside SLURM, the script uses one worker.

To combine outputs that have already been post-processed:

**python3 run_parameter_study.py collect**

# Run on CoolMUC-4

Check the Apptainer image and Conda environment names in the shell script. An allocation-free check can be run first:

**bash run_number_of_rocks_study_cm4_tiny.sh --preflight**

Submit the complete study with:

**sbatch run_number_of_rocks_study_cm4_tiny.sh**
