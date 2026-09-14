## Gravity Deposition 
Runs multiple simulations where rocks fall inside a rectangular container, and saves statistics of the packing. 

The gravity study consists of six main Python scripts:

**run_parameter_study.py**

Controls the complete 100-realisation study. It generates the parameter table, launches the YADE simulations, manages parallel execution, performs post-processing, combines the per-run results, and starts the statistical analyses.

**rock_packing_vibrating_box_random_spawn.py**

Runs one YADE simulation. It generates the four clump templates from the GTS geometries, inserts the rocks at randomized positions, simulates gravity deposition until the packing settles, and exports the final rock poses and YADE metrics.

**reconstruct_stl_w_distance.py**

Reconstructs the original STL rock geometries at their final YADE positions. It calculates surface distances, rock contacts, contact angles, coordination numbers, pile height, and the local Steinhardt (q_4) and (q_6) parameters.

**rock_porosity.py**

Calculates porosity from the reconstructed STL geometries using voxelization. It evaluates global and interior porosity, vertical and lateral profiles, diagonal profiles, wall-distance behaviour, wall-to-wall centrelines, Hamzah lateral zones, and porosity contour plots.

**analyze_parameter_study.py**

Combines the scalar results from all successful simulations and produces ensemble distributions for simulation time, overlap, porosity, contact angle, coordination number, and other metrics. It also compares the calculated (q_4) and (q_6) values with ideal reference structures.

**analyze_radial_porosity.py**

Combines the spatial porosity data from all simulations. It produces ensemble-averaged nearest-wall, full wall-to-wall, diagonal, vertical, and Hamzah-zone porosity plots, including the mean and standard deviation across the realizations.

## Run on local machine 
There are 2 options 

1. Directly run it with the following commands, where SLURM_CPUS_PER_TASK specify the number of cores you want.



**SLURM_CPUS_PER_TASK=2 python3 run_parameter_study.py dispatch**

**SLURM_CPUS_PER_TASK=2 python3 run_parameter_study.py postprocess**



2. Change the value of MAX_SIMULTANEOUS_JOBS, and run the following commands in order:



**run_parameter_study.py dispatch**

**run_parameter_study.py postprocess**



## Run on slurm 
Submit the following and it runs post-processing as well.

**sbatch run_gravity_total_cm4_tiny.sh**
