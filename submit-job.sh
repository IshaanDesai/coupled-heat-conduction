#!/bin/bash
#
# Job name:
#SBATCH -J macro-micro-desaiin
#
# Error and Output files
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err
#
# Working directory:
#SBATCH -D ./
#
#Notification and type
#SBATCH --mail-type=END
#SBATCH --mail-user=ishaan.desai@ipvs.uni-stuttgart.de
#
# Wall clock limit:
#SBATCH --time=07:00:00
#
# Compute resources
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=33

echo "SLURM_NNODES"=$SLURM_NNODES
echo "working directory="$SLURM_SUBMIT_DIR

# load the modules you need
#module purge
module load ipvs-epyc/gcc/10.2 ipvs-epyc/openmpi/4.0.4-gcc-10.2 ipvs-epyc/python/3.8.5 ub2004/libxml2/2.9.10 ub2004/boost/1.75.0
#module list

echo "Launching macro participant"
mpirun -n 1 --bind-to core python3 macro_heat.py verbose=2 &> log_macro_heat.log & 

echo "Launching micro manager"
mpirun -n 32 --bind-to core python3 run-micro-problems.py verbose=2 &> log_micro_simulations.log

echo "Simulation completed."
