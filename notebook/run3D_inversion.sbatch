#!/bin/bash
##SBATCH -J 3D_inversion
#SBATCH --array=0-4
#SBATCH --output=output/array_%A_%a.out
#SBATCH --error=output/array_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=4G
#ml load python/3.6.1
#ml load py-numpy/1.14.3_py36
python 3D_2017-2019_temporal-constraint.py $SLURM_ARRAY_TASK_ID
