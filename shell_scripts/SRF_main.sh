#!/bin/bash
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH --mem=70G


module load Python/3.6.1-intel-2016b 

echo "Stat job $PBS_JOBID started at `date`"

mkdir "$TMPDIR"/Thesis

cp -r $HOME/Thesis/data "$TMPDIR"/Thesis


