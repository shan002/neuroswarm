#!/bin/bash

## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=tenn2_n_15

## General partitions: all-HiPri, bigmem-HiPri   --   (12 hour limit)
##                     all-LoPri, bigmem-LoPri, gpuq  (5 days limit)
## Restricted: CDS_q, CS_q, STATS_q, HH_q, GA_q, ES_q, COS_q  (10 day limit)
#SBATCH --partition=normal

## Separate output and error messages into 2 files.`
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID`
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file`
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file`

## Slurm can send you updates via email
#SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=sbhatt28@gmu.edu     # Put your GMU email address here

## Specify how much memory your job needs. (2G is the default)
#SBATCH --mem=8G        # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=2-00:00  # Total time needed for job: Days-Hours:Minutes

#SBATCH --cpus-per-task 48


## Load the relevant modules needed for the job
source ~/neuromorphic/turtwig/scripts/hopper/neuromodules.sh
cd turtwig/

export MAKE_PARENTS=true

## Run your program or script | Takes approximately 1 min / epoch
# Took 1-02:13:12 on last run
srun echo y | python experiment_tenn2.py train --root ~/neuromorphic/turtwig/out/tenn2/n-15 -p 48 --save_best --epochs 500 --eons_seed 20 -N 15
srun echo y | python experiment_tenn2.py train --root ~/neuromorphic/turtwig/out/tenn2/n-15 -p 48 --save_best --epochs 500 --eons_seed 21 -N 15
srun echo y | python experiment_tenn2.py train --root ~/neuromorphic/turtwig/out/tenn2/n-15 -p 48 --save_best --epochs 500 --eons_seed 22 -N 15
srun echo y | python experiment_tenn2.py train --root ~/neuromorphic/turtwig/out/tenn2/n-15 -p 48 --save_best --epochs 500 --eons_seed 23 -N 15
srun echo y | python experiment_tenn2.py train --root ~/neuromorphic/turtwig/out/tenn2/n-15 -p 48 --save_best --epochs 500 --eons_seed 24 -N 15