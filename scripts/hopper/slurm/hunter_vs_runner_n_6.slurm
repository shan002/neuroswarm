#!/bin/bash

## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=hunter-runner-n-6

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
#SBATCH --time=1-20:00  # Total time needed for job: Days-Hours:Minutes

#SBATCH --cpus-per-task 48


## Load the relevant modules needed for the job
source ~/neuromorphic/turtwig/scripts/hopper/neuromodules.sh
cd turtwig/shan/hunter_vs_runner_to_train

export MAKE_PARENTS=true

## Run your program or script
# Took 1-01:20:07 on last run
# srun python hunter_vs_runner.py train --root out/farp/6 -p 48 --cy 2000 --save_best --epochs 200 --eons_seed 20
# srun python hunter_vs_runner.py train --root out/farp/6 -p 48 --cy 2000 --save_best --epochs 200 --eons_seed 21
# srun python hunter_vs_runner.py train --root out/farp/6 -p 48 --cy 2000 --save_best --epochs 200 --eons_seed 22
# srun python hunter_vs_runner.py train --root out/farp/6 -p 48 --cy 2000 --save_best --epochs 200 --eons_seed 23
# srun python hunter_vs_runner.py train --root out/farp/6 -p 48 --cy 2000 --save_best --epochs 200 --eons_seed 24

# Took 1-00:59:23 on last run
srun echo y | python hunter_vs_runner_test.py train --root ~/neuromorphic/turtwig/out/farp/6 --save_best -p 48 -T 10 --cy 2000 --epochs 200 --trial_seed 511 --eons_seed 20
srun echo y | python hunter_vs_runner_test.py train --root ~/neuromorphic/turtwig/out/farp/6 --save_best -p 48 -T 10 --cy 2000 --epochs 200 --trial_seed 512 --eons_seed 21
srun echo y | python hunter_vs_runner_test.py train --root ~/neuromorphic/turtwig/out/farp/6 --save_best -p 48 -T 10 --cy 2000 --epochs 200 --trial_seed 513 --eons_seed 22
srun echo y | python hunter_vs_runner_test.py train --root ~/neuromorphic/turtwig/out/farp/6 --save_best -p 48 -T 10 --cy 2000 --epochs 200 --trial_seed 514 --eons_seed 23
srun echo y | python hunter_vs_runner_test.py train --root ~/neuromorphic/turtwig/out/farp/6 --save_best -p 48 -T 10 --cy 2000 --epochs 200 --trial_seed 515 --eons_seed 24