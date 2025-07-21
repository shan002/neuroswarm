#!/bin/bash

## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=hunter-runner-n-6-wtp-ic

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
#SBATCH --time=3-23:00  # Total time needed for job: Days-Hours:Minutes

#SBATCH --cpus-per-task 48


## Load the relevant modules needed for the job
source ~/neuromorphic/turtwig/scripts/hopper/neuromodules.sh
cd turtwig/shan/hunter_vs_runner_to_train

export MAKE_PARENTS=true

## Run your program or script
# Took 16:44:04 on last run
# srun python hunter_vs_runner_wtp.py train --root out/farp/6-wtp -p 48 --cy 2000 --save_best --epochs 200 --eons_seed 20
# srun python hunter_vs_runner_wtp.py train --root out/farp/6-wtp -p 48 --cy 2000 --save_best --epochs 200 --eons_seed 21
# srun python hunter_vs_runner_wtp.py train --root out/farp/6-wtp -p 48 --cy 2000 --save_best --epochs 200 --eons_seed 22
# srun python hunter_vs_runner_wtp.py train --root out/farp/6-wtp -p 48 --cy 2000 --save_best --epochs 200 --eons_seed 23
# srun python hunter_vs_runner_wtp.py train --root out/farp/6-wtp -p 48 --cy 2000 --save_best --epochs 200 --eons_seed 24

# Took 17:13:30 on last run
echo y | python hunter_vs_runner_wtp_ternary_srunner_vs_dns.py train --root ~/neuromorphic/turtwig/out/farp/6-wtp-vs-dns-ic --save_best -p 48 -T 10 --cy 3000 --epochs 500 --trial_seed 11 --eons_seed 20
echo y | python hunter_vs_runner_wtp_ternary_srunner_vs_dns.py train --root ~/neuromorphic/turtwig/out/farp/6-wtp-vs-dns-ic --save_best -p 48 -T 10 --cy 3000 --epochs 500 --trial_seed 12 --eons_seed 21
echo y | python hunter_vs_runner_wtp_ternary_srunner_vs_dns.py train --root ~/neuromorphic/turtwig/out/farp/6-wtp-vs-dns-ic --save_best -p 48 -T 10 --cy 3000 --epochs 500 --trial_seed 13 --eons_seed 22
echo y | python hunter_vs_runner_wtp_ternary_srunner_vs_dns.py train --root ~/neuromorphic/turtwig/out/farp/6-wtp-vs-dns-ic --save_best -p 48 -T 10 --cy 3000 --epochs 500 --trial_seed 14 --eons_seed 23
echo y | python hunter_vs_runner_wtp_ternary_srunner_vs_dns.py train --root ~/neuromorphic/turtwig/out/farp/6-wtp-vs-dns-ic --save_best -p 48 -T 10 --cy 3000 --epochs 500 --trial_seed 15 --eons_seed 24