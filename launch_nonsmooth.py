import os
import sys

Nsim = sys.argv[1] ##seed

main_str = 'main_nonsmooth.py'
dir_str = '/...../AR-DEIG/'
job_str = 'gos_ardieg'


Ncores = 1 # Number of requested cores
Nhours = 24 # Requsted walltime (hours) 120
print("Current working directory:", os.getcwd())
start=0 


for i in range(start,int(Nsim)):
    seed = i+1
    jobId = job_str + '_' + str(seed)
    slurmFileName = 'run_' + jobId + '.slurm'
    print("Current working directory:", os.getcwd())
    with open(slurmFileName, 'w') as fout:
        fout.write('#!/bin/bash -l\n')
        fout.write('\n')
        fout.write('#SBATCH --job-name ' + jobId + '\n')
        fout.write('#SBATCH --nodes 1\n')
        fout.write('#SBATCH --cpus-per-task ' + str(Ncores) + '\n')
        fout.write('#SBATCH --time ' + str(Nhours) + ':00:00\n')
        fout.write('#SBATCH --output ' + 'out_adv' + jobId + '.txt\n')
        fout.write('#SBATCH --mem=100\n')
        fout.write('#SBATCH --mail-type=FAIL\n')
        fout.write('#SBATCH --mail-user=...n')
        fout.write('#SBATCH --partition=batch-hsw\n')  # Add the correct partition
        fout.write('#SBATCH --account=...\n')  # Add your account name
        fout.write('\n')
        fout.write('module load scicomp-python-env\n')
        fout.write('source activate gos\n')
        fout.write('cd ' + dir_str + '\n')
        fout.write('python ' + main_str + ' ' + str(seed))
    os.popen('sbatch ' + slurmFileName)
