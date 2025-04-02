#!/bin/zsh
#SBATCH --job-name=Gempy3_HSI

### File / path where STDOUT will be written, the %J is the job id

#SBATCH --output=./Result_%J.txt

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes or days and hours and may add or
### leave out any other parameters

#SBATCH --time=01:00:00

### Request all CPUs on one node
#SBATCH --nodes=2

### Request number of CPUs
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1


### Specify your mail address
#SBATCH --mail-user=ravi@aices.rwth-aachen.de
### Send a mail when job is done
#SBATCH --mail-type=END

### Request memory you need for your job in MB
#SBATCH --mem-per-cpu=4096M


source /home/jt925938/.bashrc
module load intel/2022.1.0
module load impi/2021.6.0

conda activate gempy_dino
srun --mpi=pmi2 python run.py