#!/bin/bash
#The previous line is mandatory

#SBATCH --job-name=explicit_integral     #Name of your job
#SBATCH --cpus-per-task=1    #Number of cores to reserve
#SBATCH --mem-per-cpu=1G     #Amount of RAM/core to reserve
#SBATCH --time=06:00:00      #Maximum allocated time
#SBATCH --qos=6hours         #Selected queue to allocate your job
#SBATCH --output=myrun.o%j   #Path and name to the file for the STDOUT
#SBATCH --error=myrun.e%j    #Path and name to the file for the STDERR

ml Python                    #Load required modules
python my_script.py inputdata.txt    #Execute your command(s)