#!/bin/bash --login
#
#PBS -l select=4
#PBS -o LVL.out
#PBS -e LVL.error
#PBS -l walltime=8:00:00
#PBS -m be
#PBS -M yizhuo.sun.2017@live.rhul.ac.uk
#PBS -A n03-rh

#rm -rf /work/n03/n03/saswata/ .instant/*
module load python-compute
module load fenics/2017.2.0
module load mshr

# Make sure any symbolic links are resolved to absolute path
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)

# Change to the directory that the job was submitted from.
cd $PBS_O_WORKDIR

export INSTANT_SYSTEM_CALL_METHOD=OS_SYSTEM

# Directs the application to be run on 24 compute node cores
aprun -n 96 python LVL_RII.py

