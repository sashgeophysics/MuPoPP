#!/bin/bash --login
#
#PBS -l select=1
#PBS -o sphere.out
#PBS -e sphere.error
#PBS -l walltime=0:40:00
#PBS -m be
#PBS -M saswata.hier-majumder@rhul.ac.uk
#PBS -A n03-rh

rm -rf /work/n03/n03/saswata/ .instant/*
module load python-compute
module load fenics/2017.2.0
#module load mshr

# Make sure any symbolic links are resolved to absolute path
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)

# Change to the directory that the job was submitted from.
cd $PBS_O_WORKDIR

export INSTANT_SYSTEM_CALL_METHOD=OS_SYSTEM

# Directs the application to be run on 24 compute node cores
aprun -n 24 python sphere.py sphere.cfg

