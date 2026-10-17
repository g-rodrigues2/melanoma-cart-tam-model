#!/bin/bash
#----------------------------------------------------
# Quick check before treatment_parallel.slurm (about 30 s):
#   - one model (_exp_ED), one group (cart_sl_b16), 50 steps
#   - separate backend: reinitiate_TEST_cart_sl_b16_exp_ED.h5
#
#   sbatch treatment_test.slurm
#   rm reinitiate_files/reinitiate_TEST_*.h5      # clean up
#----------------------------------------------------

#SBATCH -J calib_treat_test
#SBATCH -o test.%j.out
#SBATCH -e test.%j.err
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 00:10:00
#SBATCH -A <your-allocation>

export CALIB_TEST=1
export CALIB_BACKEND_ONLY=1
export CALIB_COMPRESS_H5=1
export CALIB_USE_POOL=1
export CALIB_N_CORES=8
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONWARNINGS="ignore::UserWarning"

echo "=== SETTINGS ==="
env | grep -E '^(CALIB_|HDF5_)' | sort
echo "================"

python global.py
