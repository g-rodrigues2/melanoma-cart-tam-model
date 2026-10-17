#!/bin/bash
#----------------------------------------------------
# Parameter sharing between the control groups (b16 + pdx) on one 128-core node (Lonestar6, TACC).
# Models: _exp (2 scenarios) and _log (4 scenarios) = 6 processes, each with ~20 cores through
# multiprocessing.Pool (6 x 20 = 120).
#
#   sbatch shared_parallel.slurm
#
# Adjust the partition, allocation and time to your cluster; the backends are resumable.
#----------------------------------------------------

#SBATCH -J calib_shared
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 02:00:00
#SBATCH -A <your-allocation>

export CALIB_BACKEND_ONLY=1
export CALIB_COMPRESS_H5=1
export CALIB_USE_POOL=1
export CALIB_N_CORES=20
export CALIB_CHAIN_SIZE=30000
export CALIB_WALKERS_PER_DIM=4
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONWARNINGS="ignore::UserWarning"

unset CALIB_TEST

mkdir -p logs

echo "=== SETTINGS ==="
env | grep -E '^(CALIB_|HDF5_)' | sort
echo "================"
echo "Start: $(date)"
echo

# Scenarios per model
declare -A SCEN
SCEN[_exp]="1,4"
SCEN[_log]="1,2,3,4"

MODEL_LIST=(_exp _log)
echo "MODEL_LIST = ${MODEL_LIST[@]}"
for m in "${MODEL_LIST[@]}"; do echo "  ${m} -> scenarios=${SCEN[$m]}"; done
echo

# One process per (model, scenario)
for m in "${MODEL_LIST[@]}"; do
    IFS=',' read -ra SLIST <<< "${SCEN[$m]}"
    for s in "${SLIST[@]}"; do
        log="logs/${m}_s${s}.log"
        echo "[launch] model=${m} scenario=${s} -> ${log}"
        CALIB_MODELS=$m CALIB_SCENARIOS=$s \
            python global.py > "$log" 2>&1 &
        sleep 1
    done
done

echo
echo "=== Waiting ==="
wait
echo
echo "=== All finished: $(date) ==="
echo

echo "=== HDF5 BACKENDS ==="
ls -lh reinitiate_files/reinitiate_*.h5 2>/dev/null | grep -v TEST_

echo
echo "=== PROGRESS PER PAIR ==="
python <<'PYEOF'
import h5py, glob, os
target = int(os.environ.get('CALIB_CHAIN_SIZE', '30000'))
files = sorted(f for f in glob.glob('reinitiate_files/reinitiate_*.h5')
               if 'TEST_' not in os.path.basename(f))
done, partial = 0, []
for f in files:
    try:
        with h5py.File(f, 'r') as h:
            it = int(h['mcmc'].attrs['iteration'])
    except Exception:
        it = -1
    status = 'OK' if it >= target else f'INCOMPLETE ({it}/{target})'
    if it >= target: done += 1
    else: partial.append(os.path.basename(f))
    print(f'  {os.path.basename(f):55s} {it:>6d} iters  [{status}]')
print(f'\nSummary: {done}/{len(files)} pairs at the target of {target} steps.')
if partial:
    print('To complete: sbatch shared_parallel.slurm')
PYEOF
