#!/bin/bash
#----------------------------------------------------
# Growth models of the PBS control groups on one 128-core node (Lonestar6, TACC).
#
#   - 8 models x 2 groups = 16 Python processes on the same node;
#   - each process uses 8 cores through multiprocessing.Pool (16 x 8 = 128 cores).
#
# Wall time is that of the slowest pair (about 60-90 min).
#
#   sbatch control_parallel.slurm
#
# Adjust the partition, allocation and time to your cluster. The backends are resumable: resubmitting
# completes partial chains.
#----------------------------------------------------

#SBATCH -J calib_parallel
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 02:00:00
#SBATCH -A <your-allocation>

# === Settings shared by the 16 processes ===
export CALIB_BACKEND_ONLY=1                 # sampling only (HDF5 backends, no figures)
export CALIB_COMPRESS_H5=1                  # gzip compression of the backends
export CALIB_USE_POOL=1                     # parallel walkers within each pair
export CALIB_N_CORES=8                      # 8 cores per pair x 16 pairs = 128
export CALIB_CHAIN_SIZE=30000               # total number of steps (resumable)
export HDF5_USE_FILE_LOCKING=FALSE          # shared file system
export PYTHONWARNINGS="ignore::UserWarning"

# === Optional tuning ===
export CALIB_WALKERS_PER_DIM=4        # default 2; 4 improves mixing
# export CALIB_SOLVER=odeint          # default; alternative: solve_ivp (about 2x slower)
# export CALIB_SOLVER_METHOD=LSODA    # with solve_ivp: LSODA, BDF, Radau, RK45

unset CALIB_TEST

mkdir -p logs

echo "=== SETTINGS ==="
env | grep -E '^(CALIB_|HDF5_)' | sort
echo "================"
echo "Start: $(date)"
echo

# Do not name a bash array GROUPS: it is a read-only built-in.
MODEL_LIST=(_exp _mendel _bert _lin _gomp _surf _log _mac)
GROUP_LIST=(control_b16 control_pdx)

echo "MODEL_LIST = ${MODEL_LIST[@]}"
echo "GROUP_LIST = ${GROUP_LIST[@]}"
echo

# One process per (model, group) with its own log; a short sleep spreads the imports.
for m in "${MODEL_LIST[@]}"; do
    for g in "${GROUP_LIST[@]}"; do
        log="logs/${g}${m}.log"
        echo "[launch] model=${m} group=${g} -> ${log}"
        CALIB_MODELS=$m CALIB_GROUPS=$g python global.py > "$log" 2>&1 &
        sleep 1
    done
done

echo
echo "=== All 16 pairs launched; waiting ==="
wait
echo
echo "=== All 16 pairs finished: $(date) ==="
echo

echo "=== HDF5 BACKENDS ==="
ls -lh reinitiate_files/reinitiate_control_*.h5 2>/dev/null | grep -v TEST_

echo
echo "=== PROGRESS PER PAIR ==="
python <<'PYEOF'
import h5py, glob, os
target = int(os.environ.get('CALIB_CHAIN_SIZE', '30000'))
files = sorted(f for f in glob.glob('reinitiate_files/reinitiate_control_*.h5')
               if 'TEST_' not in os.path.basename(f))
done, partial = 0, []
for f in files:
    try:
        with h5py.File(f, 'r') as h:
            it = int(h['mcmc'].attrs['iteration'])
    except Exception as e:
        it = -1
    status = 'OK' if it >= target else f'INCOMPLETE ({it}/{target})'
    if it >= target: done += 1
    else: partial.append(os.path.basename(f))
    print(f'  {os.path.basename(f):55s} {it:>6d} iters  [{status}]')
print(f'\nSummary: {done}/{len(files)} pairs at the target of {target} steps.')
if partial:
    print(f'To complete the partial chains, resubmit: sbatch control_parallel.slurm')
PYEOF
