#!/bin/bash
#----------------------------------------------------
# Treatment models of the CAR T-cell groups on one 128-core node (Lonestar6, TACC).
#
# 18 models x 3 groups = 54 pairs, up to 200k steps each:
#   16 _exp_* (1D, fast)
#    1 _mac       (2D, T+M)    -- alpha1, K1, beta1 sampled in log10 (see bounds.py)
#    1 _mac_treat (3D, T+M+C)  -- alpha1, K1, beta1 sampled in log10 (see bounds.py)
# Moves: StretchMove (0.7) + DEMove (0.3) (see global.py).
#
# Resumable: if the wall time ends, resubmitting continues from the backends (the number of walkers per
# dimension must not change).
#
#   sbatch treatment_parallel.slurm
#
# Adjust the partition, allocation and time to your cluster.
#----------------------------------------------------

#SBATCH -J calib_treat
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 24:00:00
#SBATCH -A <your-allocation>

# 54 pairs x 2 cores per pair = 108 cores (20 left for I/O)
export CALIB_BACKEND_ONLY=1
export CALIB_COMPRESS_H5=1
export CALIB_USE_POOL=1
export CALIB_N_CORES=2
export CALIB_CHAIN_SIZE=200000              # total number of steps (resumable)
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONWARNINGS="ignore::UserWarning"
export CALIB_WALKERS_PER_DIM=6

unset CALIB_TEST

mkdir -p logs

echo "=== SETTINGS ==="
env | grep -E '^(CALIB_|HDF5_)' | sort
echo "================"
echo "Start: $(date)"
echo

MODEL_LIST=(_exp \
            _exp_ED _exp_CD _exp_AccelED _exp_AccelCD \
            _exp_MM _exp_LS _exp_PF \
            _exp_dePR _exp_dePRDecay \
            _exp_CDwithLog _exp_CDwithExp _exp_CDwithSin \
            _exp_EDwithLog _exp_EDwithExp _exp_EDwithSin \
            _mac _mac_treat)
GROUP_LIST=(cart_ss_b16 cart_sm_b16 cart_sl_b16)

echo "MODEL_LIST = ${MODEL_LIST[@]}"
echo "GROUP_LIST = ${GROUP_LIST[@]}"
echo "Total: ${#MODEL_LIST[@]} models x ${#GROUP_LIST[@]} groups = $((${#MODEL_LIST[@]} * ${#GROUP_LIST[@]})) pairs"
echo

for m in "${MODEL_LIST[@]}"; do
    for g in "${GROUP_LIST[@]}"; do
        log="logs/${g}${m}.log"
        echo "[launch] model=${m} group=${g} -> ${log}"
        CALIB_MODELS=$m CALIB_GROUPS=$g python global.py > "$log" 2>&1 &
        sleep 1
    done
done

echo
echo "=== All pairs launched; waiting ==="
wait
echo
echo "=== All pairs finished: $(date) ==="
echo

echo "=== HDF5 BACKENDS ==="
ls -lh reinitiate_files/reinitiate_cart_*.h5 2>/dev/null | grep -v TEST_

echo
echo "=== PROGRESS PER PAIR ==="
python <<'PYEOF'
import h5py, glob, os
target = int(os.environ.get('CALIB_CHAIN_SIZE', '300000'))
files = sorted(f for f in glob.glob('reinitiate_files/reinitiate_cart_*.h5')
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
    print(f'To complete the partial chains, resubmit: sbatch treatment_parallel.slurm')
PYEOF
