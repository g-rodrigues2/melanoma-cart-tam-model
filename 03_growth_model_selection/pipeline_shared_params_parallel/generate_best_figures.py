"""Figures and summaries of the selected scenarios:
  - _exp scenario 1 (group-specific r; the B16 growth rate used in the thesis)
  - _log scenario 2 (group-specific r, global cc)

Outputs in Output_Calibration/:
  - scatter_best__exp_s1.png/.pdf
  - scatter_best__log_s2.png/.pdf
  - model parameters printed to the terminal (median and 95% interval)
"""
import _path_setup  # noqa: F401


from bounds import find_max_initial_condition
from data_loader import load_control_data

from bounds_shared import define_shared_bounds_labels
from plots_shared import make_scatter_both, parameter_summary
from scenarios import SCENARIOS


GROUPS = ['control_b16', 'control_pdx']
OUT = './Output_Calibration'

full_data = load_control_data(data_dir=_path_setup.DATA_DIR)
max_ic = find_max_initial_condition(full_data)[0]


# Selected scenarios
BEST = [
    ('_exp', 1),
    ('_log', 2),
]

for model_ext, snum in BEST:
    sdef = SCENARIOS[model_ext][snum]
    chain_path = f'{OUT}/multi_chain_{model_ext}_s{snum}.gz'
    npz_path   = f'{OUT}/multi_ll_pars_{model_ext}_s{snum}.npz'
    fig_name   = f'{OUT}/scatter_best_{model_ext}_s{snum}'

    print(f'\n=== {model_ext} s{snum}  scenario={sdef} ===')
    corr = make_scatter_both(
        chain_path=chain_path, npz_path=npz_path,
        model_extension=model_ext, scenario=sdef,
        full_data=full_data, groups=GROUPS,
        figure_name=fig_name, n_samp=500, seed=42,
    )

    for g in GROUPS:
        cs = corr[g]
        ccc_med, ccc_lo, ccc_hi, *_ = cs['ccc_summary']
        pcc_med, pcc_lo, pcc_hi, *_ = cs['pcc_summary']
        print(f'  {g:14s}  CCC={ccc_med:.3f} [{ccc_lo:.3f}, {ccc_hi:.3f}]   '
              f'PCC={pcc_med:.3f} [{pcc_lo:.3f}, {pcc_hi:.3f}]')

    # 95% intervals of the model parameters (not of the initial conditions)
    _, _, labels = define_shared_bounds_labels(
        model_ext, sdef, GROUPS, full_data, max_ic,
    )
    # Leading labels (model parameters) and std
    model_param_labels = [l for l in labels[:6] if not l.startswith('ic_')]
    if 'std' not in model_param_labels:
        model_param_labels.append('std')

    pars = parameter_summary(chain_path, labels, model_param_labels)
    print('  Model parameters (median and 95% interval):')
    for name, (med, lo, hi) in pars.items():
        if name == 'std':
            print(f'    {name:14s}  {med:8.3f}  [{lo:8.3f}, {hi:8.3f}]  (mm^3)')
        elif name.startswith('r_'):
            print(f'    {name:14s}  {med:8.4f}  [{lo:8.4f}, {hi:8.4f}]  (day^-1)')
        elif name.startswith('cc'):
            print(f'    {name:14s}  {med:8.1f}  [{lo:8.1f}, {hi:8.1f}]  (mm^3)')
        else:
            print(f'    {name:14s}  {med:8.4g}  [{lo:8.4g}, {hi:8.4g}]')

    print(f'  Figure: {fig_name}.png / .pdf')
