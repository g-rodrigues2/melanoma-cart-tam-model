"""LaTeX table of the treatment models tested in Chapter 3 (thesis appendix).

Reads Output_Calibration/model_selection_table.csv (written by compute_metrics.py) and writes the body of
the table with the BIC of every candidate model in the three CAR T-cell groups, with the best model of each
group in bold.

  Output_Calibration/TMC/model_table_cap3.tex
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _paths import CALIBRATION_DIR, TMC

OUT = os.path.join(CALIBRATION_DIR, 'Output_Calibration')
GROUPS = ['cart_ss_b16', 'cart_sm_b16', 'cart_sl_b16']

# model key -> (label, killing term as implemented with the normalized dose, number of free parameters
# of the term or of the mechanistic model)
ROWS = [
    ('_exp', 'No treatment', '---', 0),
    ('_exp_ED', 'Exponential decay (ED)', r'$a\,e^{-bt}$', 2),
    ('_exp_CD', 'Cumulative dose (CD)', r'$a\,e^{-bt}$', 2),
    ('_exp_AccelED', 'Accelerated ED', r'$a\,e^{-bt^{2}}$', 2),
    ('_exp_AccelCD', 'Accelerated CD', r'$a\,e^{-bt^{2}}$', 2),
    ('_exp_MM', 'Michaelis--Menten (MM)', r'$a/(1+b)$', 2),
    ('_exp_LS', 'Logistic saturation (LS)', r'$a/(1+bT)$', 2),
    ('_exp_PF', 'Power form (PF)', r'$c$', 2),
    ('_exp_dePR', 'de Pillis--Radunskaya (PR)', r'$d/(1+sT^{\lambda})$', 3),
    ('_exp_dePRDecay', 'PR with elimination', r'$d\,e^{-k t}/(1+sT^{\lambda})$', 4),
    ('_exp_EDwithLog', 'ED with logistic onset', r'$a\,e^{-bt}/(1+e^{-st})$', 3),
    ('_exp_CDwithLog', 'CD with logistic onset', r'$a\,e^{-bt}/(1+e^{-st})$', 3),
    ('_exp_EDwithExp', 'ED with exponential onset', r'$a\,e^{-bt}(1-e^{-rt})$', 3),
    ('_exp_CDwithExp', 'CD with exponential onset', r'$a\,e^{-bt}(1-e^{-rt})$', 3),
    ('_exp_EDwithSin', 'ED with sinusoidal modulation', r'$a\,e^{-bt}(1+\sin\omega t)/2$', 3),
    ('_exp_CDwithSin', 'CD with sinusoidal modulation', r'$a\,e^{-bt}(1+\sin\omega t)/2$', 3),
    ('_mac', r'$T$-$M$ model', r'mechanistic, $\mu_1=0$', 3),
    ('_mac_treat', r'$T$-$M$-$C$ model', r'mechanistic, $\mu_1C$', 8),
]


def main():
    with open(os.path.join(OUT, 'model_selection_table.csv'), newline='') as f:
        bic = {(r['group'], r['model']): float(r['BIC']) for r in csv.DictReader(f)}
    best = {g: min(bic[(g, k)] for k, _, _, _ in ROWS) for g in GROUPS}

    lines = []
    for key, label, form, npar in ROWS:
        cells = []
        for g in GROUPS:
            v = bic[(g, key)]
            cells.append(r'\textbf{%.1f}' % v if v == best[g] else '%.1f' % v)
        lines.append('%s & %s & %d & %s \\\\' % (label, form, npar, ' & '.join(cells)))

    path = os.path.join(TMC, 'model_table_cap3.tex')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print('written', path)
    for g in GROUPS:
        winner = min(((bic[(g, k)], label) for k, label, _, _ in ROWS))
        print(f'{g}: best = {winner[1]} (BIC {winner[0]:.2f})')


if __name__ == '__main__':
    main()
