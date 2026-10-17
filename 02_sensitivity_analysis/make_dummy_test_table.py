"""LaTeX table of the dummy-parameter test (appendix of Chapter 3), built from dummy_test_end.csv.

Writes results/dummy_test_table.tex, whose body is pasted into PosTextuais/Apendice.tex. Keeping the table
generated rather than typed is what makes the reported p-values traceable to the run that produced them.
"""
import os

import pandas as pd

import efast_dummy_test as dt

SYMBOL = {
    'alpha1': r'$\alpha_1$', 'alpha2': r'$\alpha_2$', 'alpha3': r'$\alpha_3$',
    'beta1': r'$\beta_1$', 'K1': r'$K_1$', 'K2': r'$K_2$', 'K3': r'$K_3$',
    'mu1': r'$\mu_1$', 'kappa2': r'$\kappa_2$', 'kappa3': r'$\kappa_3$',
    'dummy': 'dummy',
}
LONG_NAME = {'T': 'Tumor cells $T(60)$', 'M': 'TAMs $M(60)$', 'C': 'CAR T-cells $C(60)$'}


def _fmt_index(x):
    """Sensitivity indices: three decimals, scientific notation below the third decimal."""
    if x >= 5e-4:
        return f'${x:.3f}$'
    mantissa, exponent = f'{x:.1e}'.split('e')
    return rf'${mantissa}\times10^{{{int(exponent)}}}$'


def _fmt_p(p):
    if pd.isna(p):
        return '--'
    if p < 1e-3:
        mantissa, exponent = f'{p:.0e}'.split('e')
        return rf'${mantissa}\times10^{{{int(exponent)}}}$'
    return f'${p:.3f}$'


def _yes_no(flag):
    return 'yes' if flag else 'no'


def main():
    df = pd.read_csv('results/dummy_test_end.csv')
    lines = []
    for out in dt.OUTPUTS:
        sub = df[df['output'] == out]
        lines.append(rf'\multicolumn{{9}}{{l}}{{\textbf{{{LONG_NAME[out]}}}}}\\')
        lines.append(r'\midrule')
        for _, row in sub.iterrows():
            is_dummy = row['parameter'] == 'dummy'
            cells = [
                SYMBOL[row['parameter']],
                _fmt_index(row['S1_mean']),
                _fmt_p(row['S1_p']),
                '--' if is_dummy else _yes_no(row['S1_significant']),
                '--' if is_dummy else _yes_no(row['S1_above_cutoff']),
                _fmt_index(row['ST_mean']),
                _fmt_p(row['ST_p']),
                '--' if is_dummy else _yes_no(row['ST_significant']),
                '--' if is_dummy else _yes_no(row['ST_above_cutoff']),
            ]
            lines.append(' & '.join(cells) + r'\\')
        if out != dt.OUTPUTS[-1]:
            lines.append(r'\midrule')

    os.makedirs('results', exist_ok=True)
    body = '\n'.join(lines) + '\n'
    with open('results/dummy_test_table.tex', 'w', encoding='utf-8') as fh:
        fh.write(body)
    print(body)
    print("File 'results/dummy_test_table.tex' was saved.")


if __name__ == '__main__':
    main()
