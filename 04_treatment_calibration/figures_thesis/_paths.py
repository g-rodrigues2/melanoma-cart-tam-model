"""Locations of the calibration outputs used by the thesis figure scripts."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DIR = os.path.dirname(HERE) if os.path.basename(HERE) == 'figures_thesis' else HERE
TMC = os.path.join(CALIBRATION_DIR, 'Output_Calibration', 'TMC')
FIGURES = os.path.join(TMC, 'figures')
os.makedirs(FIGURES, exist_ok=True)

# Thesis figure standard, measured on the published Chapter 3 figures (fig5/fig7/fig8.pdf):
# Times New Roman for text, Computer Modern for math, matching the TeX Gyre Termes body font.
# Sizes are chosen so that, after the figure is scaled to the 6.30 in text width, the labels
# land between 8 and 10 pt, as they do in the published figures.
STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.labelsize': 15, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'legend.fontsize': 11.5, 'axes.titlesize': 15, 'axes.linewidth': 1.0,
}
TEXTWIDTH_IN = 6.30  # abntex2 A4 with 3 cm / 2 cm side margins
GROUP_LABELS = {'cart_ss_b16': 'short hinge', 'cart_sm_b16': 'medium hinge', 'cart_sl_b16': 'long hinge'}
