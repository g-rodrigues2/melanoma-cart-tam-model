"""Adds ``pipeline_control_parallel/`` to sys.path, so that models, bounds, inference, data_loader,
postprocess and convergence_check are reused.

Import this module before any ``from models import ...`` in the other files of this folder.
"""
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_CONTROL = os.path.abspath(os.path.join(_HERE, '..', 'pipeline_control_parallel'))

if _CONTROL not in sys.path:
    sys.path.insert(0, _CONTROL)

# PBS data (the same files as the growth-model selection)
DATA_DIR = os.path.abspath(os.path.join(_HERE, '..', '..', 'data', 'processed'))
