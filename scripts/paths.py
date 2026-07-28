#!/usr/bin/env python3
"""Shared locations, so no script hard-codes a path to someone's Downloads."""
import os

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# raw acquisitions stay where the microscope put them; nothing here writes to them
RAW = {
    "MoS2_2L": os.path.expanduser("~/Downloads/2L_MoS2"),
    "WSe2": os.path.expanduser("~/Downloads/071726_kl62tw4lwse2"),
}

RESULTS = os.path.join(PROJECT, "results")
GWY = os.path.join(PROJECT, "gwy")
DATA = os.path.join(PROJECT, "data")

GRID_FITS = os.path.join(DATA, "grid_fits.json")
WORKBOOK = os.path.join(RESULTS, "MoS2_2L", "moire_grid_analysis.xlsx")


def results_for(sample):
    """Output folder for a sample, created on demand."""
    out = os.path.join(RESULTS, sample)
    os.makedirs(out, exist_ok=True)
    return out


def sample_of(path):
    """Guess which sample a raw path belongs to, for default output routing."""
    p = os.path.abspath(path)
    for name, root in RAW.items():
        if p.startswith(os.path.abspath(root)):
            return name
    return "misc"
