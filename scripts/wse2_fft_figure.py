#!/usr/bin/env python3
"""
FFT-with-hexagon figure for the WSe2 folder, in the style of moire_lattices_fft.png.

Reuses the hexagon finder from moire_verify but labels panels by layer number
(2L/3L/4L) and uses the WSe2 reciprocal search band. Unlike the MoS2 figure it
applies no plausibility filter: all scans are shown so the identified hexagon can
be judged against the raw peaks, which matters here because the tri/tetralayers
carry more than one moire and the single-hexagon fit picks only the strongest.
"""
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moire_verify import fft_of, find_hexagon
from moire_calc import analyze, MATERIALS

from paths import RAW, results_for

FOLDER = RAW["WSe2"]
OUT = results_for("WSe2")
LO, HI = 0.15, 0.45          # WSe2 moire lives near 0.27 1/nm
A = MATERIALS["WSe2"]


def label(fn):
    m = re.search(r"_(\d L?\d+|\dL\d+)\.ibw", fn)
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(fn))[0]


def main():
    files = sorted(glob.glob(os.path.join(FOLDER, "*.ibw")))
    hits = []
    for path in files:
        lev, den, spec, n, px, size = fft_of(path)
        b1, b2 = find_hexagon(spec, n, px, LO, HI)
        tag = label(os.path.basename(path))
        if b1 is None:
            hits.append(dict(tag=tag, spec=spec, px=px, b1=None))
            continue
        res = analyze(b1, b2, A, "recip")
        sol = res["solutions"][0] if res["solutions"] else None
        hits.append(dict(tag=tag, spec=spec, px=px, b1=b1, b2=b2,
                         lmean=res["lmean"], sol=sol))

    ncol = 4
    nrow = int(np.ceil(len(hits) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.7 * nrow))
    axs = np.atleast_1d(axs).ravel()
    for ax, h in zip(axs, hits):
        mag = np.abs(h["spec"])
        ext = 1 / (2 * h["px"])
        ax.imshow(np.log1p(mag), cmap="magma",
                  extent=[-ext, ext, -ext, ext], origin="lower")
        if h["b1"] is not None:
            b1, b2 = h["b1"], h["b2"]
            b3 = -(b1 + b2)
            hexpts = [b1, -b3, b2, -b1, b3, -b2, b1]
            ax.plot([p[0] for p in hexpts], [p[1] for p in hexpts],
                    "-", color="cyan", lw=1.0, alpha=0.9)
            for p in (b1, b2, b3):
                for s in (1, -1):
                    ax.plot(s * p[0], s * p[1], "o", ms=11, mfc="none",
                            mec="cyan", mew=1.5)
            title = f"{h['tag']}  L={h['lmean']:.2f}nm"
            if h["sol"]:
                title += f"  {h['sol']['twist']:.2f}°  {h['sol']['hetero']*100:.1f}%"
        else:
            title = f"{h['tag']}  (no hexagon)"
        ax.set_xlim(-0.45, 0.45)
        ax.set_ylim(-0.45, 0.45)
        ax.axhline(0, color="w", lw=0.3, alpha=0.35)
        ax.axvline(0, color="w", lw=0.3, alpha=0.35)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("1/nm", fontsize=7)
        ax.set_ylabel("1/nm", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axs[len(hits):]:
        ax.axis("off")

    plt.suptitle("WSe2 - moire FFT with identified hexagon (magma, log scale)",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT, "moire_lattices_fft.png")
    plt.savefig(out, dpi=110)
    print("wrote", out)
    for h in hits:
        if h["b1"] is None:
            print(f"  {h['tag']}: no hexagon")
        else:
            s = h["sol"]
            print(f"  {h['tag']}: L={h['lmean']:.2f} nm  "
                  f"twist={s['twist']:.2f}  strain={s['hetero']*100:.1f}%" if s else h['tag'])


if __name__ == "__main__":
    main()
