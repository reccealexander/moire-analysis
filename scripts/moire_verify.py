#!/usr/bin/env python3
"""
Render verification figures for automatically identified moire lattices.

For every scan that yields a physically plausible fit this writes two panels:
a reciprocal-space view with the six identified peaks and the hexagon that was
assembled from them, and a real-space view with the moire lattice reconstructed
from those same peaks drawn on top of the data.

The real-space overlay is the honest check. Peak positions alone fix only the
lattice vectors; the absolute registry comes from the complex Fourier phase at
each peak, so if the assembled hexagon is wrong the overlaid lattice will not
sit on the blobs.

    python3 moire_verify.py FOLDER [-m MATERIAL] [--lo LO] [--hi HI]
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

from moire_prep import load_ibw, level, nlm
from moire_calc import analyze, MATERIALS
from moire_fit import find_moire
from paths import results_for, sample_of


def fft_of(path, channel="LateralTrace"):
    """Return the leveled and denoised images plus the spectrum of the latter.

    Peak finding and the Fourier phase always come from the denoised image, so
    the fitted lattice is identical whichever background is displayed. Only the
    backdrop changes.
    """
    raw, scan, _ = load_ibw(path, channel)
    lev = level(raw)
    den = nlm(lev, 1.15)
    n = den.shape[0]
    px = scan * 1e9 / n
    win = np.outer(np.hanning(n), np.hanning(n))
    spec = np.fft.fftshift(np.fft.fft2(den * win))
    return lev, den, spec, n, px, scan * 1e9




def phase_at(spec, n, px, b):
    """Complex Fourier phase at reciprocal vector b (nearest bin)."""
    c = n // 2
    x = int(round(b[0] * n * px + c))
    y = int(round(b[1] * n * px + c))
    x = min(max(x, 0), n - 1)
    y = min(max(y, 0), n - 1)
    return np.angle(spec[y, x])


def lattice_points(b1, b2, spec, n, px, extent, margin=0.0):
    """Reconstruct moire maxima positions from peak vectors + Fourier phase."""
    basis = np.column_stack([b1, b2])
    real = np.linalg.inv(basis).T          # columns are real-space primitives
    a1, a2 = real[:, 0], real[:, 1]
    ph = np.array([phase_at(spec, n, px, b1), phase_at(spec, n, px, b2)])
    # solve b_i . r0 = -phi_i / 2pi  for the lattice origin
    origin = np.linalg.solve(basis.T, -ph / (2 * np.pi))
    # x2: the (m, k) index square maps to a parallelogram in real space, which
    # would otherwise miss the corners of the requested box
    reach = 2 * int((extent + 2 * margin) /
                    min(np.linalg.norm(a1), np.linalg.norm(a2))) + 2
    pts = []
    for m in range(-reach, reach + 1):
        for k in range(-reach, reach + 1):
            p = origin + m * a1 + k * a2
            if (-margin <= p[0] <= extent + margin and
                    -margin <= p[1] <= extent + margin):
                pts.append(p)
    return np.array(pts), a1, a2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="an .ibw file or a folder of them")
    ap.add_argument("-o", "--outdir", default=None,
                    help="where figures go (default: project results/<sample>)")
    ap.add_argument("-m", "--material", default="MoS2", choices=list(MATERIALS))
    ap.add_argument("--lo", type=float, default=None,
                    help="min |k| (1/nm); default derived from the scan geometry")
    ap.add_argument("--hi", type=float, default=None,
                    help="max |k| (1/nm); default derived from the scan geometry")
    ap.add_argument("--lmin", type=float, default=None,
                    help="only keep fits with moire period above this (nm)")
    ap.add_argument("--lmax", type=float, default=None,
                    help="only keep fits with moire period below this (nm)")
    ap.add_argument("--smax", type=float, default=None,
                    help="only keep fits below this heterostrain (%%). Off by "
                         "default: the fitter already rejects >5 %% as not one moire")
    ap.add_argument("--background", choices=["leveled", "denoised"], default="leveled",
                    help="image the lattice is drawn on. leveled (default) is the\n"
                         "conservative choice: NLM reinforces periodic structure, so\n"
                         "checking a lattice against a denoised image is partly circular")
    args = ap.parse_args()

    a_lat = MATERIALS[args.material]
    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.ibw")))
    else:
        files = [args.path]
    hits = []
    for path in files:
        try:
            lev, den, spec, n, px, size = fft_of(path)
            b1, b2, res = find_moire(spec, n, px, a_lat, scan_nm=size,
                                     klo=args.lo, khi=args.hi)
            if b1 is None or not res["solutions"]:
                continue
            sol = res["solutions"][0]
            if args.smax is not None and sol["hetero"] * 100 > args.smax:
                continue
            if args.lmin is not None and res["lmean"] < args.lmin:
                continue
            if args.lmax is not None and res["lmean"] > args.lmax:
                continue
            base = os.path.basename(path)
            m = re.search(r"(\d+L\d+)", base) or re.search(r"(\d{3,})\.ibw", base)
            hits.append(dict(tag=m.group(1) if m else os.path.splitext(base)[0],
                             lev=lev, den=den, spec=spec, n=n, px=px, size=size,
                             b1=b1, b2=b2, res=res, sol=sol))
        except Exception:
            continue

    if not hits:
        print("no scans matched")
        return
    print(f"{len(hits)} scan(s) fitted")
    for h in hits:
        print(f"  {h['tag']}: L={h['res']['lmean']:.2f} nm  "
              f"twist={h['sol']['twist']:.2f} deg  "
              f"strain={h['sol']['hetero']*100:.2f} %")

    BG = "lev" if args.background == "leveled" else "den"
    for kind in ("fft", "real"):
        ncol = min(5, len(hits))
        nrow = int(np.ceil(len(hits) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.6 * nrow))
        axs = np.atleast_1d(axs).ravel()
        for ax, h in zip(axs, hits):
            b1, b2 = h["b1"], h["b2"]
            b3 = -(b1 + b2)
            if kind == "fft":
                mag = np.abs(h["spec"])
                ext = 1 / (2 * h["px"])
                ax.imshow(np.log1p(mag), cmap="magma",
                          extent=[-ext, ext, -ext, ext], origin="lower")
                hexpts = [b1, -b3, b2, -b1, b3, -b2, b1]
                hx = [p[0] for p in hexpts]
                hy = [p[1] for p in hexpts]
                ax.plot(hx, hy, "-", color="cyan", lw=1.0, alpha=0.9)
                for p in (b1, b2, b3):
                    for s in (1, -1):
                        ax.plot(s * p[0], s * p[1], "o", ms=11, mfc="none",
                                mec="cyan", mew=1.5)
                lim = max(np.hypot(*b1), np.hypot(*b2), np.hypot(*b3)) * 2.2
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_xlabel("1/nm", fontsize=7)
            else:
                ax.imshow(h[BG], cmap="afmhot",
                          extent=[0, h["size"], 0, h["size"]], origin="lower")
                pts, a1, a2 = lattice_points(b1, b2, h["spec"], h["n"], h["px"],
                                             h["size"])
                if len(pts):
                    ax.plot(pts[:, 0], pts[:, 1], "o", ms=4.5, mfc="none",
                            mec="cyan", mew=1.1)
                ax.set_xlabel("nm", fontsize=7)
            ax.set_title(f"{h['tag']}  L={h['res']['lmean']:.1f}nm  "
                         f"{h['sol']['twist']:.2f}°  {h['sol']['hetero']*100:.2f}%",
                         fontsize=8)
            ax.tick_params(labelsize=6)
        for ax in axs[len(hits):]:
            ax.axis("off")
        outdir = args.outdir or results_for(sample_of(args.path))
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, f"moire_lattices_{kind}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=105)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
