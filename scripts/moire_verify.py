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


def find_hexagon(spec, n, px, lo, hi):
    """Return (b1, b2) of the best first-order hexagon, in 1/nm."""
    mag = np.abs(spec)
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    fx = (xx - c) / (n * px)
    fy = (yy - c) / (n * px)
    rad = np.hypot(fx, fy)

    peaks = peak_local_max(np.where((rad > lo) & (rad < hi), mag, 0),
                           min_distance=3, num_peaks=25)
    cands = []
    for y, x in peaks:
        w = 2
        sub = mag[y - w:y + w + 1, x - w:x + w + 1]
        if sub.shape != (2 * w + 1, 2 * w + 1):
            continue
        gy, gx = np.mgrid[-w:w + 1, -w:w + 1]
        tot = sub.sum()
        cands.append((np.array([(x + (sub * gx).sum() / tot - c) / (n * px),
                                (y + (sub * gy).sum() / tot - c) / (n * px)]),
                      mag[y, x]))

    def amp(v):
        x = int(round(v[0] * n * px + c))
        y = int(round(v[1] * n * px + c))
        if 0 <= x < n and 0 <= y < n:
            return mag[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max()
        return 0.0

    best = None
    for i, (b1, a1) in enumerate(cands):
        for j, (b2, a2) in enumerate(cands):
            if i == j:
                continue
            if np.hypot(*(b1 - b2)) < 0.02 or np.hypot(*(b1 + b2)) < 0.02:
                continue
            b3 = -(b1 + b2)
            k = [np.hypot(*b1), np.hypot(*b2), np.hypot(*b3)]
            if min(k) < lo or max(k) > hi or max(k) / min(k) > 1.5:
                continue
            score = min(a1, a2, amp(b3))
            if best is None or score > best[0]:
                best = (score, b1, b2)
    return (best[1], best[2]) if best else (None, None)


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
    ap.add_argument("folder")
    ap.add_argument("-o", "--outdir", default=None,
                    help="where figures go (default: project results/<sample>)")
    ap.add_argument("-m", "--material", default="MoS2", choices=list(MATERIALS))
    ap.add_argument("--lo", type=float, default=0.025, help="min |k| (1/nm)")
    ap.add_argument("--hi", type=float, default=0.12, help="max |k| (1/nm)")
    ap.add_argument("--lmin", type=float, default=13.0)
    ap.add_argument("--lmax", type=float, default=21.0)
    ap.add_argument("--smax", type=float, default=1.5, help="max plausible strain %")
    ap.add_argument("--background", choices=["leveled", "denoised"], default="leveled",
                    help="image the lattice is drawn on. leveled (default) is the\n"
                         "conservative choice: NLM reinforces periodic structure, so\n"
                         "checking a lattice against a denoised image is partly circular")
    args = ap.parse_args()

    a_lat = MATERIALS[args.material]
    hits = []
    for path in sorted(glob.glob(os.path.join(args.folder, "*.ibw"))):
        try:
            lev, den, spec, n, px, size = fft_of(path)
            b1, b2 = find_hexagon(spec, n, px, args.lo, args.hi)
            if b1 is None:
                continue
            res = analyze(b1, b2, a_lat, "recip")
            if not res["solutions"]:
                continue
            sol = res["solutions"][0]
            if sol["hetero"] * 100 > args.smax:
                continue
            if not (args.lmin <= res["lmean"] <= args.lmax):
                continue
            hits.append(dict(tag=re.sub(r".*?(\d+)\.ibw", r"\1", os.path.basename(path)),
                             lev=lev, den=den, spec=spec, n=n, px=px, size=size,
                             b1=b1, b2=b2, res=res, sol=sol))
        except Exception:
            continue

    if not hits:
        print("no scans matched")
        return
    print(f"{len(hits)} scans matched")

    BG = "lev" if args.background == "leveled" else "den"
    for kind in ("fft", "real"):
        ncol = 5
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
        outdir = args.outdir or results_for(sample_of(args.folder))
        out = os.path.join(outdir, f"moire_lattices_{kind}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=105)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
