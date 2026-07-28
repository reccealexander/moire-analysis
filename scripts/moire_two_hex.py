#!/usr/bin/env python3
"""
Multi-hexagon moire fitter for multilayer stacks.

A single-hexagon fit fails on tri/tetralayers because several moire lattices
coexist in one FFT. This finds them one at a time: the strongest complete
hexagon (six peaks of comparable amplitude, 60 degrees apart, summing to zero in
triples) is fitted, its peaks are removed, and the search repeats on what is
left. Each surviving hexagon is decomposed into twist and heterostrain.

A bilayer should yield exactly one strong hexagon; a trilayer may yield two (the
two twisted interfaces), plus a longer-wavelength moire-of-moire beat.

    python3 moire_two_hex.py FILE.ibw [-m MATERIAL] [--max-hex N]
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


def spectrum(path, channel="LateralTrace"):
    raw, scan, _ = load_ibw(path, channel)
    n = raw.shape[0]
    px = scan * 1e9 / n
    den = nlm(level(raw), 1.15)
    F = np.abs(np.fft.fftshift(np.fft.fft2(den * np.outer(np.hanning(n), np.hanning(n)))))
    return F, n, px


def sample(F, v, n, px):
    c = n // 2
    x = int(round(v[0] * n * px + c))
    y = int(round(v[1] * n * px + c))
    if 0 <= x < n and 0 <= y < n:
        return F[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max()
    return 0.0


def candidates(F, n, px, klo, khi):
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    fx = (xx - c) / (n * px)
    fy = (yy - c) / (n * px)
    rad = np.hypot(fx, fy)
    pk = peak_local_max(np.where((rad > klo) & (rad < khi), F, 0),
                        min_distance=3, num_peaks=40)
    out = []
    for y, x in pk:
        w = 2
        sub = F[y - w:y + w + 1, x - w:x + w + 1]
        if sub.shape != (2 * w + 1, 2 * w + 1):
            continue
        gy, gx = np.mgrid[-w:w + 1, -w:w + 1]
        t = sub.sum()
        out.append((np.array([(x + (sub * gx).sum() / t - c) / (n * px),
                              (y + (sub * gy).sum() / t - c) / (n * px)]), F[y, x]))
    return out


def best_hexagon(cands, F, n, px, klo, khi, used_spots, min_sep, a_lat,
                 max_strain=5.0):
    """Strongest geometrically regular hexagon.

    A real moire hexagon is nearly regular, so its decomposed heterostrain is
    small; a hexagon mis-assembled from peaks of two different moires decodes to
    a huge (>5 %) strain. Rejecting those first stops the fitter locking onto the
    distorted 0.4 deg / 16 % artifact.
    """
    best = None
    for i in range(len(cands)):
        for j in range(len(cands)):
            if i == j:
                continue
            b1, _ = cands[i]
            b2, _ = cands[j]
            if np.linalg.norm(b1 + b2) > np.linalg.norm(b1 - b2):
                b2 = -b2
            b3 = -(b1 + b2)
            ks = [np.linalg.norm(v) for v in (b1, b2, b3)]
            if min(ks) < klo or max(ks) > khi or max(ks) / min(ks) > 1.25:
                continue
            spots = [b1, b2, b3, -b1, -b2, -b3]
            if used_spots and any(
                    min(np.linalg.norm(s - u) for u in used_spots) < min_sep
                    for s in spots):
                continue
            res = analyze(b1, b2, a_lat, "recip")
            if not res["solutions"] or res["solutions"][0]["hetero"] * 100 > max_strain:
                continue
            amps = [sample(F, s, n, px) for s in spots]
            score = min(amps)                 # require a complete hexagon
            if best is None or score > best[0]:
                best = (score, b1, b2, b3, spots)
    return best


def fit_scan(path, a_lat, klo, khi, max_hex):
    F, n, px = spectrum(path)
    cands = candidates(F, n, px, klo, khi)
    floor = np.median([a for _, a in cands]) if cands else 0.0
    used = []
    hexes = []
    for _ in range(max_hex):
        b = best_hexagon(cands, F, n, px, klo, khi, used, 0.03, a_lat)
        if b is None:
            break
        score, b1, b2, b3, spots = b
        if score < 1.4 * floor:               # too weak to be a real hexagon
            break
        res = analyze(b1, b2, a_lat, "recip")
        sol = res["solutions"][0] if res["solutions"] else None
        hexes.append(dict(b1=b1, b2=b2, b3=b3, spots=spots, score=score,
                          L=res["lmean"], sol=sol))
        used.extend(spots)
    return F, n, px, hexes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="an .ibw file or a folder")
    ap.add_argument("-m", "--material", default="WSe2", choices=list(MATERIALS))
    ap.add_argument("--klo", type=float, default=0.20)
    ap.add_argument("--khi", type=float, default=0.36)
    ap.add_argument("--max-hex", type=int, default=3)
    ap.add_argument("-o", "--outdir", default=None,
                    help="where the figure goes (default: project results/<sample>)")
    args = ap.parse_args()

    a_lat = MATERIALS[args.material]
    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.ibw")))
    else:
        files = [args.path]

    results = []
    for path in files:
        F, n, px, hexes = fit_scan(path, a_lat, args.klo, args.khi, args.max_hex)
        tag = re.search(r"(\dL\d+|\d+)\.ibw", os.path.basename(path))
        tag = tag.group(1) if tag else os.path.basename(path)
        results.append((tag, F, n, px, hexes))
        print(f"{tag}: {len(hexes)} hexagon(s)")
        for h in hexes:
            ang = np.degrees(np.arctan2(h["b1"][1], h["b1"][0])) % 60
            if h["sol"]:
                print(f"    L={h['L']:.2f} nm  twist={h['sol']['twist']:.2f} deg  "
                      f"strain={h['sol']['hetero']*100:.1f}%  orient~{ang:.0f} deg  "
                      f"(rel.strength {h['score']:.0f})")

    # figure: each scan's FFT with the found hexagons in distinct colours
    colors = ["cyan", "lime", "orange"]
    ncol = min(4, len(results))
    nrow = int(np.ceil(len(results) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 4.1 * nrow))
    axs = np.atleast_1d(axs).ravel()
    for ax, (tag, F, n, px, hexes) in zip(axs, results):
        ext = 1 / (2 * px)
        ax.imshow(np.log1p(F), cmap="magma", extent=[-ext, ext, -ext, ext], origin="lower")
        for h, col in zip(hexes, colors):
            pts = [h["b1"], -h["b3"], h["b2"], -h["b1"], h["b3"], -h["b2"], h["b1"]]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col, lw=1.0)
            for s in h["spots"]:
                ax.plot(s[0], s[1], "o", ms=10, mfc="none", mec=col, mew=1.5)
        tw = " / ".join(f"{h['sol']['twist']:.1f}" for h in hexes if h["sol"])
        ax.set_title(f"{tag}   twist(s): {tw} deg", fontsize=9)
        ax.set_xlim(-0.42, 0.42)
        ax.set_ylim(-0.42, 0.42)
        ax.set_xlabel("1/nm", fontsize=7)
    for ax in axs[len(results):]:
        ax.axis("off")
    plt.suptitle(f"{args.material} multilayer - each moire hexagon fitted separately",
                 fontsize=12)
    plt.tight_layout()
    outdir = args.outdir or results_for(sample_of(args.path))
    out = os.path.join(outdir, "moire_two_hex.png")
    plt.savefig(out, dpi=110)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
