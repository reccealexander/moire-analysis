#!/usr/bin/env python3
"""
Local strain mapping figure in the format of Science ade9995 Fig. S13.

Formatting matched to the published figure: per-site hexagonal tiles (not a
smooth field), RdBu_r diverging colormap centred on zero, a horizontal
"epsilon (%)" colorbar, x (nm) on the vertical axis and y (nm) on the
horizontal, bold panel letters with colour-coded labels, and a panel A locating
each map.

The paper derives strain from next-neighbor site distances. That needs cleanly
resolvable sites, which these low-contrast LFM scans do not give (site detection
recovers ~20 % of the expected sites). The same quantity is obtained here from
the geometric phase instead: the local moire wavelength L(r) comes from the
phase gradient, and

    epsilon(r) = theta * (L(r) - <L>) / <L>

is the layer heterostrain whose moire amplification produces the observed local
wavelength deviation, with theta the fitted twist in radians. Tiles are laid on
the reconstructed moire lattice, optionally subdivided, and each Voronoi cell is
coloured by epsilon at its centre.

Tile spacing is display density, not resolution: the real resolution is set by
the Bragg mask width (--sigma-frac). Subdividing is free; widening the mask is
not - it inflates noise and leaks the neighbouring Bragg peak.

    python3 strainmap_figS13.py --sample MoS2_2L
    python3 strainmap_figS13.py --sample WSe2
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from scipy.spatial import Voronoi

from moire_prep import load_ibw, level, nlm
from moire_verify import find_hexagon, lattice_points
from moire_two_hex import candidates, best_hexagon
from moire_calc import analyze, MATERIALS
from moire_map import bragg_phase, local_wavevectors
from paths import RAW, GRID_FITS, results_for

SAMPLES = {
    # MoS2: one moire per scan, weak peaks close to DC, positions on a 3x3 grid
    "MoS2_2L": dict(material="MoS2", klo=0.025, khi=0.12, max_strain=None,
                    layout="grid", title=r"2L MoS$_2$, 3$\times$3 grid"),
    # WSe2: strong peaks near 0.27 1/nm. max_strain rejects the geometrically
    # irregular hexagons that otherwise decode to the bogus ~0.4 deg / 16 % root
    "WSe2": dict(material="WSe2", klo=0.20, khi=0.36, max_strain=5.0,
                 layout="layers", title=r"WSe$_2$ multilayers"),
}

ROW = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2}
COL = {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2, 7: 0, 8: 1, 9: 2}
ROMAN = {1: "i", 2: "ii", 3: "iii", 4: "iv", 5: "v",
         6: "vi", 7: "vii", 8: "viii", 9: "ix"}
TINT = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd",
        "#8c564b", "#e377c2", "#17becf", "#7f7f7f"]

L_MIN, L_MAX, S_MAX = 13.0, 21.0, 1.5
# weakest tiles keep this much opacity, so they are de-emphasised, not erased
ALPHA_FLOOR = 0.5


def entries_for(sample):
    """One panel entry per scan: path, label, colour, and panel slot."""
    out = []
    if sample == "MoS2_2L":
        fits = json.load(open(GRID_FITS))
        for g in range(1, 10):
            rows = [r for r in fits[str(g)]
                    if L_MIN <= r[1] <= L_MAX and r[3] < S_MAX]
            if not rows:
                continue
            med = np.median([r[2] for r in rows])
            num = int(min(rows, key=lambda r: abs(r[2] - med))[0])
            out.append(dict(
                path=os.path.join(RAW["MoS2_2L"], f"LFM_2LMoS2{num:04d}.ibw"),
                label=ROMAN[g], tag=f"{num:04d}", tint=TINT[g - 1],
                slot=(1 + ROW[g], COL[g]), group=None))
    else:
        paths = sorted(glob.glob(os.path.join(RAW["WSe2"], "*.ibw")))
        for i, p in enumerate(paths):
            m = re.search(r"_(\d)L(\d+)\.ibw", os.path.basename(p))
            layer, num = (m.group(1), m.group(2)) if m else ("?", str(i))
            out.append(dict(path=p, label=f"{layer}L{num}", tag=f"{layer}L{num}",
                            tint=TINT[i % len(TINT)], slot=None,
                            group=f"{layer}L"))
    return out


def pick_hexagon(spec_w, n, px, cfg, a_lat):
    """Two moire vectors, using the strain filter where the sample needs it."""
    if cfg["max_strain"] is None:
        return find_hexagon(spec_w, n, px, cfg["klo"], cfg["khi"])
    F = np.abs(spec_w)
    cands = candidates(F, n, px, cfg["klo"], cfg["khi"])
    b = best_hexagon(cands, F, n, px, cfg["klo"], cfg["khi"], [], 0.03,
                     a_lat, cfg["max_strain"])
    return (b[1], b[2]) if b else (None, None)


def strain_field(path, cfg, crop=0.08, subdiv=3, sigma_frac=0.35):
    """Signed heterostrain field plus the lattice used for tiling."""
    a_lat = MATERIALS[cfg["material"]]
    raw, scan, _ = load_ibw(path, "LateralTrace")
    n = raw.shape[0]
    px = scan * 1e9 / n
    size = scan * 1e9
    den = nlm(level(raw), 1.15)

    spec_w = np.fft.fftshift(np.fft.fft2(den * np.outer(np.hanning(n), np.hanning(n))))
    b1, b2 = pick_hexagon(spec_w, n, px, cfg, a_lat)
    if b1 is None:
        return None
    res = analyze(b1, b2, a_lat, "recip")
    if not res["solutions"]:
        return None
    theta = np.radians(res["solutions"][0]["twist"])

    spec = np.fft.fftshift(np.fft.fft2(den))
    G1 = bragg_phase(spec, b1, n, px, sigma_frac)
    G2 = bragg_phase(spec, b2, n, px, sigma_frac)
    b1x, b1y = local_wavevectors(G1, b1, px)
    b2x, b2y = local_wavevectors(G2, b2, px)
    b3x, b3y = -(b1x + b2x), -(b1y + b2y)
    amp = np.abs(G1) * np.abs(G2)

    with np.errstate(divide="ignore", invalid="ignore"):
        periods = [2.0 / (np.sqrt(3.0) * np.hypot(bx, by))
                   for bx, by in ((b1x, b1y), (b2x, b2y), (b3x, b3y))]
    L = np.mean(periods, axis=0)

    m = int(crop * n)
    sl = slice(m, n - m)
    Lc = L[sl, sl]
    eps = theta * (Lc - np.nanmean(Lc)) / np.nanmean(Lc) * 100.0
    ampc = amp[sl, sl]

    pts, a1, a2 = lattice_points(b1, b2, spec_w, n, px, size,
                                 margin=2.5 * res["lmean"])
    if subdiv > 1:
        pts = np.concatenate([pts + (i / subdiv) * a1 + (j / subdiv) * a2
                              for i in range(subdiv) for j in range(subdiv)])
    return dict(eps=eps, amp=ampc, px=px, m=m, n=n, size=size, pts=pts,
                L=res["lmean"], twist=res["solutions"][0]["twist"])


def voronoi_tiles(pts, lo, hi):
    """Hexagonal Voronoi tiles for the lattice points inside [lo, hi].

    Points outside the box are tessellated but not drawn: they bound the cells of
    the outermost drawn sites, which a naive clip would turn into large wedges.
    """
    vor = Voronoi(pts)
    inside = ((pts[:, 0] >= lo) & (pts[:, 0] <= hi) &
              (pts[:, 1] >= lo) & (pts[:, 1] <= hi))
    polys, centers = [], []
    for i in np.where(inside)[0]:
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            continue
        poly = np.clip(vor.vertices[region], lo, hi)
        if poly.shape[0] >= 3:
            polys.append(poly)
            centers.append(pts[i])
    if not polys:
        return [], np.empty((0, 2))
    areas = np.array([0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) -
                                np.dot(p[:, 1], np.roll(p[:, 0], 1)))
                      for p in polys])
    keep = areas < 2.5 * np.median(areas)
    return ([p for p, k in zip(polys, keep) if k], np.array(centers)[keep])


def panel_a(ax, sample, entries):
    if sample == "MoS2_2L":
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(2.6, -0.6)
        for e in entries:
            r, c = e["slot"][0] - 1, e["slot"][1]
            ax.add_patch(Rectangle((c - 0.42, r - 0.42), 0.84, 0.84,
                                   fill=False, ec=e["tint"], lw=1.6))
            ax.text(c, r, e["label"], color=e["tint"], fontsize=10,
                    style="italic", ha="center", va="center", fontweight="bold")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["-1.69", "-1.49", "-1.29"], fontsize=7)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["532", "732", "932"], fontsize=7)
        ax.set_xlabel("column coordinate", fontsize=7.5)
        ax.set_ylabel("row coordinate", fontsize=7.5)
        ax.set_title("scan positions", fontsize=8, pad=4)
    else:
        groups = sorted({e["group"] for e in entries})
        ax.set_xlim(-0.55, 2.3)
        ax.set_ylim(len(groups) - 0.45, -0.55)
        for gi, gname in enumerate(groups):
            for k, e in enumerate([x for x in entries if x["group"] == gname]):
                ax.add_patch(Rectangle((k * 0.8 - 0.33, gi - 0.26), 0.66, 0.52,
                                       fill=False, ec=e["tint"], lw=1.6))
                ax.text(k * 0.8, gi, e["label"][2:], color=e["tint"],
                        fontsize=8, ha="center", va="center", fontweight="bold")
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=9)
        ax.set_xticks([])
        ax.set_ylabel("stack", fontsize=7.5)
        ax.set_title("scans by layer count", fontsize=8, pad=4)
    ax.tick_params(length=2)
    ax.text(-0.30, 1.16, "A", transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="top")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="MoS2_2L", choices=list(SAMPLES))
    ap.add_argument("--vmax", type=float, default=None,
                    help="symmetric colour limit in %% (overrides --pct)")
    ap.add_argument("--pct", type=float, default=98.0,
                    help="percentile of |eps| that sets the colour limit "
                         "(default 99). Higher = wider scale = fewer saturated "
                         "tiles, closer to the pale look of the published figure")
    ap.add_argument("--subdiv", type=int, default=3,
                    help="tile-lattice subdivision; spacing becomes L/subdiv. "
                         "Display density only - samples the same field more "
                         "finely without adding noise (default 3)")
    ap.add_argument("--sigma-frac", type=float, default=0.35,
                    help="Bragg mask width in units of |k| (default 0.35). "
                         "Widening genuinely sharpens the field but inflates "
                         "noise and leaks the neighbouring peak")
    ap.add_argument("--amp-pct", type=float, default=25.0,
                    help="tiles below this percentile of local Bragg amplitude "
                         "fade out, since a weak moire signal gives an "
                         "unreliable phase gradient; 0 disables (default 25)")
    ap.add_argument("--crop", type=float, default=0.08)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = SAMPLES[args.sample]
    entries = entries_for(args.sample)
    fields, tiles = {}, {}
    for e in entries:
        f = strain_field(e["path"], cfg, args.crop, args.subdiv, args.sigma_frac)
        if f is None:
            print(f"  {e['tag']}: no hexagon, skipped")
            continue
        lo = f["m"] * f["px"]
        hi = (f["n"] - f["m"]) * f["px"]
        polys, centers = voronoi_tiles(f["pts"], lo, hi)
        if not len(centers):
            print(f"  {e['tag']}: no tiles, skipped")
            continue
        ei = np.clip(((centers[:, 1] - lo) / f["px"]).astype(int),
                     0, f["eps"].shape[0] - 1)
        ej = np.clip(((centers[:, 0] - lo) / f["px"]).astype(int),
                     0, f["eps"].shape[1] - 1)
        fields[e["tag"]] = (e, f, lo, hi)
        tiles[e["tag"]] = (polys, f["eps"][ei, ej], f["amp"][ei, ej])
        print(f"  {e['tag']}: L={f['L']:.2f} nm  twist={f['twist']:.2f} deg  "
              f"{len(polys)} tiles")

    if not fields:
        print("no usable scans")
        return

    allbits = np.concatenate([v for _, v, _ in tiles.values()])
    vmax = args.vmax or float(np.round(np.nanpercentile(np.abs(allbits), args.pct), 1))
    sat = 100.0 * np.mean(np.abs(allbits) > vmax)
    frac = 100.0 * np.mean(np.abs(allbits) > 0.3)
    Lm = np.mean([f["L"] for _, f, _, _ in fields.values()])
    spacing = Lm / args.subdiv
    fwhm = 2.355 / (2 * np.pi * args.sigma_frac * (2 / (np.sqrt(3) * Lm)))
    print(f"\ncolour limit +/- {vmax:.2f} % (p{args.pct:g})   median |eps| = "
          f"{np.median(np.abs(allbits)):.2f} %   {sat:.1f} % of tiles saturate")
    print(f"the paper uses +/- 0.3 %; {frac:.0f} % of these tiles exceed that")
    print(f"{int(np.mean([len(v) for _, v, _ in tiles.values()]))} tiles/panel   "
          f"spacing {spacing:.2f} nm   field FWHM ~{fwhm:.1f} nm "
          f"({fwhm/spacing:.1f}x oversampled, neighbouring tiles correlated)")

    if args.sample == "MoS2_2L":
        nrows, ratios = 4, [0.85, 1, 1, 1]
        cb_box = [0.52, 0.905, 0.20, 0.016]      # free space beside panel A
    else:
        nrows = int(np.ceil((len(fields) + 2) / 3))
        ratios = [1] * nrows
        free = [(r, c) for r in range(nrows) for c in range(3)][1:]
        for i, tag in enumerate(fields):
            fields[tag][0]["slot"] = free[i]
        cb_box = None                            # placed in the leftover cell
    fig_h = 3.0 + 2.9 * (nrows - 1)
    fig = plt.figure(figsize=(11.0, fig_h))
    # packed layouts put panels on the top row, so leave room for the title
    top = 0.94 if args.sample == "MoS2_2L" else 0.895
    gs = fig.add_gridspec(nrows, 3, height_ratios=ratios,
                          hspace=0.42, wspace=0.34,
                          left=0.085, right=0.965, top=top, bottom=0.055)
    panel_a(fig.add_subplot(gs[0, 0]), args.sample, entries)

    if cb_box is None:
        used = {tuple(fields[t][0]["slot"]) for t in fields} | {(0, 0)}
        spare = next((r, c) for r in range(nrows) for c in range(3)
                     if (r, c) not in used)
        cell = gs[spare[0], spare[1]].get_position(fig)
        cb_box = [cell.x0 + 0.02, cell.y0 + cell.height * 0.52,
                  cell.width * 0.85, 0.014]
    cax = fig.add_axes(cb_box)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                               norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([-vmax, 0, vmax])
    cb.set_ticklabels([f"{-vmax:g}", "0", f"{vmax:g}"])
    cb.ax.tick_params(labelsize=8, length=2, pad=1)
    cb.outline.set_linewidth(0.6)
    cax.set_title(r"$\varepsilon$ (%)", fontsize=9, pad=3)

    letters = "BCDEFGHIJ"
    for idx, (tag, (e, f, lo, hi)) in enumerate(fields.items()):
        ax = fig.add_subplot(gs[e["slot"][0], e["slot"][1]])
        polys, vals, amps = tiles[tag]
        # opacity tracks the local Bragg amplitude: where the moire signal is
        # weak the phase gradient is unreliable, so those tiles fade towards the
        # background instead of showing a confident-looking spurious spike
        cols = plt.cm.RdBu_r(plt.Normalize(-vmax, vmax)(vals))
        if args.amp_pct > 0:
            a0 = np.percentile(amps, args.amp_pct)
            a1 = np.percentile(amps, 85.0)
            w = np.clip((amps - a0) / max(a1 - a0, 1e-30), 0.0, 1.0)
            cols[:, 3] = ALPHA_FLOOR + (1.0 - ALPHA_FLOOR) * w
        pc = PolyCollection(polys, facecolors=cols,
                            edgecolors="white", linewidths=0.12)
        ax.add_collection(pc)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("y (nm)", fontsize=8)
        ax.set_ylabel("x (nm)", fontsize=8)
        ax.tick_params(labelsize=7, length=2)
        ax.set_title(f"({e['label']})", color=e["tint"], fontsize=10,
                     style="italic", fontweight="bold", pad=3)
        ax.text(-0.30, 1.14, letters[idx], transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="top")
        ax.text(0.02, 0.02, f"{f['twist']:.2f}°", transform=ax.transAxes,
                fontsize=6, color="0.35", va="bottom")

    fig.suptitle(f"Local strain mapping — {cfg['title']}", fontsize=12,
                 y=0.975 if args.sample == "MoS2_2L" else 0.962)
    out = args.out or os.path.join(results_for(args.sample),
                                   "figS13_strain_mapping.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
