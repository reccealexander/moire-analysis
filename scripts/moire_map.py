#!/usr/bin/env python3
"""
Spatial twist / heterostrain maps from a single moire scan (paper Figs S12/S13).

Instead of detecting discrete moire sites (unreliable on low-contrast LFM data),
this uses geometric phase analysis. Two moire Bragg peaks are isolated in the
FFT; the local phase of each, demodulated against its average wavevector, has a
gradient that gives the local moire reciprocal vector at every pixel. The two
local vectors are decomposed per pixel with the same A = E - theta*J relation
used for the global fit, yielding theta(x,y) and heterostrain(x,y).

Validation: the spatial mean of the maps must reproduce the global single-number
fit for the same scan. The script prints this check.

    python3 moire_map.py FILE.ibw [-m MATERIAL] [--sigma-frac F]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moire_prep import load_ibw, level, nlm
from moire_verify import find_hexagon
from moire_calc import analyze, MATERIALS, atomic_triple
from paths import results_for, sample_of


def bragg_phase(spec_shifted, g, n, px, sigma_frac=0.35):
    """Geometric phase field for one Bragg peak.

    Returns G(r) = A(r) exp(i P(r)), the complex analytic signal after
    demodulating by the average wavevector g. The local wavevector deviation
    is (1/2pi) grad(P), recovered analytically below to avoid phase wrapping.
    """
    c = n // 2
    # gaussian mask around +g, radius set to roughly half the peak spacing so it
    # captures local variation without leaking neighbouring peaks
    yy, xx = np.mgrid[0:n, 0:n]
    fx = (xx - c) / (n * px)
    fy = (yy - c) / (n * px)
    kmag = np.hypot(*g)
    # wider mask -> finer real-space resolution, but the neighbouring first-order
    # peak sits |k| away, so leakage grows as exp(-1/(2*sigma_frac^2))
    sig = sigma_frac * kmag
    mask = np.exp(-((fx - g[0]) ** 2 + (fy - g[1]) ** 2) / (2 * sig ** 2))
    filtered = np.fft.ifft2(np.fft.ifftshift(spec_shifted * mask))

    x = (np.arange(n) - c) * px
    X, Y = np.meshgrid(x, x)
    carrier = np.exp(-2j * np.pi * (g[0] * X + g[1] * Y))
    return filtered * carrier          # G(r)


def local_wavevectors(G, g, px):
    """Local reciprocal vector field b(r) = g + (1/2pi) grad(P)."""
    gy, gx = np.gradient(G, px)         # d/dy, d/dx
    # grad(P) = Im(conj(G) grad G) / |G|^2
    denom = (np.abs(G) ** 2) + 1e-30
    px_field = np.imag(np.conj(G) * gx) / denom / (2 * np.pi)
    py_field = np.imag(np.conj(G) * gy) / denom / (2 * np.pi)
    bx = g[0] + px_field
    by = g[1] + py_field
    return bx, by


def decompose_field(b1x, b1y, b2x, b2y, a_lat, phi0):
    """Per-pixel A = E - theta J -> twist(deg) and heterostrain(%) fields."""
    g_mag = 2.0 / (np.sqrt(3.0) * a_lat)
    g1, g2, _ = atomic_triple(phi0, g_mag)
    Ginv = np.linalg.inv(np.column_stack([g1, g2]))     # 2x2

    ny, nx = b1x.shape
    twist = np.empty((ny, nx))
    hetero = np.empty((ny, nx))
    # A = [b1 b2] @ Ginv, evaluated per pixel with vectorised algebra
    Bxx, Bxy = b1x, b2x     # top row of [b1 b2] is x-components
    Byx, Byy = b1y, b2y     # bottom row is y-components
    a11 = Bxx * Ginv[0, 0] + Bxy * Ginv[1, 0]
    a12 = Bxx * Ginv[0, 1] + Bxy * Ginv[1, 1]
    a21 = Byx * Ginv[0, 0] + Byy * Ginv[1, 0]
    a22 = Byx * Ginv[0, 1] + Byy * Ginv[1, 1]
    twist = np.degrees((a12 - a21) / 2.0)
    exx, eyy, exy = a11, a22, (a12 + a21) / 2.0
    # principal strain difference = 2*sqrt(((exx-eyy)/2)^2 + exy^2)
    hetero = 2.0 * np.sqrt(((exx - eyy) / 2.0) ** 2 + exy ** 2) * 100.0
    return twist, hetero


def global_phi0(b1, b2, a_lat):
    res = analyze(b1, b2, a_lat, "recip")
    if not res["solutions"]:
        return None, res
    return np.radians(res["solutions"][0]["phi0"]), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("-m", "--material", default="MoS2", choices=list(MATERIALS))
    ap.add_argument("--lo", type=float, default=0.025)
    ap.add_argument("--hi", type=float, default=0.12)
    ap.add_argument("-o", "--outdir", default=None,
                    help="where the map goes (default: project results/<sample>)")
    ap.add_argument("--crop", type=float, default=0.12,
                    help="fraction of each edge to discard (GPA is unreliable at borders)")
    args = ap.parse_args()

    a_lat = MATERIALS[args.material]
    raw, scan, _ = load_ibw(args.file, "LateralTrace")
    n = raw.shape[0]
    px = scan * 1e9 / n
    size = scan * 1e9
    den = nlm(level(raw), 1.15)

    spec = np.fft.fftshift(np.fft.fft2(den * np.outer(np.hanning(n), np.hanning(n))))
    b1, b2 = find_hexagon(spec, n, px, args.lo, args.hi)
    if b1 is None:
        print("no hexagon found - cannot map this scan")
        return
    phi0, res = global_phi0(b1, b2, a_lat)
    gsol = res["solutions"][0]
    print(f"global fit: L={res['lmean']:.2f} nm  twist={gsol['twist']:.3f} deg  "
          f"heterostrain={gsol['hetero']*100:.2f} %")

    # phase fields use the plain (unwindowed) spectrum
    spec_raw = np.fft.fftshift(np.fft.fft2(den))
    G1 = bragg_phase(spec_raw, b1, n, px)
    G2 = bragg_phase(spec_raw, b2, n, px)
    b1x, b1y = local_wavevectors(G1, b1, px)
    b2x, b2y = local_wavevectors(G2, b2, px)
    twist, hetero = decompose_field(b1x, b1y, b2x, b2y, a_lat, phi0)

    # discard borders
    m = int(args.crop * n)
    sl = slice(m, n - m)
    twist_c = twist[sl, sl]
    hetero_c = hetero[sl, sl]

    print(f"map mean : twist={np.nanmean(twist_c):.3f} deg  "
          f"heterostrain={np.nanmean(hetero_c):.2f} %   "
          f"(should match global fit)")
    print(f"map range: twist {np.nanpercentile(twist_c,2):.2f}..{np.nanpercentile(twist_c,98):.2f} deg  "
          f"strain {np.nanpercentile(hetero_c,2):.2f}..{np.nanpercentile(hetero_c,98):.2f} %")

    ext = [m * px, (n - m) * px, (n - m) * px, m * px]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].imshow(den[sl, sl], cmap="afmhot", extent=ext)
    axs[0].set_title("denoised LateralTrace")
    tm = np.nanmedian(twist_c)
    im1 = axs[1].imshow(twist_c, cmap="viridis", extent=ext,
                        vmin=tm - 0.25, vmax=tm + 0.25)
    axs[1].set_title("local twist (deg)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)
    im2 = axs[2].imshow(hetero_c, cmap="magma", extent=ext,
                        vmin=0, vmax=np.nanpercentile(hetero_c, 98))
    axs[2].set_title("local heterostrain (%)")
    plt.colorbar(im2, ax=axs[2], fraction=0.046)
    for ax in axs:
        ax.set_xlabel("nm")
    tag = os.path.splitext(os.path.basename(args.file))[0]
    plt.suptitle(f"{tag}  -  GPA strain map")
    plt.tight_layout()
    outdir = args.outdir or results_for(sample_of(args.file))
    out = os.path.join(outdir, f"strainmap_{tag}.png")
    plt.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
