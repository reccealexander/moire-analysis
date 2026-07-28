#!/usr/bin/env python3
"""
Single source of truth for locating a moire hexagon in an FFT.

This exists because two near-duplicate finders drifted apart: one used a 1.5
regularity ratio with no strain filter, the other 1.25 with a 5 % filter. They
disagreed badly on MoS2 (tight 1.07 deg cluster vs values scattered over
0.1-1.7 deg). Sweeping both knobs against samples with known answers gave one
setting that works for both:

    ratio 1.5   a real hexagon can be visibly distorted, because heterostrain is
                amplified by ~1/theta. 1.25 is too strict: at small twist the
                true hexagon gets rejected and a rounder noise triple wins.
    strain 5 %  rejects hexagons assembled from peaks belonging to *different*
                moires, which is how multilayer WSe2 otherwise decodes to a
                spurious ~0.4 deg / 16 % root.

Neither knob alone is enough - dropping the filter breaks WSe2, tightening the
ratio breaks MoS2.
"""
import numpy as np
from skimage.feature import peak_local_max

from moire_calc import analyze

RATIO_MAX = 1.5          # max / min of the three |k| in one hexagon
STRAIN_MAX = 5.0         # per cent; above this the triple is not one moire
NUM_PEAKS = 25


def auto_band(scan_nm, n_px, min_px_per_period=8.0, min_periods=5.0):
    """Reciprocal search band (1/nm) implied by the scan geometry alone.

    A moire must be coarse enough to resolve (at least a few pixels per period)
    and fine enough to repeat within the frame. That brackets |k| without any
    per-sample constant, which is what lets an arbitrary scan be analysed.

    min_periods=5 was tuned against 15 scans with independently known answers:
    at 3.5 the low end is loose enough that long-wavelength background wins on
    weak scans (one MoS2 scan fitted 48.9 nm / 0.04 deg instead of 16.5 nm /
    1.10 deg). 5 fixes that and 6-8 add nothing.

    A scan carrying several periodicities can still be fitted to the wrong one;
    pass klo/khi explicitly for those.
    """
    px = scan_nm / n_px
    l_min = min_px_per_period * px
    l_max = scan_nm / min_periods
    return 2.0 / (np.sqrt(3.0) * l_max), 2.0 / (np.sqrt(3.0) * l_min)


def candidates(mag, n, px, klo, khi, num_peaks=NUM_PEAKS):
    """Subpixel-refined FFT peaks inside the annulus, brightest first."""
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    fx = (xx - c) / (n * px)
    fy = (yy - c) / (n * px)
    rad = np.hypot(fx, fy)
    peaks = peak_local_max(np.where((rad > klo) & (rad < khi), mag, 0),
                           min_distance=3, num_peaks=num_peaks)
    out = []
    for y, x in peaks:
        w = 2
        sub = mag[y - w:y + w + 1, x - w:x + w + 1]
        if sub.shape != (2 * w + 1, 2 * w + 1):
            continue
        gy, gx = np.mgrid[-w:w + 1, -w:w + 1]
        tot = sub.sum()
        out.append((np.array([(x + (sub * gx).sum() / tot - c) / (n * px),
                              (y + (sub * gy).sum() / tot - c) / (n * px)]),
                    mag[y, x]))
    return out


def _amp(mag, v, n, px):
    c = n // 2
    x = int(round(v[0] * n * px + c))
    y = int(round(v[1] * n * px + c))
    if 0 <= x < n and 0 <= y < n:
        return mag[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max()
    return 0.0


def best_hexagon(cands, mag, n, px, klo, khi, a_lat, used_spots=(),
                 min_sep=0.03, ratio_max=RATIO_MAX, strain_max=STRAIN_MAX):
    """Strongest first-order hexagon, scored by its weakest member.

    Scoring on the weakest of the six spots demands a *complete* hexagon rather
    than one bright peak plus noise. used_spots lets a caller exclude an
    already-claimed hexagon and search for a second, independent moire.
    """
    best = None
    for i, (b1, a1) in enumerate(cands):
        for j, (b2, a2) in enumerate(cands):
            if i == j:
                continue
            # adopt the 120-degree convention so b1 + b2 + b3 = 0
            v2 = -b2 if np.linalg.norm(b1 + b2) > np.linalg.norm(b1 - b2) else b2
            b3 = -(b1 + v2)
            ks = [np.linalg.norm(v) for v in (b1, v2, b3)]
            if min(ks) < klo or max(ks) > khi or max(ks) / min(ks) > ratio_max:
                continue
            spots = [b1, v2, b3, -b1, -v2, -b3]
            if len(used_spots) and any(
                    min(np.linalg.norm(s - u) for u in used_spots) < min_sep
                    for s in spots):
                continue
            res = analyze(b1, v2, a_lat, "recip")
            if not res["solutions"]:
                continue
            if res["solutions"][0]["hetero"] * 100.0 > strain_max:
                continue
            score = min(a1, a2, _amp(mag, b3, n, px))
            if best is None or score > best[0]:
                best = (score, b1, v2, b3, spots, res)
    return best


def find_moire(spec, n, px, a_lat, scan_nm=None, klo=None, khi=None):
    """Convenience wrapper: complex spectrum in, (b1, b2, result) out.

    The band is derived from the scan geometry unless given explicitly.
    Returns (None, None, None) when no acceptable hexagon exists.
    """
    mag = np.abs(spec)
    if klo is None or khi is None:
        if scan_nm is None:
            scan_nm = n * px
        klo, khi = auto_band(scan_nm, n)
    best = best_hexagon(candidates(mag, n, px, klo, khi), mag, n, px,
                        klo, khi, a_lat)
    if best is None:
        return None, None, None
    return best[1], best[2], best[5]
