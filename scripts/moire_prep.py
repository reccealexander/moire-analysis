#!/usr/bin/env python3
"""
Batch-preprocess Asylum Research .ibw SPM scans for moire lattice fitting in Gwyddion.

Homebrew Gwyddion is built without pygwy, so Gwyddion itself cannot be scripted.
This script performs every step up to the lattice fit -- including non-local-means
denoising, which Gwyddion has no equivalent for -- and writes a calibrated .gwy
per scan.

Open a .gwy in Gwyddion and run Measure Lattice on the "FFT" channel, placing the
two vectors by hand on adjacent moire peaks (Estimate tends to grab noise). The
"denoised" channel is the real-space cross-check. Fitting the ACF is NOT
recommended: residual line noise forms a ridge that dominates its maxima.

Leveling reproduces Gwyddion's own pipeline -- Align Rows (Median) then Remove
Polynomial Background with independent degree 3 -- to 0.24 % rms, verified
against a Gwyddion-processed file.

Channels written per file:
    raw        original channel, unmodified
    leveled    align-rows (median) + degree-3 tensor polynomial background
    denoised   leveled + non-local means
    ACF        autocorrelation of denoised, central half (diagnostic only)
    FFT        log |FFT| of denoised   <-- fit Measure Lattice here

Usage:
    python3 moire_prep.py PATH [-c CHANNEL] [-o OUTDIR] [--h-factor F]

PATH is a single .ibw file or a folder containing them.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
from scipy import ndimage
from igor2 import binarywave as bw
from skimage.restoration import denoise_nl_means, estimate_sigma
from gwyfile.objects import GwyContainer, GwyDataField

from paths import GWY, sample_of


def load_ibw(path, channel):
    """Return (image, scan_size_m, label) for the requested channel, or None."""
    data = bw.load(path)
    wave = data["wave"]
    arr = np.array(wave["wData"]).astype(float)
    labels = [b.decode("ascii", "ignore") for chunk in wave["labels"] for b in chunk if b]
    note = wave["note"]
    note = note.decode("ascii", "ignore") if isinstance(note, bytes) else str(note)

    match = re.search(r"ScanSize[:=]\s*([\d.eE+-]+)", note)
    if arr.ndim != 3 or match is None:
        return None
    scan = float(match.group(1))

    idx = next((i for i, l in enumerate(labels) if channel.lower() in l.lower()), None)
    if idx is None or idx >= arr.shape[2]:
        return None
    # Gwyddion's .ibw importer transposes and flips vertically relative to the
    # raw igor2 array. Matching it matters: without this the fast-scan axis is
    # swapped, so median row alignment removes column offsets instead of the
    # actual scan-line offsets and the line noise survives. Verified against a
    # Gwyddion-processed file at correlation 1.00000.
    return np.flipud(arr[:, :, idx].T), scan, labels[idx]


def level(img, destripe=False, xdeg=3, ydeg=3, poly_mode="tensor"):
    """Median row alignment, then polynomial background removal.

    Row alignment matches Gwyddion's Align Rows -> Median: each row has its own
    median subtracted.

    poly_mode="tensor" matches Gwyddion's Remove Polynomial Background with
    independent degrees: the tensor-product basis {x^i y^j : i<=xdeg, j<=ydeg}.
    Degree 3/3 (16 terms) reproduces Gwyddion to 0.24% rms; degree 2 is 6.5% off.
    poly_mode="total" uses only terms with i+j <= max degree, a smaller basis.

    destripe additionally subtracts column medians, which kills vertical
    striping. Use with care: it also removes any moire component whose
    wavevector lies exactly along the horizontal axis.
    """
    img = img - np.median(img, axis=1, keepdims=True)
    if destripe:
        img = img - np.median(img, axis=0, keepdims=True)

    ny, nx = img.shape
    # normalise to [-1, 1] so the high powers stay well conditioned
    x = np.linspace(-1.0, 1.0, nx)[None, :]
    y = np.linspace(-1.0, 1.0, ny)[:, None]
    cols = []
    for i in range(xdeg + 1):
        for j in range(ydeg + 1):
            if poly_mode == "total" and i + j > max(xdeg, ydeg):
                continue
            cols.append((x ** i * y ** j * np.ones((ny, nx))).ravel())
    basis = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(basis, img.ravel(), rcond=None)
    return img - (basis @ coef).reshape(img.shape)


def nlm(img, h_factor):
    """Non-local means, the denoiser used in the Science SI (Gwyddion lacks it)."""
    scale = img.std()
    if scale == 0:
        return img
    z = (img - img.mean()) / scale
    sigma = float(estimate_sigma(z))
    out = denoise_nl_means(
        z, h=h_factor * sigma, sigma=sigma,
        patch_size=5, patch_distance=11, fast_mode=True,
    )
    return out * scale + img.mean()


def autocorr(img):
    spec = np.fft.fft2(img - img.mean())
    out = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(spec) ** 2)))
    return out / out.max()


def fft_modulus(img):
    ny, nx = img.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    spec = np.abs(np.fft.fftshift(np.fft.fft2((img - img.mean()) * win)))
    return np.log1p(spec / spec.max() * 1e4)


def field(data, xreal, yreal, unit_xy="m", unit_z="V"):
    return GwyDataField(
        np.ascontiguousarray(data, dtype=float),
        xreal=xreal, yreal=yreal,
        si_unit_xy=unit_xy, si_unit_z=unit_z,
    )


def process(path, channel, outdir, h_factor, destripe=False, poly_mode="tensor"):
    got = load_ibw(path, channel)
    if got is None:
        return None
    raw, scan, label = got
    ny, nx = raw.shape

    lev = level(raw, destripe, poly_mode=poly_mode)
    den = nlm(lev, h_factor)
    ac = autocorr(den)
    ft = fft_modulus(den)

    # ACF: keep the central half so moire spots dominate the frame
    cy, cx = ny // 2, nx // 2
    ry, rx = ny // 4, nx // 4
    ac = ac[cy - ry:cy + ry, cx - rx:cx + rx]
    acf_real = scan * (2 * ry) / ny

    container = GwyContainer()
    entries = [
        ("raw", field(raw, scan, scan)),
        ("leveled", field(lev, scan, scan)),
        ("denoised", field(den, scan, scan)),
        ("ACF", field(ac, acf_real, acf_real, unit_z="")),
        ("FFT", field(ft, nx / scan, ny / scan, unit_xy="m^-1", unit_z="")),
    ]
    for i, (title, fld) in enumerate(entries):
        container[f"/{i}/data"] = fld
        container[f"/{i}/data/title"] = title

    os.makedirs(outdir, exist_ok=True)
    dest = os.path.join(outdir, os.path.splitext(os.path.basename(path))[0] + ".gwy")
    container.tofile(dest)
    return dest, scan, label, ny


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="an .ibw file or a folder of them")
    ap.add_argument("-c", "--channel", default="LateralTrace",
                    help="channel name substring (default: LateralTrace)")
    ap.add_argument("-o", "--outdir", default=None,
                    help="output folder (default: <path>/gwy)")
    ap.add_argument("--h-factor", type=float, default=1.15,
                    help="NLM strength in units of noise sigma (default: 1.15)")
    ap.add_argument("--poly-mode", choices=["tensor", "total"], default="tensor",
                    help="tensor matches Gwyddion's Remove Polynomial Background "
                         "(default); total uses the smaller i+j<=deg basis")
    ap.add_argument("--destripe", action="store_true",
                    help="also subtract column medians to remove vertical striping")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.ibw")))
        outdir = args.outdir or os.path.join(GWY, sample_of(args.path))
    else:
        files = [args.path]
        outdir = args.outdir or os.path.join(GWY, sample_of(args.path))

    if not files:
        sys.exit(f"no .ibw files found in {args.path}")

    done = skipped = 0
    for path in files:
        try:
            result = process(path, args.channel, outdir, args.h_factor, args.destripe, args.poly_mode)
        except Exception as exc:
            print(f"  !! {os.path.basename(path)}: {exc}")
            skipped += 1
            continue
        if result is None:
            skipped += 1
            continue
        dest, scan, label, npx = result
        print(f"  {os.path.basename(path):26s} {label:18s} "
              f"{scan * 1e9:7.1f} nm  {npx}px  ->  {os.path.basename(dest)}")
        done += 1

    print(f"\n{done} written to {outdir}   ({skipped} skipped: no {args.channel} channel)")
    print("Open a .gwy in Gwyddion, select the FFT channel, run Measure Lattice.")


if __name__ == "__main__":
    main()
