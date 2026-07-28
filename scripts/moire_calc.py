#!/usr/bin/env python3
"""
Moire twist / heterostrain calculator.

Paste the two lattice vectors from Gwyddion's Measure Lattice tool and get the
twist angle, moire periods and heterostrain tensor.

    python3 moire_calc.py

Fit the ACF channel and the vectors are real-space moire vectors in nm; fit the
FFT channel and they are reciprocal vectors in 1/nm. Pick the matching mode --
the units label on the Measure Lattice panel says "nm" in both cases, because
Gwyddion does not invert the unit symbol for transformed images.

The decomposition is exact, not the small-angle linearisation: the relative
layer transform T is recovered from the moire vectors and polar-decomposed as
T = R(theta) * U, giving the twist from R and the strain from U.
"""
import tkinter as tk
from tkinter import ttk

import numpy as np

# in-plane lattice constants, nm
MATERIALS = {
    "WSe2": 0.3282,
    "MoS2": 0.3160,
    "WS2": 0.3153,
    "MoSe2": 0.3288,
    "graphene": 0.2461,
    "hBN": 0.2504,
}


def first_order_triple(b1, b2):
    """Return three first-order reciprocal vectors that sum to zero."""
    if np.linalg.norm(b1 + b2) > np.linalg.norm(b1 - b2):
        b2 = -b2
    return b1, b2, -(b1 + b2)


def atomic_triple(phi0, g_mag):
    return [
        g_mag * np.array([np.cos(phi0 + 2 * np.pi * i / 3),
                          np.sin(phi0 + 2 * np.pi * i / 3)])
        for i in range(3)
    ]


def transform_for(phi0, b1, b2, g_mag):
    """Relative layer transform T implied by a crystal orientation phi0."""
    g1, g2, _ = atomic_triple(phi0, g_mag)
    amat = np.column_stack([b1, b2]) @ np.linalg.inv(np.column_stack([g1, g2]))
    resid = np.eye(2) - amat
    if abs(np.linalg.det(resid)) < 1e-12:
        return None
    return np.linalg.inv(resid).T


def decompose(tmat):
    """Polar-decompose T = R(theta) U. Returns twist deg, strains, axis deg."""
    w, s, vt = np.linalg.svd(tmat)
    rot = w @ vt
    twist = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
    # U = V diag(s) V^T, so principal stretches are the singular values
    axis = np.degrees(np.arctan2(vt[0, 1], vt[0, 0])) % 180.0
    return twist, s[0] - 1.0, s[1] - 1.0, axis


def solve(b1, b2, a_lat):
    """Find orientations giving an area-preserving (det=1) layer transform."""
    g_mag = 2.0 / (np.sqrt(3.0) * a_lat)

    def constraint(phi0):
        g1, g2, _ = atomic_triple(phi0, g_mag)
        amat = np.column_stack([b1, b2]) @ np.linalg.inv(np.column_stack([g1, g2]))
        return np.linalg.det(np.eye(2) - amat) - 1.0

    # [g1 g2] changes under phi0 -> phi0 + 120 deg (the columns permute), so the
    # constraint has period 2*pi, not 2*pi/3. Scanning the short range misses roots.
    grid = np.linspace(0.0, 2 * np.pi, 4001)
    vals = np.array([constraint(p) for p in grid])
    roots = []
    for i in range(len(grid) - 1):
        if not np.isfinite(vals[i]) or not np.isfinite(vals[i + 1]):
            continue
        if vals[i] == 0.0 or vals[i] * vals[i + 1] < 0:
            lo, hi = grid[i], grid[i + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if constraint(lo) * constraint(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append(0.5 * (lo + hi))

    out = []
    for phi0 in roots:
        tmat = transform_for(phi0, b1, b2, g_mag)
        if tmat is None:
            continue
        twist, e1, e2, axis = decompose(tmat)
        out.append({
            "phi0": np.degrees(phi0), "twist": abs(twist),
            "e1": e1, "e2": e2, "hetero": abs(e1 - e2), "axis": axis,
        })
    out.sort(key=lambda d: d["hetero"])
    return out


def analyze(a1, a2, a_lat, mode):
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)

    if mode == "real":
        basis = np.column_stack([a1, a2])
        if abs(np.linalg.det(basis)) < 1e-12:
            raise ValueError("the two vectors are parallel - no lattice")
        recip = np.linalg.inv(basis).T
        b1, b2 = recip[:, 0], recip[:, 1]
    else:
        b1, b2 = a1, a2
        if abs(np.linalg.det(np.column_stack([b1, b2]))) < 1e-12:
            raise ValueError("the two vectors are parallel - no lattice")

    b1, b2, b3 = first_order_triple(b1, b2)
    periods = [2.0 / (np.sqrt(3.0) * np.linalg.norm(b)) for b in (b1, b2, b3)]
    lmean = float(np.mean(periods))

    # exact twist from the mean period, assuming no strain
    ratio = a_lat / (2.0 * lmean)
    twist_simple = np.degrees(2.0 * np.arcsin(np.clip(ratio, -1.0, 1.0)))

    ang = np.degrees(np.arccos(np.clip(
        np.dot(a1, a2) / np.linalg.norm(a1) / np.linalg.norm(a2), -1.0, 1.0)))

    return {
        "periods": periods, "lmean": lmean, "twist_simple": twist_simple,
        "phi_input": ang, "solutions": solve(b1, b2, a_lat),
        "spread": (max(periods) - min(periods)) / lmean,
    }


def format_report(res, a_lat, mode, material):
    L = res["periods"]
    lines = []
    lines.append(f"material {material}   a = {a_lat:.4f} nm")
    lines.append(f"input    {'real-space (ACF), nm' if mode=='real' else 'reciprocal (FFT), 1/nm'}")
    lines.append("")
    lines.append("MOIRE PERIODS")
    for i, v in enumerate(L, 1):
        lines.append(f"   L{i} = {v:8.3f} nm")
    lines.append(f"   mean = {res['lmean']:7.3f} nm     spread = {res['spread']*100:.1f} %")
    lines.append("")
    lines.append("TWIST ANGLE")
    lines.append(f"   from mean period      theta = {res['twist_simple']:.3f} deg")

    sols = res["solutions"]
    if sols:
        best = sols[0]
        lines.append(f"   full decomposition    theta = {best['twist']:.3f} deg")
        lines.append("")
        lines.append("HETEROSTRAIN  (area-preserving solution, smallest strain)")
        lines.append(f"   principal strains     {best['e1']*100:+.3f} %  /  {best['e2']*100:+.3f} %")
        lines.append(f"   heterostrain e1-e2  = {best['hetero']*100:.3f} %")
        lines.append(f"   principal axis      = {best['axis']:.1f} deg")
        if len(sols) > 1:
            others = ", ".join(f"{s['twist']:.2f}deg/{s['hetero']*100:.2f}%" for s in sols[1:4])
            lines.append(f"   other valid solutions: {others}")
    else:
        lines.append("   full decomposition failed - check the vectors")

    lines.append("")
    lines.append("SANITY CHECKS")
    phi = res["phi_input"]
    lines.append(f"   angle between input vectors = {phi:.2f} deg")
    lines.append(f"   period spread               = {res['spread']*100:.1f} %")

    if sols:
        best = sols[0]
        theta_rad = np.radians(best["twist"])
        ratio = best["hetero"] / theta_rad if theta_rad > 0 else float("inf")
        lines.append(f"   strain / twist ratio        = {ratio*100:.0f} %")
        lines.append("")
        lines.append("   A distorted hexagon is EXPECTED, not a bad fit: heterostrain")
        lines.append("   is amplified by ~1/theta, so period spread ~= eps/theta and the")
        lines.append("   vector angle drifts off 60/120 deg in proportion. At small twist")
        lines.append("   even 0.3 % strain gives ~20 % spread.")
        lines.append("")
        if best["hetero"] > 0.01:
            lines.append("   [WARN] heterostrain above 1 %, which is larger than is")
            lines.append("          physically typical. Suspect scanner drift/creep, or")
            lines.append("          a fit straddling two overlapping moires (common in")
            lines.append("          trilayers). Check trace vs retrace: drift reverses,")
            lines.append("          real strain does not.")
        else:
            lines.append("   [ok]   heterostrain is in the physically plausible range.")

    lines.append("")
    lines.append("   This fit is exactly determined, so it always fits perfectly and")
    lines.append("   carries no error bar. Repeat over several scans and take the")
    lines.append("   mean and spread to get a real uncertainty.")
    return "\n".join(lines)


class App:
    def __init__(self, root):
        root.title("Moire twist / strain calculator")
        pad = {"padx": 6, "pady": 3}

        frm = ttk.Frame(root, padding=10)
        frm.grid(sticky="nsew")

        ttk.Label(frm, text="Material").grid(row=0, column=0, sticky="w", **pad)
        self.material = ttk.Combobox(frm, values=list(MATERIALS), width=12,
                                     state="readonly")
        self.material.set("WSe2")
        self.material.grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(frm, text="a (nm)").grid(row=0, column=2, sticky="e", **pad)
        self.alat = ttk.Entry(frm, width=10)
        self.alat.grid(row=0, column=3, sticky="w", **pad)
        self.material.bind("<<ComboboxSelected>>", self.sync_a)
        self.sync_a()

        self.mode = tk.StringVar(value="real")
        ttk.Radiobutton(frm, text="ACF / real space (nm)", value="real",
                        variable=self.mode).grid(row=1, column=0, columnspan=2,
                                                 sticky="w", **pad)
        ttk.Radiobutton(frm, text="FFT / reciprocal (1/nm)", value="recip",
                        variable=self.mode).grid(row=1, column=2, columnspan=2,
                                                 sticky="w", **pad)

        ttk.Separator(frm, orient="horizontal").grid(row=2, column=0, columnspan=4,
                                                     sticky="ew", pady=8)

        ttk.Label(frm, text="Measure Lattice vectors").grid(row=3, column=0,
                                                            columnspan=4, sticky="w", **pad)
        ttk.Label(frm, text="x").grid(row=4, column=1, **pad)
        ttk.Label(frm, text="y").grid(row=4, column=2, **pad)

        ttk.Label(frm, text="a₁").grid(row=5, column=0, sticky="e", **pad)
        self.a1x = ttk.Entry(frm, width=14); self.a1x.grid(row=5, column=1, **pad)
        self.a1y = ttk.Entry(frm, width=14); self.a1y.grid(row=5, column=2, **pad)

        ttk.Label(frm, text="a₂").grid(row=6, column=0, sticky="e", **pad)
        self.a2x = ttk.Entry(frm, width=14); self.a2x.grid(row=6, column=1, **pad)
        self.a2y = ttk.Entry(frm, width=14); self.a2y.grid(row=6, column=2, **pad)

        ttk.Button(frm, text="Compute", command=self.compute).grid(
            row=5, column=3, rowspan=2, sticky="nsew", **pad)

        self.out = tk.Text(frm, width=64, height=26, font=("Menlo", 11),
                           borderwidth=1, relief="solid")
        self.out.grid(row=7, column=0, columnspan=4, sticky="nsew", **pad)

        for e in (self.a1x, self.a1y, self.a2x, self.a2y, self.alat):
            e.bind("<Return>", lambda _e: self.compute())
        self.a1x.focus_set()

    def sync_a(self, _evt=None):
        self.alat.delete(0, tk.END)
        self.alat.insert(0, f"{MATERIALS[self.material.get()]:.4f}")

    def compute(self):
        self.out.delete("1.0", tk.END)
        try:
            vals = [float(e.get()) for e in (self.a1x, self.a1y, self.a2x, self.a2y)]
            a_lat = float(self.alat.get())
            if a_lat <= 0:
                raise ValueError("lattice constant must be positive")
        except ValueError as exc:
            self.out.insert("1.0", f"input error: {exc}")
            return
        try:
            res = analyze(vals[:2], vals[2:], a_lat, self.mode.get())
        except Exception as exc:
            self.out.insert("1.0", f"error: {exc}")
            return
        self.out.insert("1.0", format_report(res, a_lat, self.mode.get(),
                                             self.material.get()))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
