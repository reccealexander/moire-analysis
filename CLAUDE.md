# moire_analysis

Twist angle and heterostrain from AFM/LFM scans of twisted 2D materials (2L MoS₂,
multilayer WSe₂), following Science `ade9995` SI Figs. S6, S12–S14.

`README.md` is the full write-up — read it for derivations, caveats and numbers.
This file is the operational quick-reference.

## Hard rules

- **Raw `.ibw` data lives outside this repo** — `~/Downloads/2L_MoS2` (66 scans) and
  `~/Downloads/071726_kl62tw4lwse2` (7 scans). Read-only. Never modify, never copy
  into the repo (`*.ibw` is gitignored as a guard).
- `~/Downloads/2L_MoS2/LFM_2LMoS20017_FFT.gwy` is the hand-processed Gwyddion
  reference the pipeline is validated against. Leave it where it is.
- Do not edit `results/`, `gwy/`, or `data/` by hand — all regenerable.

## Layout

| path | contents |
|---|---|
| `scripts/` | all analysis code; run from inside this dir |
| `results/MoS2_2L/`, `results/WSe2/` | figures + `moire_grid_analysis.xlsx` |
| `gwy/` | 70 preprocessed `.gwy` (~512 MB, gitignored, regenerable) |
| `data/grid_fits.json` | cached per-scan fits the workbook is built from |
| `reference/` | paper SI copy (~28 MB, gitignored) |

Deps: `numpy scipy scikit-image matplotlib igor2 gwyfile openpyxl` + `tkinter`.

## Scripts

Locations resolve through `paths.py` (`RAW`, `RESULTS`, `GWY`, `results_for`,
`sample_of`); nothing should hard-code a Downloads path.

| script | purpose |
|---|---|
| `moire_prep.py` | `.ibw` → calibrated `.gwy` with raw / leveled / denoised / ACF / FFT channels |
| `moire_calc.py` | Tk GUI: paste Measure Lattice vectors → twist + heterostrain. Owns `MATERIALS` lattice constants (MoS₂ 0.3160 nm, WSe₂ 0.3282 nm) and the `analyze()` decomposition |
| `moire_verify.py` | overlay figures: identified hexagon in FFT, reconstructed lattice in real space |
| `moire_map.py` | spatial θ(x,y), ε(x,y) maps via geometric phase analysis (Figs. S12/S13) |
| `moire_two_hex.py` | fits several coexisting moiré hexagons, for multilayer stacks |
| `strainmap_figS13.py` | strain map in published Fig. S13 format (signed ε, hex tiles) |
| `moire_fit.py` | **shared hexagon finder** — all fitting goes through here |
| `make_grid_workbook.py` | builds the Excel workbook from `data/grid_fits.json` |
| `add_all_images_sheet.py` | appends the per-image sheet to that workbook |

**Interface:** every analysis script takes a path (file *or* folder) plus
`-m MATERIAL`. The reciprocal band is derived from scan geometry, so a new sample
needs no config — only its lattice constant. `--preset MoS2_2L` / `--preset WSe2`
reproduce the published figures (curated layout + pinned band).

**All hexagon fitting must go through `moire_fit`.** Two near-duplicate finders
previously drifted apart and disagreed badly on MoS₂. The rule that works for
both samples is **regularity ratio 1.5 + 5 % strain ceiling** — neither knob alone
suffices: dropping the filter breaks WSe₂, tightening the ratio to 1.25 breaks
MoS₂. Do not re-implement this locally.

**Workbook order matters:** `make_grid_workbook.py` writes a *fresh* file,
`add_all_images_sheet.py` *appends*. Reversed, the per-image sheet is silently
lost. The append step refits every scan (minutes).

## Traps that have already cost time

- **`.ibw` orientation.** Gwyddion's importer transposes *and* vertically flips
  relative to raw `igor2` output — `moire_prep.load_ibw` returns
  `np.flipud(arr[:, :, idx].T)`. Get this wrong and median row alignment strips
  column offsets instead of scan-line offsets: line noise survives untouched, with
  no error. This previously produced the wrong conclusion that the MoS₂ scans had
  no moiré.
- **Fit the FFT channel, not the ACF.** Residual line noise forms a ridge that
  dominates the ACF's maxima.
- **In Gwyddion, place Measure Lattice vectors by hand**, then Refine. `Estimate`
  reliably locks onto the streak.
- **The decomposition has two roots** fitting the same peaks: twist-dominated, and
  strain-dominated (≈0.2° / 4 %). They are separated by physical plausibility and
  by agreement between repeat scans at one position — *not* from first principles.
  Breaking it properly needs crystallographic orientation (atomic resolution or
  flake edges).
- **Regenerating figures churns git.** matplotlib stamps a timestamp into PNG
  metadata, so identical plots are byte-different files.

## Preprocessing (matches Gwyddion to 0.238 % rms, corr. 0.9999972)

1. Align Rows → **Median** (other methods are 5.6–22 % off).
2. Remove Polynomial Background → **tensor product, independent degree 3**
   (`independent=1, col_degree=3, row_degree=3`) — *not* total-degree. Idempotent.
3. Non-local-means denoise — the paper's step; **Gwyddion has no equivalent**, so
   this one cannot be reproduced in the GUI.

Figure look: `magma` gradient (in `~/.gwyddion/gradients/`) + Color range →
Adaptive (display-only, non-destructive).

## Results and how far to trust them

| | MoS₂ 2L (3×3 grid) | WSe₂ (7 scans) |
|---|---|---|
| twist | **1.074 ± 0.041°** | **4.58 ± 0.15°** |
| moiré period | ~16–18 nm | ~4 nm |
| heterostrain | 0.5–0.85 % | ~1.6 % |

- **Twist is robust** — driven by FFT peak radius, stable against imperfect
  hexagon assembly.
- **Heterostrain is soft** (±0.2 % for MoS₂) — depends on relative lengths/angles
  of the three moiré vectors, which a mis-assembled hexagon corrupts.
- **Strain magnitudes do not compare across samples.** ε = θ·(δL/L), so values
  scale with θ; WSe₂'s 4.3× larger twist inflates ε ~4.3× for the same fractional
  noise. Compare ε only within a sample.
- Strain *maps* (magnitude) are positively biased; the signed Fig. S13 quantity is
  the meaningful one. Tile density is display resolution, not information.
- The left→right twist gradient across the MoS₂ grid is the weakest claim (right
  column rests on single scans).

## Git

Local-only, no remote. Branch `master`. `gwy/` and `reference/` are gitignored and
regenerable — do not commit them.
