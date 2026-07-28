# Moiré analysis — twisted 2L MoS₂ and multilayer WSe₂

Twist-angle and heterostrain extraction from LFM scans, following the method in
Science `ade9995` (SI Figs. S6, S12–S14). Everything here was generated from the
raw `.ibw` acquisitions; the raw data itself is **not** in this folder and was
never modified.

## Layout

```
moire_analysis/
├── scripts/            analysis code (see below)
├── results/
│   ├── MoS2_2L/        figures + moire_grid_analysis.xlsx
│   └── WSe2/           figures
├── gwy/
│   ├── MoS2_2L/        63 preprocessed .gwy for Gwyddion
│   └── WSe2/            7 preprocessed .gwy
├── data/
│   └── grid_fits.json  cached per-scan fits the workbook is built from
└── reference/          copy of the paper SI
```

Raw data stays put and is read-only to these scripts:

| sample | raw location | scans |
|---|---|---|
| MoS₂ 2L | `~/Downloads/2L_MoS2` | 66 `.ibw` |
| WSe₂ 2L/3L/4L | `~/Downloads/071726_kl62tw4lwse2` | 7 `.ibw` |

`~/Downloads/2L_MoS2/LFM_2LMoS20017_FFT.gwy` is your own hand-processed Gwyddion
file. It stayed in Downloads deliberately — it is the reference the preprocessing
was validated against.

## Scripts

Paths live in `paths.py`; nothing hard-codes a Downloads path. All of them
default their output into `results/<sample>/`, overridable with `-o`.

| script | what it does |
|---|---|
| `moire_prep.py` | `.ibw` → calibrated `.gwy` with raw / leveled / denoised / ACF / FFT channels |
| `moire_calc.py` | GUI calculator: paste Measure Lattice vectors → twist + heterostrain |
| `moire_verify.py` | overlay figures — identified hexagon in FFT, reconstructed lattice in real space |
| `moire_map.py` | spatial θ(x,y) and ε(x,y) maps by geometric phase analysis (Figs. S12/S13) |
| `moire_two_hex.py` | fits several coexisting moiré hexagons, for multilayer stacks |
| `strainmap_figS13.py` | strain-map figure in the published Fig. S13 format (`--sample MoS2_2L` / `WSe2`) |
| `wse2_fft_figure.py` | the WSe₂ FFT + hexagon figure |
| `make_grid_workbook.py` | builds the Excel workbook from `data/grid_fits.json` |
| `add_all_images_sheet.py` | appends the per-image sheet to that workbook |

Run them from `scripts/`, e.g.

```bash
cd ~/moire_analysis/scripts
python3 moire_prep.py ~/Downloads/2L_MoS2          # regenerate all .gwy
python3 moire_map.py ~/Downloads/2L_MoS2/LFM_2LMoS20017.ibw -m MoS2
python3 moire_calc.py                              # the GUI calculator
```

**Workbook regeneration order matters.** `make_grid_workbook.py` writes a fresh
file; `add_all_images_sheet.py` appends to it. Run them in that order or the
per-image sheet is lost. The second one refits every scan and takes a few minutes.

## Results

**MoS₂ 2L**, 3×3 grid, medians over repeat scans at each position:

| | twist |
|---|---|
| across 9 positions | **1.074 ± 0.041°** (0.995–1.128°) |
| moiré period | ~16–18 nm |
| heterostrain | 0.5–0.85 % |

Twist decreases monotonically left→right across the grid (1.113 → 1.070 → 1.039°),
but the right-hand column rests on single scans, so that gradient is the weakest
claim in the set.

**WSe₂**, all 7 scans: **twist 4.58 ± 0.15°**, moiré period ~4 nm, strain ~1.6 %.
One resolvable hexagon per scan — see the caveat below.

## What is trustworthy, and what isn't

- **Twist is robust.** It depends mostly on the FFT peak radius, which is stable
  against imperfect hexagon assembly. Repeat scans at a fixed grid position agree
  to 0.04–0.13°.
- **Heterostrain is softer.** It depends on the *relative* lengths and angles of
  the three moiré vectors, which a mis-assembled hexagon corrupts. Treat MoS₂
  values as good to roughly ±0.2 %.
- **The decomposition has two roots.** A twist-dominated one and a
  strain-dominated one (≈0.2°/4 %) fit the same peaks. They are separated here by
  physical plausibility and by disagreement with repeat scans at the same
  position — not from first principles. Breaking the degeneracy properly needs
  crystallographic orientation from atomic resolution or flake edges.
- **Strain maps are positively biased.** Heterostrain is a magnitude, so phase
  noise can only push it up; the map mean runs above the global fit. The twist
  map averages correctly. Bright filaments mark the domain-wall network, but
  individual peak values along them are not quantitative.
- **The Fig. S13-format map** (`figS13_strain_mapping.png`) plots a *signed*
  quantity instead, ε = θ·(L(r) − ⟨L⟩)/⟨L⟩, so the diverging scale is meaningful.
  Median |ε| is 0.23 %, comparable to the paper's ±0.3 % range, but the
  distribution is heavy-tailed. The colour limit defaults to the 98th percentile
  of |ε| (`--pct`), which keeps the pale look of the published figure instead of
  saturating a fifth of the tiles. Tile opacity tracks the local Bragg amplitude
  (`--amp-pct`): where the moire signal is weak the phase gradient is
  unreliable, so those tiles fade rather than showing a confident-looking spike.
  Neither control alters the values — only how they are displayed.
- **WSe₂ strain magnitudes are not comparable to MoS₂ ones.** Median |ε| is
  0.77 % for WSe₂ vs 0.23 % for MoS₂, but ε = θ·(δL/L), and WSe₂'s twist (4.6°)
  is 4.3× MoS₂'s (1.07°). The same fractional wavelength noise therefore maps to
  ~4.3× larger ε — 0.23 × 4.3 ≈ 1.0 %, close to what is seen. Most of the
  difference is that scaling, not more strain. Compare ε only within a sample.
- **Tile density is display resolution, not information.** ~1015 tiles/panel at
  5.7 nm spacing, but the field's true resolution is FWHM ~16 nm, set by the
  Bragg mask (`--sigma-frac`, default 0.35). Tiles are therefore ~2.8x
  oversampled and neighbours are correlated. Subdividing costs nothing —
  4x the tiles left median |ε| unchanged at 0.153 → 0.154 % — but *widening the
  mask* to buy real resolution inflates strain ~36 % in noise and leaks the
  neighbouring Bragg peak, so it is not a free lunch. Read patterns, not tiles.
- **The WSe₂ multilayers.** A trilayer has two twisted interfaces, but the fitter
  finds only one hexagon per scan. That is consistent with L3 sitting near L1's
  orientation — the two ~4° moirés then nearly coincide, and the L1–L3 moiré moves
  to very long wavelength near DC. Consistent with, but not proof of, that
  stacking.

## Preprocessing, validated against Gwyddion

`moire_prep.py` reproduces your Gwyddion pipeline to **0.238 % rms, correlation
0.9999972**, checked against `LFM_2LMoS20017_FFT.gwy` and its processing log:

- Align Rows → **Median** (log `method=1`; mean/median-of-differences/trimmed-mean
  are 5.6–22 % off)
- Remove Polynomial Background → **tensor product, independent degree 3** (log
  `independent=1, col_degree=3, row_degree=3`; degree 2 is 6.5 % off). Idempotent,
  so applying it repeatedly is harmless.
- Non-local means denoising — the paper's denoiser, which **Gwyddion has no
  equivalent for**. This is the one step you cannot reproduce in the GUI.

One importer quirk worth knowing: **Gwyddion transposes and vertically flips
`.ibw` data** relative to raw `igor2` output. Matching that matters — with the
wrong orientation, median row alignment removes column offsets instead of
scan-line offsets and the line noise survives untouched.

## Using the .gwy files in Gwyddion

Open any file in `gwy/`, select the **FFT** channel, and place the two Measure
Lattice vectors **by hand** on adjacent moiré peaks before hitting Refine —
`Estimate` reliably locks onto the streak or noise instead. Do not fit the ACF
channel on this data: residual line noise forms a ridge that dominates its maxima.

For the figures' look: `magma` gradient (installed in `~/.gwyddion/gradients/`,
alongside `afmhot`, `viridis`, `inferno`) plus **Color range → Adaptive**.
Adaptive is display-only and non-destructive — it changes contrast, not data.
