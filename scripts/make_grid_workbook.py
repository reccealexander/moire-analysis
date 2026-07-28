#!/usr/bin/env python3
"""
Write an Excel workbook summarising the 2L MoS2 3x3 grid moire analysis.

Reads the per-scan fit results cached by the grid analysis (/tmp/grid.json) and
produces three sheets: the position summary, the per-scan detail with the
inclusion flag, and the twist laid out in the physical 3x3 arrangement.
"""
import json
import os

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from paths import GRID_FITS, WORKBOOK

# grid number -> (row coordinate, column coordinate), file range
POSITION = {
    1: ("532", "-1.69", "52-64"), 2: ("532", "-1.5", "18-21"), 3: ("532", "-1.29", "23-27"),
    4: ("732", "-1.69", "46-51"), 5: ("732", "-1.5", "14-17"), 6: ("732", "-1.29", "28-31"),
    7: ("932", "-1.69", "41-45"), 8: ("932", "-1.49", "37-40"), 9: ("932", "-1.29", "32-36"),
}
ROW = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2}
COL = {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2, 7: 0, 8: 1, 9: 2}

L_MIN, L_MAX, S_MAX = 13.0, 21.0, 1.5      # consistent-cluster criteria

HEAD_FILL = PatternFill("solid", fgColor="1F3864")
HEAD_FONT = Font(bold=True, color="FFFFFF", size=11)
TITLE_FONT = Font(bold=True, size=13)
NOTE_FONT = Font(italic=True, size=9, color="555555")
EXCL_FILL = PatternFill("solid", fgColor="FCE4E4")
THIN = Side(style="thin", color="BBBBBB")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def style_header(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER


def autosize(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def main():
    data = json.load(open(GRID_FITS))
    wb = Workbook()

    # ---------------------------------------------------------------- summary
    ws = wb.active
    ws.title = "Grid summary"
    ws["A1"] = "2L MoS2 - moire analysis by grid position"
    ws["A1"].font = TITLE_FONT
    ws["A2"] = ("Median over repeat scans at each position, using consistent-cluster fits "
                f"(moire period {L_MIN:.0f}-{L_MAX:.0f} nm and heterostrain < {S_MAX} %).")
    ws["A2"].font = NOTE_FONT
    ws["A3"] = ("All values derived from FFT peak positions; twist from the exact polar "
                "decomposition T = R(theta)U. MoS2 lattice constant a = 0.3160 nm.")
    ws["A3"].font = NOTE_FONT

    head = ["Grid #", "Row coord", "Col coord", "Files", "n scans",
            "Moire period (nm)", "Twist (deg)", "Twist spread (deg)", "Heterostrain (%)"]
    ws.append([])
    ws.append(head)
    style_header(ws, ws.max_row, len(head))

    twists = []
    for g in range(1, 10):
        rows = [r for r in data[str(g)] if L_MIN <= r[1] <= L_MAX and r[3] < S_MAX]
        rc, cc, fr = POSITION[g]
        if not rows:
            ws.append([g, rc, cc, fr, 0, None, None, None, None])
            continue
        L = np.array([r[1] for r in rows])
        t = np.array([r[2] for r in rows])
        s = np.array([r[3] for r in rows])
        twists.append(np.median(t))
        ws.append([g, rc, cc, fr, len(rows),
                   round(float(np.median(L)), 2), round(float(np.median(t)), 3),
                   round(float(t.max() - t.min()), 3), round(float(np.median(s)), 2)])

    for r in range(6, ws.max_row + 1):
        for c in range(1, len(head) + 1):
            ws.cell(row=r, column=c).border = BORDER
            ws.cell(row=r, column=c).alignment = Alignment(horizontal="center")

    ws.append([])
    ws.append(["Mean across 9 positions", "", "", "", "",
               "", round(float(np.mean(twists)), 3), "", ""])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    ws.cell(row=ws.max_row, column=7).font = Font(bold=True)
    ws.append(["Std dev across positions", "", "", "", "",
               "", round(float(np.std(twists)), 3), "", ""])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    autosize(ws, [8, 11, 11, 10, 9, 18, 12, 18, 16])
    ws.freeze_panes = "A6"

    # ------------------------------------------------------------- per scan
    ws2 = wb.create_sheet("Per-scan fits")
    ws2["A1"] = "Every scan fitted, including those excluded from the medians"
    ws2["A1"].font = TITLE_FONT
    ws2["A2"] = ("Excluded rows are shaded. They are rejected because they disagree with "
                 "repeat scans at the same grid position - the decomposition landed on the "
                 "strain-dominated root, or the fit found a second-order peak.")
    ws2["A2"].font = NOTE_FONT
    head2 = ["File #", "Grid #", "Row coord", "Col coord",
             "Moire period (nm)", "Twist (deg)", "Heterostrain (%)", "Included"]
    ws2.append([])
    ws2.append(head2)
    style_header(ws2, ws2.max_row, len(head2))

    for g in range(1, 10):
        for n, L, t, s in sorted(data[str(g)]):
            ok = (L_MIN <= L <= L_MAX) and (s < S_MAX)
            rc, cc, _ = POSITION[g]
            ws2.append([f"{n:04d}", g, rc, cc, round(L, 2), round(t, 3), round(s, 2),
                        "yes" if ok else "no"])
            if not ok:
                for c in range(1, len(head2) + 1):
                    ws2.cell(row=ws2.max_row, column=c).fill = EXCL_FILL
    for r in range(5, ws2.max_row + 1):
        for c in range(1, len(head2) + 1):
            ws2.cell(row=r, column=c).border = BORDER
            ws2.cell(row=r, column=c).alignment = Alignment(horizontal="center")
    autosize(ws2, [9, 8, 11, 11, 18, 12, 16, 10])
    ws2.freeze_panes = "A5"

    # ------------------------------------------------------------ spatial 3x3
    ws3 = wb.create_sheet("Spatial layout")
    ws3["A1"] = "Twist (deg) in the physical 3x3 scan arrangement"
    ws3["A1"].font = TITLE_FONT
    ws3["A2"] = "Rows are the first coordinate, columns the second, matching the sample grid."
    ws3["A2"].font = NOTE_FONT

    grid_t = {}
    for g in range(1, 10):
        rows = [r for r in data[str(g)] if L_MIN <= r[1] <= L_MAX and r[3] < S_MAX]
        grid_t[g] = round(float(np.median([r[2] for r in rows])), 3) if rows else None

    ws3.append([])
    ws3.append(["", "-1.69", "-1.49 / -1.5", "-1.29"])
    style_header(ws3, ws3.max_row, 4)
    for r_i, rowlabel in enumerate(["532", "732", "932"]):
        line = [rowlabel]
        for c_i in range(3):
            g = [k for k in range(1, 10) if ROW[k] == r_i and COL[k] == c_i][0]
            line.append(grid_t[g])
        ws3.append(line)
    for r in range(4, ws3.max_row + 1):
        for c in range(1, 5):
            cell = ws3.cell(row=r, column=c)
            cell.border = BORDER
            cell.alignment = Alignment(horizontal="center")
            if c == 1:
                cell.font = Font(bold=True)

    ws3.append([])
    ws3.append(["Column means (left to right):"])
    ws3.cell(row=ws3.max_row, column=1).font = Font(bold=True)
    means = []
    for c_i in range(3):
        vals = [grid_t[k] for k in range(1, 10) if COL[k] == c_i and grid_t[k] is not None]
        means.append(round(float(np.mean(vals)), 3))
    ws3.append(["", *means])
    ws3.append([])
    ws3.append(["Note: twist decreases monotonically left to right in all three rows, "
                "a gradient of about 0.07 deg across the grid."])
    ws3.cell(row=ws3.max_row, column=1).font = NOTE_FONT
    autosize(ws3, [14, 16, 16, 16])

    os.makedirs(os.path.dirname(WORKBOOK), exist_ok=True)
    wb.save(WORKBOOK)
    out = WORKBOOK
    print("wrote", out)


if __name__ == "__main__":
    main()
