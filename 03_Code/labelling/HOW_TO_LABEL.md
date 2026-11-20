# How to label wires on a new photo (about 30–45 minutes)

**Why:** every score so far comes from two labelled photos from the 30-minute growth
run. Nikhil's pipeline measures 50,000× photos from the 60-minute run. One labelled
quarter of one of those photos tells us whether the method ranking holds where it
matters.

**What you'll make:** one straight line per wire, about 80 lines on a quarter of
`SF251213M_50000×_37`. The tool and the scoring script are already set up.

## 1. Open the tool

Open a terminal (PowerShell) and run these two commands:

```
cd C:\Users\Acer\Downloads\Materials_ThangPham\Materials_ThangPham\03_Code\labelling
..\..\..\labelling_env\Scripts\python label_wires.py
```

A napari window opens with the photo and a yellow **wires** layer, already in
line mode. The first start can take about 20 seconds.

## 2. Draw one line per wire

- **Zoom** with the mouse wheel. **Pan** by holding the space bar and dragging.
- **Draw a wire:** click at one end of the wire, then click at the other end.
  (In some napari versions you press, drag to the other end and release; both give a line.)
- **Undo a mistake:** press `Ctrl+Z`. Or click the arrow tool (Select shapes) in the
  top-left panel, click the bad line, press `Delete`, then click the line tool again.
- **Save** at any time with `Ctrl+S`. Closing the window also saves.
- **Stopping halfway is fine.** Run the same command again and your lines come back.

## 3. Rules (keep them the same for every wire)

1. **One line per straight wire**, from where it visibly starts to where it visibly ends.
2. **Crossings:** if two wires cross, draw **each one straight through** the crossing.
   Do not stop at the crossing.
3. **Corners:** where a wire changes direction (a V or a zig-zag), that is **two wires**.
   Draw one line per straight piece, meeting at the corner.
4. **Hidden part:** if a wire runs under another and you can see it continue on the
   other side in a straight line, draw it as one line through.
5. **Photo edge:** draw wires that leave the photo up to the edge.
6. **Bright rods and dark wires are both wires.** Label both.
7. **Don't label** round dots or blobs, or bright compact particles (hexagons or diamonds).
8. **Very faint lines:** label them only if you're sure they're wires. Consistency
   matters more than catching every one.
9. Click on the **middle of the wire's width**, not on one edge.

Write down any wire you weren't sure about (rough position and why). Those notes
are good questions for Dr. Chen.

## 4. Score every method on your labels

When you're done, run these two commands in the terminal:

```
cd C:\Users\Acer\Downloads\Materials_ThangPham\Ridge-Filtering-Nanowires-main
venv\Scripts\python compare\external.py
```

It takes about 5 minutes. It prints a table of every method scored on your photo and saves it
as `results\compare\external_SF251213M_50000x_37_tl.csv`. The pictures in
`results\compare\external_overlays\` show each method against your labels:
green = correct, orange = false alarm, blue = missed.

## Optional: label more

Other quarters of the same photo: add `--quarter tr`, `--quarter bl` or `--quarter br`.
Another photo: add `--photo SF251213M_50000x_50`. Each gets its own labels file.
`external.py` scores the most recently saved one, or the file you name:
`venv\Scripts\python compare\external.py --labels ..\Materials_ThangPham\03_Code\labelling\labels_....csv`
