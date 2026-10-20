"""Render SVG diagrams to PNG beside them (resvg-py; see make_discovery_diagram.py).

Usage: PYTHONPATH=<dir with resvg_py> python scripts/diagrams/render_svgs.py FILE.svg ...
"""
import os, sys
import matplotlib, resvg_py
fontdir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
svgs = sys.argv[1:]
if not svgs:
    sys.exit("usage: render_svgs.py FILE.svg [FILE.svg ...]  (the repo keeps PNGs; SVG "
             "sources of the older diagrams are in git history)")
for svg in svgs:
    data = resvg_py.svg_to_bytes(svg_path=svg, font_dirs=["/usr/share/fonts", fontdir],
                                 sans_serif_family="DejaVu Sans", font_family="DejaVu Sans")
    open(svg[:-4] + ".png", "wb").write(bytes(data))
    print(svg[:-4] + ".png", f"{len(data)/1e6:.2f} MB")
