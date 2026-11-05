#!/usr/bin/env python3
"""Fill slide_template.html with slide_data.json -> <out>/slide.html (a self-contained 1280x720 slide page)."""
import os, sys
out = sys.argv[1] if len(sys.argv) > 1 else "/root/capsule/results/residual_merge_detection"
here = os.path.dirname(os.path.abspath(__file__))
html = open(f"{here}/slide_template.html").read().replace("__DATA__", open(f"{out}/slide_data.json").read())
open(f"{out}/slide.html", "w").write(html); print("wrote", f"{out}/slide.html")
