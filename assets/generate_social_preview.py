"""Generate the original CAUSALOS social image from its genuine lab capture.

Requires Pillow. Run from any directory; writes social-preview.png and an
editable, self-contained social-preview.svg beside this source. No sampled
artwork or generated statistical plots are used. System fonts are not bundled.
"""
from pathlib import Path
import base64
import hashlib
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "lab-after-top.png"
SCALE = 2
PAPER, INK, BLUE = "#f5f4ee", "#222a31", "#2846c9"


def font(kind, size):
    paths = {
        "sans": ("/System/Library/Fonts/SFNS.ttf", 0),
        "serif": ("/System/Library/Fonts/Supplemental/Iowan Old Style.ttc", 0),
        "italic": ("/System/Library/Fonts/Supplemental/Iowan Old Style.ttc", 2),
        "mono": ("/System/Library/Fonts/Menlo.ttc", 0),
    }
    path, index = paths[kind]
    if Path(path).exists():
        return ImageFont.truetype(path, size * SCALE, index=index)
    family = "DejaVuSerif" if kind in ("serif", "italic") else "DejaVuSans"
    suffix = "-Italic" if kind == "italic" else ""
    return ImageFont.truetype(family + suffix + ".ttf", size * SCALE)


canvas = Image.new("RGB", (1280 * SCALE, 640 * SCALE), PAPER)
draw = ImageDraw.Draw(canvas)


def text(x, y, value, kind, size, color=INK):
    draw.text((x * SCALE, y * SCALE), value, font=font(kind, size), fill=color, anchor="lt")


def line(coords, color, width=1):
    draw.line([v * SCALE for v in coords], fill=color, width=width * SCALE)


# The laboratory's C-and-point identity, drawn directly from its vector geometry.
draw.arc((48*SCALE, 47*SCALE, 100*SCALE, 99*SCALE), 44, 316, fill=BLUE, width=5*SCALE)
draw.ellipse((88*SCALE, 69*SCALE, 98*SCALE, 79*SCALE), fill=BLUE)
text(116, 54, "CAUSALOS", "sans", 36)
line((48, 128, 1232, 128), "#d7dbcf")
text(48, 226, "Evidence", "serif", 64)
text(48, 301, "before", "serif", 64)
text(48, 376, "conclusions.", "italic", 64, BLUE)
text(48, 575, "CAUSAL RESEARCH LAB", "mono", 14, "#69746a")

# The image is a genuine application capture, uniformly scaled, with no data edits.
x, y, width, height = 476, 158, 756, 425
shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
sd = ImageDraw.Draw(shadow)
sd.rounded_rectangle((x*SCALE, (y+12)*SCALE, (x+width)*SCALE, (y+height+12)*SCALE),
                     radius=10*SCALE, fill=(34, 42, 49, 25))
shadow = shadow.filter(ImageFilter.GaussianBlur(14*SCALE))
canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow)
shot = Image.open(SOURCE).convert("RGB").resize((width*SCALE, height*SCALE), Image.Resampling.LANCZOS)
mask = Image.new("L", shot.size)
ImageDraw.Draw(mask).rounded_rectangle((0, 0, shot.width-1, shot.height-1), radius=8*SCALE, fill=255)
canvas.paste(shot, (x*SCALE, y*SCALE), mask)
draw = ImageDraw.Draw(canvas)
draw.rounded_rectangle((x*SCALE, y*SCALE, (x+width)*SCALE, (y+height)*SCALE),
                       radius=8*SCALE, outline="#cdd2c6", width=SCALE)
canvas.convert("RGB").resize((1280,640), Image.Resampling.LANCZOS).save(ROOT/"social-preview.png", optimize=True)

encoded = base64.b64encode(SOURCE.read_bytes()).decode()
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640" width="1280" height="640" role="img" aria-labelledby="title desc">
<title id="title">CAUSALOS · Evidence before conclusions.</title>
<desc id="desc">The CAUSALOS identity beside a genuine capture of its simulated causal experiment laboratory.</desc>
<defs><clipPath id="frame"><rect x="476" y="158" width="756" height="425" rx="8"/></clipPath><filter id="shadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="12" stdDeviation="14" flood-color="#222a31" flood-opacity=".1"/></filter></defs>
<rect width="1280" height="640" fill="{PAPER}"/>
<path d="M94 54A26 26 0 1 0 94 92" fill="none" stroke="{BLUE}" stroke-width="5"/><circle cx="93" cy="74" r="5" fill="{BLUE}"/>
<text x="116" y="86" fill="{INK}" font-family="system-ui,sans-serif" font-size="36">CAUSALOS</text><path d="M48 128H1232" stroke="#d7dbcf"/>
<g font-family="Iowan Old Style,Palatino,Georgia,serif" font-size="64" fill="{INK}"><text x="48" y="278">Evidence</text><text x="48" y="353">before</text><text x="48" y="428" font-style="italic" fill="{BLUE}">conclusions.</text></g>
<text x="48" y="590" fill="#69746a" font-family="Menlo,monospace" font-size="14">CAUSAL RESEARCH LAB</text>
<rect x="476" y="158" width="756" height="425" rx="8" fill="#fcfbf7" filter="url(#shadow)"/>
<image x="476" y="158" width="756" height="425" preserveAspectRatio="none" clip-path="url(#frame)" href="data:image/png;base64,{encoded}"/>
<rect x="476" y="158" width="756" height="425" rx="8" fill="none" stroke="#cdd2c6"/>
</svg>'''
(ROOT/"social-preview.svg").write_text(svg)
(ROOT/"social-preview-provenance.json").write_text(json.dumps({
    "source": "assets/lab-after-top.png", "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "size": [1280,640], "label": "genuine application screenshot, simulated laboratory",
    "transformations": "uniform resize, rounded frame and shadow; no data or result changes",
    "output_sha256": hashlib.sha256((ROOT/"social-preview.png").read_bytes()).hexdigest(),
}, indent=2))
print(ROOT/"social-preview.png")
