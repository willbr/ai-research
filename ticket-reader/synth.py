"""
Synthetic UK National Rail (orange) ticket generator.

Produces realistic "customer photos": a flat ticket rendered with random field
values, then warped through a random homography onto a cluttered background.
Because we choose the homography, the 4 ticket-corner coordinates fall out for
free -> labels for the corner-detection net with zero hand-annotation.

Two layers, kept deliberately separate:
  render_flat_ticket(...) -> (flat RGBA image, fields dict, field boxes)   # canonical space
  warp_onto_background(...) -> (photo RGB, corners [4x2])                  # camera space

Run directly to dump a few inspectable samples (corners drawn) into ./out/.
"""
import random, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# --- canonical flat-ticket size (px). Real card is 86x54mm; keep that aspect. ---
TW, TH = 540, 340

MONO = "/System/Library/Fonts/Menlo.ttc"
SANS = "/System/Library/Fonts/Helvetica.ttc"

# field value pools -----------------------------------------------------------
STATIONS = ["LONDON WATERLOO", "MANCHESTER PICC", "BIRMINGHAM NEW ST", "LEEDS",
            "EDINBURGH", "GLASGOW CENTRAL", "BRISTOL TEMPLE M", "READING",
            "YORK", "LIVERPOOL LIME ST", "OXFORD", "CAMBRIDGE", "BRIGHTON"]
TYPES = ["ANYTIME DAY SINGLE", "ANYTIME DAY RETURN", "OFF-PEAK DAY RETURN",
         "SUPER OFF-PEAK RTN", "ADVANCE SINGLE", "ANYTIME RETURN"]
ROUTES = ["ANY PERMITTED", "NOT VIA LONDON", "VIA BIRMINGHAM", "HIGH SPEED 1"]
CLASSES = ["STANDARD CLASS", "FIRST CLASS"]


def _font(path, size):
  try: return ImageFont.truetype(path, size)
  except Exception: return ImageFont.load_default()


def random_fields():
  o, d = random.sample(STATIONS, 2)
  return {
    "class":  random.choice(CLASSES),
    "origin": o,
    "dest":   d,
    "type":   random.choice(TYPES),
    "route":  random.choice(ROUTES),
    "price":  f"£{random.randint(3, 240)}.{random.choice(['00','50','40','20','80','10'])}",
    "date":   f"{random.randint(1,28):02d} {random.choice(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])} {random.randint(24,26)}",
    "adults": f"{random.randint(1,4)} ADULT",
    "number": "".join(random.choice("0123456789") for _ in range(11)),
  }


# Fixed layout regions (canonical px) where each field lives. Used for BOTH
# recognizer training crops and end-to-end inference, so the two always match.
# Text is left-aligned at the field origin; regions are sized for the longest value.
REGIONS = {
  "class":  (16,  14, 264,  46),
  "origin": (16,  46, 374,  90),
  "dest":   (60,  84, 374, 128),
  "type":   (16, 134, 380, 168),
  "route":  (16, 166, 380, 198),
  "price":  (380, 46, 532,  94),
  "date":   (16, 204, 300, 242),
  "adults": (380,134, 532, 168),
  "number": (16, 286, 430, 332),
}

def render_flat_ticket(fields):
  """Render the canonical front-on ticket. Returns (RGBA image, boxes dict).
  boxes are (x0,y0,x1,y1) in canonical px -> used later for OCR field crops."""
  img = Image.new("RGBA", (TW, TH), (0, 0, 0, 0))
  d = ImageDraw.Draw(img)

  # body: orange card with a cream serial band along the bottom, rounded corners
  d.rounded_rectangle([0, 0, TW-1, TH-1], radius=18, fill=(228, 124, 64, 255))
  d.rectangle([0, TH-58, TW-1, TH-1], fill=(244, 238, 218, 255))
  # subtle inner border like the real guilloche frame
  d.rounded_rectangle([10, 10, TW-11, TH-68], radius=10, outline=(150, 70, 30, 255), width=2)

  boxes = {}
  def line(key, text, x, y, size, font_path=MONO, fill=(20, 12, 8, 255)):
    f = _font(font_path, size)
    d.text((x, y), text, font=f, fill=fill)
    bb = d.textbbox((x, y), text, font=f)
    boxes[key] = (bb, text)   # (x0,y0,x1,y1), drawn string

  line("class",  fields["class"],  24, 20, 20, MONO)
  line("origin", fields["origin"], 24, 52, 30)
  line("arrow",  ">>", 24, 92, 22, SANS, fill=(150,70,30,255))
  line("dest",   fields["dest"],   70, 90, 30)
  line("type",   fields["type"],   24, 140, 22)
  line("route",  "ROUTE " + fields["route"], 24, 172, 18)
  line("price",  fields["price"],  TW-150, 52, 34, MONO)
  line("date",   fields["date"],   24, 210, 22)
  line("adults", fields["adults"], TW-150, 140, 20, MONO)
  line("number", fields["number"], 24, TH-46, 26)
  return img, boxes


# --- homography / warp -------------------------------------------------------
def find_coeffs(dst, src):
  """PIL perspective coeffs mapping OUTPUT coords (dst) back to INPUT (src)."""
  A = []
  for (xo, yo), (xi, yi) in zip(dst, src):
    A.append([xo, yo, 1, 0, 0, 0, -xi*xo, -xi*yo])
    A.append([0, 0, 0, xo, yo, 1, -yi*xo, -yi*yo])
  A = np.array(A, dtype=np.float64)
  b = np.array(src, dtype=np.float64).reshape(8)
  return np.linalg.solve(A, b)


def _background(W, H):
  # procedural clutter: gradient base + colour blobs + noise (a desk/table-ish scene)
  base = np.zeros((H, W, 3), np.float32)
  c1, c2 = np.random.randint(40, 200, 3), np.random.randint(40, 200, 3)
  for y in range(H): base[y] = c1 + (c2 - c1) * (y / H)
  bg = Image.fromarray(base.astype(np.uint8))
  dd = ImageDraw.Draw(bg)
  for _ in range(random.randint(2, 6)):
    x0, y0 = random.randint(0, W), random.randint(0, H)
    dd.rectangle([x0, y0, x0+random.randint(20, 200), y0+random.randint(20, 200)],
                 fill=tuple(np.random.randint(30, 220, 3)))
  bg = bg.filter(ImageFilter.GaussianBlur(random.uniform(1, 4)))
  return bg


def warp_onto_background(flat, W=640, H=480):
  """Warp the flat ticket by a random homography onto clutter. Returns
  (RGB photo, corners) where corners are the ticket's 4 corners in the photo,
  ordered TL,TR,BR,BL."""
  src = [(0, 0), (TW, 0), (TW, TH), (0, TH)]
  # base placement: scaled rectangle centred with margin, then jitter each corner
  scale = random.uniform(0.45, 0.85)
  bw, bh = TW * scale, TH * scale
  cx = random.uniform(bw/2 + 20, W - bw/2 - 20)
  cy = random.uniform(bh/2 + 20, H - bh/2 - 20)
  rect = np.array([[-bw/2, -bh/2], [bw/2, -bh/2], [bw/2, bh/2], [-bw/2, bh/2]])
  ang = random.uniform(-math.pi, math.pi)          # full rotation: customers hold it any way
  ca, sa = math.cos(ang), math.sin(ang)
  rot = rect @ np.array([[ca, sa], [-sa, ca]])
  jit = bw * random.uniform(0.04, 0.18)            # perspective tilt
  dst = [(cx + x + random.uniform(-jit, jit), cy + y + random.uniform(-jit, jit)) for x, y in rot]

  coeffs = find_coeffs(dst, src)
  warped = flat.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=(0, 0, 0, 0))

  photo = _background(W, H).convert("RGBA")
  photo.alpha_composite(warped)
  photo = photo.convert("RGB")

  # camera-ish degradation
  photo = ImageEnhance.Brightness(photo).enhance(random.uniform(0.7, 1.25))
  photo = ImageEnhance.Contrast(photo).enhance(random.uniform(0.8, 1.2))
  if random.random() < 0.6: photo = photo.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.4)))
  arr = np.asarray(photo).astype(np.float32)
  arr += np.random.normal(0, random.uniform(2, 10), arr.shape)
  photo = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
  return photo, np.array(dst, dtype=np.float32)


def sample(W=640, H=480):
  fields = random_fields()
  flat, boxes = render_flat_ticket(fields)
  photo, corners = warp_onto_background(flat, W, H)
  return photo, corners, fields, boxes


if __name__ == "__main__":
  import os
  os.makedirs("out", exist_ok=True)
  # one clean flat reference
  flat, _ = render_flat_ticket(random_fields())
  Image.alpha_composite(Image.new("RGBA", (TW, TH), (255,255,255,255)), flat).convert("RGB").save("out/flat.png")
  # a grid of warped "customer photos" with corners drawn
  for i in range(8):
    photo, corners, fields, _ = sample()
    dr = ImageDraw.Draw(photo)
    pts = [tuple(p) for p in corners]
    dr.line(pts + [pts[0]], fill=(0, 255, 0), width=3)
    for p in pts: dr.ellipse([p[0]-5, p[1]-5, p[0]+5, p[1]+5], fill=(255, 0, 0))
    photo.save(f"out/sample_{i}.png")
    if i == 0: print("fields:", fields)
  print("wrote out/flat.png + out/sample_0..7.png")
