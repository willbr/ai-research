"""
Build the field-recognizer dataset: render flat tickets, crop each field by its
known box, augment to mimic an imperfect rectification (small rotation/blur/
noise/jitter), store (crop, label) pairs.

  uv run python make_ocr_data.py [n_tickets_train] [n_tickets_val]
"""
import os, sys, time, random
import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageFilter, ImageEnhance
import synth, ocr

def _init():
  seed = int.from_bytes(os.urandom(4), "little")
  random.seed(seed); np.random.seed(seed)

def _crop_region(img, region):
  # fixed layout region + small jitter (mimics imperfect rectification alignment)
  x0, y0, x1, y1 = region
  jx, jy = random.randint(-4, 4), random.randint(-3, 3)
  return img.crop((x0+jx, y0+jy, x1+jx, y1+jy))

def _augment(crop):
  # applied to the native-size crop, BEFORE height-normalisation
  if random.random() < 0.7:
    crop = crop.rotate(random.uniform(-3, 3), resample=Image.BILINEAR, fillcolor=ocr.BG)
  crop = ImageEnhance.Brightness(crop).enhance(random.uniform(0.75, 1.2))
  crop = ImageEnhance.Contrast(crop).enhance(random.uniform(0.8, 1.2))
  if random.random() < 0.5:
    crop = crop.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
  return crop

def _one(_):
  fields = synth.random_fields()
  flat, boxes = synth.render_flat_ticket(fields)
  rgb = Image.alpha_composite(Image.new("RGBA", (synth.TW, synth.TH), (228,124,64,255)), flat).convert("RGB")
  out = []
  for key, region in synth.REGIONS.items():
    text = boxes[key][1]
    a = ocr.to_input(_augment(_crop_region(rgb, region))).astype(np.float32)  # (3,32,256)
    a += np.random.normal(0, random.uniform(2, 9), a.shape)
    out.append((np.clip(a, 0, 255).astype(np.uint8), ocr.encode(text)))
  return out

def build(n_tickets, path):
  t = time.perf_counter()
  with Pool(initializer=_init) as p:
    res = p.map(_one, range(n_tickets), chunksize=16)
  flat = [c for tk in res for c in tk]
  imgs = np.stack([c[0] for c in flat]).astype(np.uint8)
  labs = np.stack([c[1] for c in flat]).astype(np.uint8)
  np.savez(path, images=imgs, labels=labs)
  dt = time.perf_counter() - t
  print(f"{path}: {len(flat)} crops from {n_tickets} tickets in {dt:.1f}s  images{imgs.shape} labels{labs.shape}")

if __name__ == "__main__":
  os.makedirs("data", exist_ok=True)
  n_tr = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
  n_va = int(sys.argv[2]) if len(sys.argv) > 2 else 500
  build(n_tr, "data/ocr_train.npz")
  build(n_va, "data/ocr_val.npz")
