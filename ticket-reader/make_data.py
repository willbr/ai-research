"""
Build a cached corner-detection dataset, in parallel across CPU cores.

synth.sample() is CPU-bound (PIL), so we fan it out with a process Pool and
cache the result as .npz. Images are stored downscaled + uint8 to keep the file
(and later GPU memory) small; corners are normalised to [0,1] so they're
resolution-independent.

  uv run python make_data.py [n_train] [n_val]
"""
import os, sys, time
import numpy as np
from multiprocessing import Pool
from PIL import Image
import synth

GEN_W, GEN_H = 640, 480   # photo size we render at
N_IN = 128                # net input size (corners are coarse; 128 is plenty)

def _init():
  # each worker needs its own RNG state or they all generate identical tickets
  seed = int.from_bytes(os.urandom(4), "little")
  import random; random.seed(seed); np.random.seed(seed)

def _one(_):
  photo, corners, _fields, _boxes = synth.sample(GEN_W, GEN_H)
  img = np.asarray(photo.resize((N_IN, N_IN), Image.BILINEAR), dtype=np.uint8)  # (H,W,3)
  c = corners.astype(np.float32).copy()
  c[:, 0] /= GEN_W; c[:, 1] /= GEN_H   # normalise to [0,1]
  return img.transpose(2, 0, 1), c.reshape(8)   # (3,H,W), (8,)

def build(n, path):
  t = time.perf_counter()
  with Pool(initializer=_init) as p:
    res = p.map(_one, range(n), chunksize=16)
  imgs = np.stack([r[0] for r in res]).astype(np.uint8)
  cs = np.stack([r[1] for r in res]).astype(np.float32)
  np.savez(path, images=imgs, corners=cs)
  dt = time.perf_counter() - t
  print(f"{path}: {n} samples in {dt:.1f}s ({n/dt:.0f}/s)  images{imgs.shape} corners{cs.shape}")

if __name__ == "__main__":
  os.makedirs("data", exist_ok=True)
  n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
  n_val   = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
  build(n_train, "data/train.npz")
  build(n_val, "data/val.npz")
