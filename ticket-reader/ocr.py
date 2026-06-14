"""
Fixed-length field recognizer: a field crop (3x32x256) -> L character slots,
each a softmax over the charset. No CTC: fields are fixed-format, so plain
per-slot cross-entropy is simpler and exports to WebGPU as a static graph.

decode() drops PAD; strings shorter than L are PAD-padded on the right.
"""
import numpy as np
from PIL import Image
from tinygrad import Tensor, nn

INW, INH = 352, 32      # net input canvas (W,H); width = L * 16 so columns align to char cells
TEXT_H = 26             # normalise every crop to this glyph height -> char width ~16px = one column
BG = (228, 124, 64)     # ticket orange, used to pad

def to_input(crop: Image.Image) -> np.ndarray:
  """Field crop -> fixed (3,INH,INW) uint8, height-normalised + left-aligned, NOT stretched.
  Keeps monospace character cells a constant pixel width across all fields."""
  w, h = crop.size
  nw = min(INW, max(1, round(w * TEXT_H / h)))
  r = crop.resize((nw, TEXT_H), Image.BILINEAR)
  canvas = Image.new("RGB", (INW, INH), BG)
  canvas.paste(r, (2, (INH - TEXT_H) // 2))
  return np.asarray(canvas, np.uint8).transpose(2, 0, 1)

PAD = "\x00"
CHARS = [PAD] + list(" ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789£>-/.")
C = len(CHARS)          # number of classes
L = 22                  # max field length
_to_idx = {c: i for i, c in enumerate(CHARS)}

def encode(s: str) -> np.ndarray:
  s = s[:L]
  idx = [_to_idx.get(c, _to_idx[" "]) for c in s] + [0] * (L - len(s))
  return np.array(idx, dtype=np.uint8)

def decode(idxs) -> str:
  return "".join(CHARS[int(i)] for i in idxs if int(i) != 0)

class OCRNet:
  """Column-aligned conv recognizer: downsample width 352->L=22 so each output
  column maps to one ~16px monospace char cell, then classify per column.
  Spatial alignment is baked in -> no dense flatten, no CTC."""
  def __init__(self):
    self.c1 = nn.Conv2d(3,  32, 3, stride=2, padding=1)   # 32x352 -> 16x176
    self.c2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)   # -> 8x88
    self.c3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # -> 4x44
    self.c4 = nn.Conv2d(128,128, 3, stride=2, padding=1)  # -> 2x22
    self.c5 = nn.Conv2d(128, C, (2, 1))                   # collapse height -> 1x22
  def __call__(self, x: Tensor) -> Tensor:
    x = self.c1(x).relu(); x = self.c2(x).relu(); x = self.c3(x).relu(); x = self.c4(x).relu()
    x = self.c5(x)                                        # (B, C, 1, L)
    return x.reshape(x.shape[0], C, L).transpose(1, 2)    # (B, L, C) logits
