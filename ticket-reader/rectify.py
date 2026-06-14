"""
Rectify: photo -> CornerNet -> homography warp -> flat canonical ticket.

This validates the front-end design. Positional corner error just softens the
result; a corner *identity* (orientation) mistake rotates/flips it -> good
diagnostic for how well the net learned orientation under full rotation.

  uv run python rectify.py
"""
import numpy as np
from PIL import Image, ImageDraw
from tinygrad import Tensor
from tinygrad.nn.state import safe_load, load_state_dict
import synth
from train_corners import CornerNet

GEN_W, GEN_H = 640, 480

def predict_corners(net, photo):
  arr = np.asarray(photo.resize((128, 128), Image.BILINEAR), np.float32) / 255.0
  x = Tensor(arr.transpose(2, 0, 1)[None])           # (1,3,128,128)
  p = net(x).numpy().reshape(4, 2)
  p[:, 0] *= GEN_W; p[:, 1] *= GEN_H                  # -> photo pixels
  return p

def rectify(photo, quad):
  """Warp the photo so quad (TL,TR,BR,BL in photo px) maps to a flat TWxTH card."""
  rect = [(0, 0), (synth.TW, 0), (synth.TW, synth.TH), (0, synth.TH)]
  coeffs = synth.find_coeffs(rect, [tuple(p) for p in quad])   # output(rect)->input(photo)
  return photo.transform((synth.TW, synth.TH), Image.PERSPECTIVE, coeffs, Image.BILINEAR)

if __name__ == "__main__":
  net = CornerNet()
  load_state_dict(net, safe_load("corner_net.safetensors"))

  errs = []
  for i in range(6):
    photo, true, fields, _ = synth.sample(GEN_W, GEN_H)
    pred = predict_corners(net, photo)
    errs.append(float(np.sqrt(((pred - true) ** 2).sum(1)).mean()))

    ann = photo.copy(); dr = ImageDraw.Draw(ann)
    for q, col in ((true, (255,0,0)), (pred, (0,255,0))):
      pts = [tuple(map(float, p)) for p in q]
      dr.line(pts + [pts[0]], fill=col, width=3)
    ann.save(f"out/rect_{i}_photo.png")
    rectify(photo, pred).convert("RGB").save(f"out/rect_{i}_pred.png")
    rectify(photo, true).convert("RGB").save(f"out/rect_{i}_true.png")

  print(f"mean corner err over 6 samples: {np.mean(errs):.1f}px")
  print("wrote out/rect_{i}_{photo,pred,true}.png  (photo: red=true green=pred)")
