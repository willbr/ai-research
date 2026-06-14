"""
Corner-detection net: photo (3x128x128) -> 8 numbers = 4 ticket corners (x,y)
normalised to [0,1], ordered TL,TR,BR,BL in the ticket's own frame.

This is the tinygrad-heavy step. The train loop mirrors examples/beautiful_mnist.py:
keep the dataset on-device, draw random batches inside a @TinyJit step, MSE loss,
Adam, .realize(). val metric is mean corner error in 640x480 pixels (interpretable).

  uv run python train_corners.py            # default short run
  EPOCHS=30 uv run python train_corners.py   # longer
"""
import time
import numpy as np
from PIL import Image, ImageDraw
from tinygrad import Tensor, nn, TinyJit
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_state_dict, safe_save
from tinygrad.helpers import getenv

GEN_W, GEN_H = 640, 480

class CornerNet:
  def __init__(self):
    self.c1 = nn.Conv2d(3, 16, 3, stride=2)    # 128 -> 63
    self.c2 = nn.Conv2d(16, 32, 3, stride=2)   # 63 -> 31
    self.c3 = nn.Conv2d(32, 64, 3, stride=2)   # 31 -> 15
    self.c4 = nn.Conv2d(64, 64, 3, stride=2)   # 15 -> 7
    self.l1 = nn.Linear(64*7*7, 128)
    self.l2 = nn.Linear(128, 8)
  def __call__(self, x: Tensor) -> Tensor:
    x = self.c1(x).relu(); x = self.c2(x).relu(); x = self.c3(x).relu(); x = self.c4(x).relu()
    return self.l2(self.l1(x.flatten(1)).relu())

def load_u8(path):
  d = np.load(path)
  return Tensor(d["images"]), Tensor(d["corners"])   # uint8 (N,3,128,128), f32 (N,8)

if __name__ == "__main__":
  Xtr, Ytr = load_u8("data/train.npz")
  Xv, Yv = load_u8("data/val.npz")
  N = Xtr.shape[0]
  print(f"train {Xtr.shape} val {Xv.shape}")

  net = CornerNet()
  opt = Adam(nn.state.get_parameters(net), lr=1e-3)
  BS = 256
  scale = Tensor([GEN_W, GEN_H])  # for pixel-space error

  @TinyJit
  @Tensor.train()
  def train_step() -> Tensor:
    opt.zero_grad()
    s = Tensor.randint(BS, high=N)
    x = Xtr[s].float() / 255.0
    loss = ((net(x) - Ytr[s]) ** 2).mean().backward()
    return loss.realize(*opt.schedule_step())

  def evaluate() -> float:
    errs = []
    for i in range(0, Xv.shape[0], 512):
      p = net(Xv[i:i+512].float() / 255.0)
      d = (p - Yv[i:i+512]).reshape(-1, 4, 2) * scale
      errs.append((d*d).sum(axis=2).sqrt().mean().item())
    return float(np.mean(errs))

  EPOCHS = getenv("EPOCHS", 10)
  steps = N // BS
  for e in range(EPOCHS):
    t = time.perf_counter(); losses = []
    for _ in range(steps): losses.append(train_step().item())
    print(f"epoch {e:2d}: loss {np.mean(losses):.5f}  val_corner_err {evaluate():5.1f}px  ({time.perf_counter()-t:.1f}s)")

  safe_save(get_state_dict(net), "corner_net.safetensors")
  print("saved corner_net.safetensors")

  # visual sanity check: draw predicted (green) vs true (red) corners on a few val imgs
  import os; os.makedirs("out", exist_ok=True)
  pred = net(Xv[:6].float() / 255.0).numpy().reshape(-1, 4, 2)
  true = Yv[:6].numpy().reshape(-1, 4, 2)
  imgs = Xv[:6].numpy().transpose(0, 2, 3, 1).astype(np.uint8)  # (6,128,128,3)
  for i in range(6):
    im = Image.fromarray(imgs[i]).resize((256, 256), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    for pts, col in ((true[i], (255,0,0)), (pred[i], (0,255,0))):
      q = [(float(x)*256, float(y)*256) for x, y in pts]
      dr.line(q + [q[0]], fill=col, width=2)
    im.save(f"out/pred_{i}.png")
  print("wrote out/pred_0..5.png (red=true, green=pred)")
