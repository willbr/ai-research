"""
Train the fixed-length field recognizer. Same training shape as train_corners.py
but the loss is per-slot cross-entropy (sparse), summed over the L character slots.

  uv run python train_ocr.py
  EPOCHS=25 uv run python train_ocr.py
"""
import time
import numpy as np
from tinygrad import Tensor, nn, TinyJit
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_state_dict, safe_save
from tinygrad.helpers import getenv
import ocr

def load(path):
  d = np.load(path)
  return Tensor(d["images"]), Tensor(d["labels"].astype(np.int32))

if __name__ == "__main__":
  Xtr, Ytr = load("data/ocr_train.npz")
  Xv, Yv = load("data/ocr_val.npz")
  N = Xtr.shape[0]
  print(f"train {Xtr.shape} val {Xv.shape}  charset={ocr.C} L={ocr.L}")

  net = ocr.OCRNet()
  opt = Adam(nn.state.get_parameters(net), lr=1e-3)
  BS = 256

  @TinyJit
  @Tensor.train()
  def train_step() -> Tensor:
    opt.zero_grad()
    s = Tensor.randint(BS, high=N)
    y = Ytr[s]                                                 # (BS,L) int, 0 == PAD
    logp = net(Xtr[s].float() / 255.0).log_softmax(axis=2)     # (BS,L,C)
    nll = -(logp * y.one_hot(ocr.C)).sum(axis=2)               # (BS,L) per-slot loss
    w = (y != 0).float() * 0.9 + 0.1                           # downweight PAD slots 9:1
    loss = ((nll * w).sum() / w.sum()).backward()
    return loss.realize(*opt.schedule_step())

  def evaluate():
    char_ok = char_tot = str_ok = str_tot = 0
    for i in range(0, Xv.shape[0], 512):
      pred = net(Xv[i:i+512].float() / 255.0).argmax(axis=2).numpy()   # (b,L)
      true = Yv[i:i+512].numpy()
      mask = true != 0
      char_ok += int(((pred == true) & mask).sum()); char_tot += int(mask.sum())
      for p, t in zip(pred, true):
        str_ok += int(ocr.decode(p) == ocr.decode(t)); str_tot += 1
    return char_ok/char_tot*100, str_ok/str_tot*100

  EPOCHS = getenv("EPOCHS", 12)
  steps = N // BS
  for e in range(EPOCHS):
    t = time.perf_counter(); losses = []
    for _ in range(steps): losses.append(train_step().item())
    ca, sa = evaluate()
    print(f"epoch {e:2d}: loss {np.mean(losses):.4f}  char_acc {ca:5.1f}%  field_acc {sa:5.1f}%  ({time.perf_counter()-t:.1f}s)")

  safe_save(get_state_dict(net), "ocr_net.safetensors")
  print("saved ocr_net.safetensors")

  # show a few decodes
  pred = net(Xv[:12].float() / 255.0).argmax(axis=2).numpy()
  true = Yv[:12].numpy()
  print("\n  PRED                   | TRUTH")
  for p, t in zip(pred, true):
    print(f"  {ocr.decode(p):22s} | {ocr.decode(t)}")
