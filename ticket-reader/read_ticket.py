"""
End-to-end: photo -> corners -> rectify -> crop fixed regions -> OCR -> JSON.

  uv run python read_ticket.py path/to/photo.jpg   # real photo
  uv run python read_ticket.py                      # demo on synthetic samples
"""
import sys, json
import numpy as np
from PIL import Image
from tinygrad import Tensor
from tinygrad.nn.state import safe_load, load_state_dict
import synth, ocr
from train_corners import CornerNet
from rectify import predict_corners, rectify

# region key -> output JSON key
FIELD_JSON = {
  "class": "class", "origin": "origin", "dest": "destination", "type": "ticket_type",
  "route": "route", "price": "price", "date": "date", "adults": "passengers", "number": "ticket_number",
}

def read_ticket(photo, corner_net, ocr_net):
  quad = predict_corners(corner_net, photo)
  flat = rectify(photo, quad).convert("RGB")
  keys = list(synth.REGIONS.keys())
  crops = [ocr.to_input(flat.crop(synth.REGIONS[k])).astype(np.float32) / 255.0 for k in keys]
  pred = ocr_net(Tensor(np.stack(crops))).argmax(axis=2).numpy()   # one batched forward over all fields
  out = {}
  for k, idxs in zip(keys, pred):
    s = ocr.decode(idxs).strip()
    if k == "route": s = s.replace("ROUTE ", "")
    out[FIELD_JSON[k]] = s
  return flat, out

if __name__ == "__main__":
  cn = CornerNet(); load_state_dict(cn, safe_load("corner_net.safetensors"))
  on = ocr.OCRNet(); load_state_dict(on, safe_load("ocr_net.safetensors"))

  if len(sys.argv) > 1:
    photo = Image.open(sys.argv[1]).convert("RGB").resize((640, 480))
    _, data = read_ticket(photo, cn, on)
    print(json.dumps(data, indent=2, ensure_ascii=False))
  else:
    import os; os.makedirs("out", exist_ok=True)
    for i in range(3):
      photo, _, fields, _ = synth.sample()
      flat, data = read_ticket(photo, cn, on)
      flat.save(f"out/read_{i}_rectified.png")
      print(f"\n=== sample {i} ===  truth: {fields}")
      print(json.dumps(data, indent=2, ensure_ascii=False))
