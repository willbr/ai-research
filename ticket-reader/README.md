# ticket-reader

WebGPU train-ticket reader: photo → corners → rectify → read fields → JSON.
Built on [tinygrad](https://github.com/tinygrad/tinygrad). Trained entirely on
synthetic data (free labels), so no hand-annotation.

## Pipeline

```
photo → CornerNet (find 4 corners) → homography warp (flat canonical ticket)
      → crop fixed field regions → OCR per field → JSON
```

The corner net + rectification absorb rotation/perspective ("customer photos at
funny angles"); the recognizer only ever sees a clean, front-on ticket.

## Setup

Depends on tinygrad as an editable path dependency at `../tinygrad`, so clone it
alongside this repo:

```bash
git clone https://github.com/tinygrad/tinygrad.git ../tinygrad
uv sync
```

## Run

```bash
# 1. corner detection
uv run python make_data.py 6000 1000     # cached synthetic dataset (parallel)
uv run python train_corners.py           # -> corner_net.safetensors
uv run python rectify.py                 # photo -> rectified ticket (out/)

# 2. field recognizer
uv run python make_ocr_data.py 700 120   # field-crop dataset
uv run python train_ocr.py               # -> ocr_net.safetensors

# 3. end to end
uv run python read_ticket.py             # demo on synthetic samples -> JSON
uv run python read_ticket.py photo.jpg   # real photo
```

On an 8GB machine keep datasets small (it hangs above ~350MB of GPU tensors).
A 16GB+ box (e.g. M4 mini) handles full-size datasets.

## Status

- Corner detection + rectification: working.
- Field recognizer: in progress — fixed-slot/column heads hit alignment limits;
  next is a CTC-based recognizer (variable-length text, no rigid column alignment).

## Files

| file | role |
|------|------|
| `synth.py` | synthetic orange-ticket generator (flat render + homography warp) |
| `make_data.py` | parallel cached corner dataset |
| `train_corners.py` | CornerNet (corner regression) |
| `rectify.py` | corners → homography warp → flat ticket |
| `ocr.py` | charset, input preprocessing, recognizer net |
| `make_ocr_data.py` | parallel field-crop dataset |
| `train_ocr.py` | recognizer training |
| `read_ticket.py` | end-to-end photo → JSON |
