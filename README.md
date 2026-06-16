# YOLO26n Product Recognition, beginner guide

This repo is a notebook-first starting point for learning product detection with Ultralytics YOLO26n on CPU only.

The main lesson lives in `notebooks/01_product_detection_yolo26n.ipynb`. Start there, then come back to this guide when you want the setup steps or the learning sequence. The empty starter repo supports a safe scaffold checkpoint first; real training and prediction need product images and YOLO labels under `data/products/`.

## What this project is for

Use this repo to learn:

- what object detection does
- how YOLO label files work
- how to run Ultralytics on a CPU laptop
- how to smoke test a model load with `yolo26n.pt`
- how to move from a notebook into a real dataset later

## CPU only setup

You do not need a GPU for the first pass. CPU training is slow, so this starter begins with install checks, model-load checks, and scaffold dataset validation before any training cells run.

After you add real product images and matching YOLO label files, keep the dataset tiny and treat the first CPU runs as learning checks, not accuracy checks.

## Quickstart

1. Create and activate a virtual environment.
2. Install the beginner dependencies.
3. Open the notebook and run the scaffold checkpoint sections first. Do not run the training, validation, or prediction cells as a success path until `data/products/` contains train and val images with matching labels.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Learning sequence

Follow this order when you study the project:

1. Read this README for the big picture.
2. Open `notebooks/01_product_detection_yolo26n.ipynb`.
3. Learn the detection basics, boxes, classes, and YOLO label format.
4. Run the install, model-load, and scaffold dataset checks with CPU only.
5. Add product images and labels under `data/products/`, then rerun the notebook so the strict preflight can unlock the training, validation, and prediction lessons.

## Expected files

- `README.md`
- `requirements.txt`
- `notebooks/`
- `data/`
- `scripts/`

## Dependency list

Install the notebook and Ultralytics stack with `requirements.txt`. It includes:

- `ultralytics`
- `notebook`
- `jupyter`
- `nbconvert`
- `pyyaml`
- `Pillow`

## Notes for later lessons

- Keep model runs CPU only unless a later lesson says otherwise.
- Use `yolo26n.pt` for the beginner smoke check.
- Keep temporary notebook output out of git.
