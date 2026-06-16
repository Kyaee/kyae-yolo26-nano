## 2026-06-14 - Task 2 dataset config and validator

- Added `data/product_dataset.yaml` with repo-relative `data/products` root, `images/train`, `images/val`, and placeholder classes `product_a`, `product_b`, `product_c`.
- Added `data/README.md` covering MVP size (`5 train + 2 val images per class`), recommended size (`30-50 images per class`), YOLO label rows, empty label files for no-object images, Roboflow export guidance, and local fallback layout.
- Added `scripts/validate_yolo_dataset.py` as a dependency-light checker. It uses PyYAML when available and a small fallback parser for the tutorial YAML subset when PyYAML is absent.
- Validation edge case discovered: because the dataset YAML itself lives under `data/`, `path: data/products` must be treated as repo-command relative for the documented root-level command. Resolving it relative to the YAML file would incorrectly point at `data/data/products`.
- Smoke checks run: `python -m py_compile scripts/validate_yolo_dataset.py`, `python scripts/validate_yolo_dataset.py --data data/product_dataset.yaml --allow-empty`, and a malformed-label rejection check for width `1.5`.

## 2026-06-14 - Task 2 acceptance evidence

- Created `.omo/evidence/task-2-acceptance.txt` recording the passed compile check, allow-empty scaffold validation, and malformed-label rejection evidence.

## 2026-06-14 - Task 4 notebook and layout note

- Extended `notebooks/01_product_detection_yolo26n.ipynb` with a beginner dataset lesson covering placeholder classes, train/val splits, Roboflow as optional, and the local fallback layout under `data/products/images/...` and `data/products/labels/...`.
- Added wording for common annotation mistakes, including empty label files, missing image or label pairs, class order mismatch, multiple products in one frame, and packaging variation.

## 2026-06-14 - Task 5 CPU training and validation lesson

- Extended `notebooks/01_product_detection_yolo26n.ipynb` with a preflight validator cell that runs `scripts/validate_yolo_dataset.py --data data/product_dataset.yaml` before training; the notebook now says `--allow-empty` is only for scaffold validation, not real training.
- Added a CPU-safe smoke training cell using `model='yolo26n.pt'`, `data='data/product_dataset.yaml'`, `epochs=1`, `imgsz=320`, and `device='cpu'` so beginners can learn the workflow without a GPU or a long training run.
- Validation wording now frames precision, recall, mAP, and class confusion as beginner learning signals only, avoiding any promise of useful accuracy from the tiny smoke dataset.

## 2026-06-14 - Task 6 prediction, troubleshooting, and advanced note

- Extended `notebooks/01_product_detection_yolo26n.ipynb` with a prediction flow that prefers the newest `runs/detect/train*/weights/best.pt` checkpoint and falls back to `yolo26n.pt` as a smoke path when no trained weights exist yet.
- Added prediction output wording that points learners to `runs/detect/predict_cpu_smoke` so the saved artifacts are easy to find under the expected `runs/detect/predict*` pattern.
- Added beginner troubleshooting notes for missing dataset files, malformed labels, CPU slowness, similar-looking products, packaging changes, and class confusion.
- Added the advanced YOLO26 note that inference is end-to-end and NMS-free by default, while export/NMS tuning stays out of the MVP scope.
- Local prediction smoke remains blocked here because `ultralytics` is not installed, so task 6 uses blocker evidence instead of a live prediction run.

## 2026-06-14 - Task 7 main launcher and smoke evidence

- Replaced the placeholder `main.py` with a guidance-only beginner launcher that prints `notebooks/01_product_detection_yolo26n.ipynb`, four copyable smoke commands, and an explicit note that it does not train, download weights, or require a GPU.
- Passed smoke commands: `python main.py > .omo/evidence/task-7-main-acceptance.txt`, `python -m py_compile main.py scripts/validate_yolo_dataset.py`, and `python scripts/validate_yolo_dataset.py --data data/product_dataset.yaml --allow-empty`.
- Blocked smoke commands: `python -c "import ultralytics; print(ultralytics.__version__)"` and the prediction smoke command because `ultralytics` is not installed; `jupyter nbconvert --to notebook --execute notebooks/01_product_detection_yolo26n.ipynb --output /tmp/yolo26n_product_smoke.ipynb` because `jupyter`/`nbconvert` is not installed in this environment.
- Blocker files written: `.omo/evidence/task-7-notebook-blocker.txt` and `.omo/evidence/prediction-smoke-blocker.txt`.

## 2026-06-14 - Final review scaffold wording remediation

- Updated `README.md` quickstart and learning sequence so the empty starter repo is described as a scaffold checkpoint first, not a full top-to-bottom training/prediction run.
- Updated `main.py` to separate scaffold-safe commands from data-ready commands, with wording that strict dataset validation must pass before full notebook training and prediction.
- Made `notebooks/01_product_detection_yolo26n.ipynb` scaffold-safe: empty train/val image folders trigger `--allow-empty` validation and explicit skips for training, validation, and prediction; real train/val images switch the notebook back to strict preflight and CPU smoke cells.
- Updated stale notebook opening, prediction, and closing wording that still referenced environment-only checks or task 7 wiring future work.
- Added `.gitignore` rules for local `data/products/**` artifacts while preserving `.gitkeep` sentinels and README files if they are added under that tree.
