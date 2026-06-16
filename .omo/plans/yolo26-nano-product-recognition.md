# Beginner YOLO26n Product Recognition Tutorial

## TL;DR
> **Summary**: Build a beginner-friendly, notebook-first YOLO26n product-recognition workflow from a blank repo using Ultralytics, Roboflow annotation/export guidance, CPU-safe smoke training, validation, and local prediction evidence.
> **Deliverables**:
> - `README.md` learning guide and quickstart
> - `requirements.txt` for the beginner environment
> - `notebooks/01_product_detection_yolo26n.ipynb` tutorial notebook
> - `data/product_dataset.yaml` placeholder dataset config
> - `data/README.md` dataset collection, Roboflow export, and local YOLO-format fallback guide
> - `scripts/validate_yolo_dataset.py` label/config smoke validator
> - updated `main.py` that points beginners to the notebook and smoke commands
> **Effort**: Medium
> **Parallel**: YES - 5 waves; Wave 1 runs in parallel, later waves are serial because they edit the same notebook and depend on prior tutorial sections
> **Critical Path**: Wave 1 Tasks 1-3 → Task 4 → Task 5 → Task 6 → Task 7 → Final Verification

## Context

### Original Request
The user asked for help building a YOLO26-nano object detector for product recognition and wants to learn how YOLO26-nano development works while building it.

### Interview Summary
- Use the official `ultralytics` package.
- Treat "YOLO26-nano" as Ultralytics `yolo26n.pt` / `yolo26n.yaml`; no separate `YOLO26-nano` package was found.
- Repository is blank except `main.py` with a comment and `.gitignore` ignoring `.venv`.
- First deliverable is a notebook tutorial.
- Dataset starts from scratch.
- Placeholder classes: `product_a`, `product_b`, `product_c`.
- Annotation workflow: Roboflow recommended, local YOLO-format fallback required.
- Compute: CPU only.
- Verification: smoke tests only; do not add pytest or CI in this first version.

### Metis Review (gaps addressed)
- Add mandatory `yolo26n.pt` availability check before assuming the model works.
- Define tiny MVP dataset size separately from recommended real dataset size.
- Keep Roboflow optional/fallback-safe because accounts and UI steps can vary.
- Avoid API, deployment, Docker, CI, custom YOLO code, and accuracy promises.
- Use executable smoke checks for install, model load, dataset YAML, label format, notebook execution, and prediction artifact creation.

## Work Objectives

### Core Objective
Create a tutorial project that teaches a beginner how to collect product images, annotate/export them for YOLO detection, train or smoke-train YOLO26n on CPU, validate the model, and run predictions locally.

### Deliverables
- `README.md`: repo overview, beginner learning path, commands, expected artifacts.
- `requirements.txt`: minimal dependencies for notebook and validation.
- `notebooks/01_product_detection_yolo26n.ipynb`: end-to-end guided tutorial.
- `data/product_dataset.yaml`: concrete placeholder dataset config.
- `data/README.md`: image collection, Roboflow export, and local fallback instructions.
- `scripts/validate_yolo_dataset.py`: local YOLO label/config validator.
- `main.py`: beginner-friendly pointer to the notebook and smoke commands.

### Definition of Done (verifiable conditions with commands)
- `test -f README.md && test -f requirements.txt && test -f notebooks/01_product_detection_yolo26n.ipynb && test -f data/product_dataset.yaml && test -f data/README.md && test -f scripts/validate_yolo_dataset.py`
- `python -m py_compile main.py scripts/validate_yolo_dataset.py`
- `python -c "import ultralytics; print(ultralytics.__version__)"`
- `python -c "from ultralytics import YOLO; model = YOLO('yolo26n.pt'); print(type(model).__name__)"` or documented blocker if upstream model availability fails.
- `python scripts/validate_yolo_dataset.py --data data/product_dataset.yaml --allow-empty`
- `jupyter nbconvert --to notebook --execute notebooks/01_product_detection_yolo26n.ipynb --output /tmp/yolo26n_product_smoke.ipynb` succeeds in smoke mode.
- `python - <<'PY'
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
img = Path('.omo/evidence/prediction-smoke-input.jpg')
img.parent.mkdir(parents=True, exist_ok=True)
Image.new('RGB', (320, 320), color='white').save(img)
YOLO('yolo26n.pt').predict(source=str(img), save=True, project='.omo/evidence/prediction-smoke', name='predict', exist_ok=True)
assert any(Path('.omo/evidence/prediction-smoke/predict').glob('*'))
print('prediction artifact ok')
PY` succeeds, or `.omo/evidence/prediction-smoke-blocker.txt` documents upstream `yolo26n.pt` availability failure.

### Must Have
- CPU-safe settings: notebook smoke training uses `epochs=1`, tiny dataset guidance, and clear warnings about speed/accuracy.
- Beginner explanations for detection, classes, bounding boxes, YOLO label format, train/val split, validation, prediction, and overfitting.
- MVP dataset guidance: `5 train + 2 val images per class` for structure/smoke testing.
- Recommended learning dataset guidance: `30-50 images per class` with varied backgrounds, angles, lighting, and packaging states.
- Roboflow workflow plus local YOLO folder fallback.
- Clear advanced note that YOLO26 is end-to-end/NMS-free by default; do not make NMS/export configuration part of the MVP.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- Must NOT build a production API, FastAPI service, Streamlit app, Docker setup, CI workflow, or deployment path.
- Must NOT add pytest in the first version.
- Must NOT require GPU or paid Roboflow features.
- Must NOT implement custom YOLO model internals, custom loss functions, or manual NMS.
- Must NOT promise accuracy targets from tiny beginner datasets.
- Must NOT over-engineer a package framework around the notebook.
- Must NOT leave vague placeholders like `[dataset_path]`; use concrete paths under `data/products`.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: smoke tests only; framework is shell/Python/Jupyter commands.
- QA policy: Every task has agent-executed happy and failure/edge scenarios.
- Evidence: `.omo/evidence/task-{N}-{slug}.{ext}`.

## Execution Strategy

### Parallel Execution Waves
> Wave 1 extracts independent shared foundations for parallelism. Later waves are intentionally serial because they update the same notebook and depend on prior tutorial sections.
> Precondition before launching Wave 1: run `mkdir -p .omo/evidence` so all parallel tasks can write evidence safely.

Wave 1: Task 1 scaffold/docs foundation, Task 2 dataset config/validator, Task 3 notebook skeleton/model smoke.
Wave 2: Task 4 data collection/annotation lesson.
Wave 3: Task 5 CPU training/validation lesson.
Wave 4: Task 6 prediction/troubleshooting lesson.
Wave 5: Task 7 README/main integration and complete repo smoke validation.

### Dependency Matrix (full, all tasks)
- Task 1: no dependencies; blocks Tasks 4, 7.
- Task 2: no dependencies; blocks Tasks 4, 5, 7.
- Task 3: no dependencies; blocks Tasks 5, 6, 7.
- Task 4: blocked by Tasks 1, 2; blocks Tasks 5, 7.
- Task 5: blocked by Tasks 2, 3, 4; blocks Tasks 6, 7.
- Task 6: blocked by Tasks 3, 5; blocks Task 7.
- Task 7: blocked by Tasks 1-6.

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 3 tasks → `writing`, `unspecified-high`, `unspecified-high`.
- Wave 2 → 1 task → `writing`.
- Wave 3 → 1 task → `unspecified-high`.
- Wave 4 → 1 task → `quick`.
- Wave 5 → 1 task → `unspecified-high`.

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Create beginner project scaffold and learning guide

  **What to do**: Create `README.md`, `requirements.txt`, `notebooks/`, `data/`, `scripts/`, and `.omo/evidence/` if missing. `requirements.txt` must include `ultralytics`, `notebook`, `jupyter`, `nbconvert`, `pyyaml`, and `Pillow`. `README.md` must explain the repo goal, CPU-only expectations, quickstart commands, expected files, and the learning sequence. Keep `.gitignore` compatible with `.venv`, `runs/`, model weights (`*.pt`), and temporary notebook outputs.
  **Must NOT do**: Do not create API/deployment/CI files. Do not add pytest. Do not add a large `src/` package.

  **Recommended Agent Profile**:
  - Category: `writing` - Reason: mostly beginner-facing documentation and minimal dependency/scaffold decisions.
  - Skills: [] - no specialized skill needed.
  - Omitted: [`frontend-ui-ux`, `debugging`] - no UI or runtime bug yet.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4, 7 | Blocked By: none

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `main.py` - current file is a comment-only placeholder; repo has no established structure.
  - External: `https://docs.ultralytics.com/tasks/detect` - official detect train/val/predict task docs.
  - External: `https://github.com/ultralytics/ultralytics/blob/5ed793530fc438f1105a88cecd75923332881397/README.md#L57-L63` - official install/import pattern.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `test -f README.md && test -f requirements.txt && test -d notebooks && test -d data && test -d scripts`
  - [ ] `grep -q "CPU" README.md && grep -q "yolo26n.pt" README.md && grep -q "notebooks/01_product_detection_yolo26n.ipynb" README.md`
  - [ ] `grep -q "ultralytics" requirements.txt && grep -q "nbconvert" requirements.txt && grep -q "pyyaml" requirements.txt`
  - [ ] `test -s .omo/evidence/task-1-acceptance.txt`

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Scaffold exists and explains beginner path
    Tool: Bash
    Steps: Run `test -f README.md && test -f requirements.txt && test -d notebooks && test -d data && test -d scripts && grep -q "CPU" README.md && grep -q "yolo26n.pt" README.md`
    Expected: Command exits 0.
    Evidence: .omo/evidence/task-1-scaffold.txt

  Scenario: Guardrail files are not accidentally added
    Tool: Bash
    Steps: Run `test ! -f .github/workflows/ci.yml && test ! -f Dockerfile && test ! -f app.py && test ! -f pyproject.toml`
    Expected: Command exits 0 because CI/deployment/package framework is out of scope.
    Evidence: .omo/evidence/task-1-no-scope-creep.txt
  ```

  **Commit**: YES | Message: `docs(scaffold): add beginner yolo26n project foundation` | Files: [`README.md`, `requirements.txt`, `.gitignore`, `notebooks/`, `data/`, `scripts/`]

- [x] 2. Add dataset config, dataset guide, and YOLO label validator

  **What to do**: Create `data/product_dataset.yaml` with concrete placeholder paths under `data/products`, names `product_a`, `product_b`, `product_c`, and relative `train`/`val` paths. Create `data/README.md` explaining MVP dataset size (`5 train + 2 val images per class`), recommended dataset size (`30-50 images per class`), YOLO label format, train/val layout, empty-label behavior, and common annotation mistakes. Create `scripts/validate_yolo_dataset.py` that validates YAML keys, class names, relative paths, image/label pairing, label row length, integer class IDs, and normalized coordinates. Add `--allow-empty` for scaffold validation before real images exist.
  **Must NOT do**: Do not require real images in this task. Do not require a paid Roboflow account. Do not hardcode absolute local machine paths.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: combines small Python validator with exact dataset documentation.
  - Skills: [] - no specialized skill needed.
  - Omitted: [`security-research`] - no security review needed for local validator.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4, 5, 7 | Blocked By: none

  **References**:
  - External: `https://docs.ultralytics.com/datasets/detect` - official YOLO detection dataset format.
  - External: `https://github.com/ultralytics/ultralytics/blob/5ed793530fc438f1105a88cecd75923332881397/docs/en/datasets/detect/index.md#L13-L24` - label format and normalization rules.
  - API/Type: YAML must include `path`, `train`, `val`, and `names`.

  **Acceptance Criteria**:
  - [ ] `test -f data/product_dataset.yaml && test -f data/README.md && test -f scripts/validate_yolo_dataset.py`
  - [ ] `python -m py_compile scripts/validate_yolo_dataset.py`
  - [ ] `python scripts/validate_yolo_dataset.py --data data/product_dataset.yaml --allow-empty`
  - [ ] `python - <<'PY'
from pathlib import Path
import base64
root = Path('/tmp/yolo_bad_labels')
(root / 'images/train').mkdir(parents=True, exist_ok=True)
(root / 'images/val').mkdir(parents=True, exist_ok=True)
(root / 'labels/train').mkdir(parents=True, exist_ok=True)
(root / 'labels/val').mkdir(parents=True, exist_ok=True)
png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII='
(root / 'images/train/bad.png').write_bytes(base64.b64decode(png))
(root / 'labels/train/bad.txt').write_text('0 0.5 0.5 1.5 0.2\n')
Path('/tmp/yolo_bad_labels.yaml').write_text('path: /tmp/yolo_bad_labels\ntrain: images/train\nval: images/val\nnames: [product_a]\n')
PY
mkdir -p .omo/evidence
if python scripts/validate_yolo_dataset.py --data /tmp/yolo_bad_labels.yaml > .omo/evidence/task-2-bad-label-command.txt 2>&1; then exit 1; else grep -Eq '0-1|normalized|coordinate' .omo/evidence/task-2-bad-label-command.txt; fi`
  - [ ] `test -s .omo/evidence/task-2-acceptance.txt`

  **QA Scenarios**:
  ```
  Scenario: Empty beginner scaffold validates in allow-empty mode
    Tool: Bash
    Steps: Run `python scripts/validate_yolo_dataset.py --data data/product_dataset.yaml --allow-empty`
    Expected: Exits 0 and prints a success message that includes `dataset yaml ok` or equivalent.
    Evidence: .omo/evidence/task-2-validator-empty.txt

  Scenario: Malformed YOLO label is rejected
    Tool: Bash
    Steps: Create a temporary dataset under `/tmp/yolo_bad_labels` with one image placeholder and one label row `0 0.5 0.5 1.5 0.2`; run validator against a temporary YAML pointing to it.
    Expected: Exits non-zero and message identifies normalized coordinate outside 0-1.
    Evidence: .omo/evidence/task-2-validator-bad-label.txt
  ```

  **Commit**: YES | Message: `feat(data): add yolo dataset guide and validator` | Files: [`data/product_dataset.yaml`, `data/README.md`, `scripts/validate_yolo_dataset.py`]

- [x] 3. Create notebook skeleton with environment and YOLO26n availability smoke checks

  **What to do**: Create `notebooks/01_product_detection_yolo26n.ipynb` with beginner markdown sections: what object detection is, why YOLO26n/nano is used, CPU expectations, install/import checks, and model-load check. Include executable cells for `import ultralytics`, `from ultralytics import YOLO`, and `YOLO('yolo26n.pt')`. If `yolo26n.pt` fails because upstream availability changed, the notebook must stop with a clear markdown/code message telling the learner to verify Ultralytics YOLO26 support rather than silently switching models.
  **Must NOT do**: Do not hide model-load failure by automatically falling back to YOLO11. Do not add GPU-only instructions.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: notebook JSON must be valid and educational while keeping executable smoke checks.
  - Skills: [] - no specialized skill needed.
  - Omitted: [`debugging`] - this is creation, not bug investigation.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5, 6, 7 | Blocked By: none

  **References**:
  - External: `https://docs.ultralytics.com/models/yolo26` - official YOLO26 model docs.
  - External: `https://docs.ultralytics.com/tasks/detect` - official object detection task docs.
  - External: `https://github.com/ultralytics/ultralytics/blob/5ed793530fc438f1105a88cecd75923332881397/docs/en/tasks/detect.md#L44-L79` - official `yolo26n.pt` train/val/predict examples.

  **Acceptance Criteria**:
  - [ ] `test -f notebooks/01_product_detection_yolo26n.ipynb`
  - [ ] `python - <<'PY'
import json
json.load(open('notebooks/01_product_detection_yolo26n.ipynb'))
print('notebook json ok')
PY`
  - [ ] `grep -q "yolo26n.pt" notebooks/01_product_detection_yolo26n.ipynb && grep -q "CPU" notebooks/01_product_detection_yolo26n.ipynb`
  - [ ] `python -c "from ultralytics import YOLO; model = YOLO('yolo26n.pt'); print(type(model).__name__)"` succeeds or blocker is documented in `.omo/evidence/task-3-yolo26n-blocker.txt`.
  - [ ] `test -s .omo/evidence/task-3-acceptance.txt`

  **QA Scenarios**:
  ```
  Scenario: Notebook is valid JSON and contains YOLO26n smoke check
    Tool: Bash
    Steps: Run the JSON load command and grep for `yolo26n.pt` and `YOLO`.
    Expected: Commands exit 0.
    Evidence: .omo/evidence/task-3-notebook-skeleton.txt

  Scenario: Upstream model unavailable is documented, not silently ignored
    Tool: Bash
    Steps: Run `python -c "from ultralytics import YOLO; model = YOLO('yolo26n.pt'); print(type(model).__name__)"`.
    Expected: If exit 0, save output; if non-zero, create `.omo/evidence/task-3-yolo26n-blocker.txt` with the exact error and do not substitute another model in notebook training cells.
    Evidence: .omo/evidence/task-3-yolo26n-load.txt
  ```

  **Commit**: YES | Message: `feat(notebook): add yolo26n environment smoke tutorial` | Files: [`notebooks/01_product_detection_yolo26n.ipynb`]

- [x] 4. Add data collection, Roboflow annotation, and local fallback notebook lesson

  **What to do**: Extend the notebook and `data/README.md` with a beginner lesson for collecting images, choosing placeholder classes, avoiding duplicate/near-identical photos, splitting train/val, annotating bounding boxes in Roboflow, exporting YOLO format, and placing files under `data/products`. Include a local fallback folder structure: `data/products/images/train`, `data/products/images/val`, `data/products/labels/train`, `data/products/labels/val`. Include edge cases: empty label file for no-object images, multiple objects same class, multiple classes, labels without images, images without labels, class order mismatch.
  **Must NOT do**: Do not require Roboflow API keys in code. Do not automate browser UI steps. Do not require the user to provide real product names.

  **Recommended Agent Profile**:
  - Category: `writing` - Reason: education-heavy content with concrete folder and annotation instructions.
  - Skills: [] - no specialized skill needed.
  - Omitted: [`playwright`] - no browser automation should be performed; Roboflow steps are documented only.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 5, 7 | Blocked By: 1, 2

  **References**:
  - Pattern: `data/product_dataset.yaml` - use its exact paths and class names.
  - External: `https://docs.ultralytics.com/datasets/detect` - YOLO folder and label rules.
  - External: `https://github.com/ultralytics/ultralytics/blob/5ed793530fc438f1105a88cecd75923332881397/docs/en/datasets/detect/index.md#L31-L33` - COCO8-style layout guidance.

  **Acceptance Criteria**:
  - [ ] `grep -q "Roboflow" data/README.md && grep -q "data/products/images/train" data/README.md`
  - [ ] `grep -q "product_a" notebooks/01_product_detection_yolo26n.ipynb && grep -q "normalized" notebooks/01_product_detection_yolo26n.ipynb`
  - [ ] `grep -q "5 train + 2 val" data/README.md && grep -q "30-50" data/README.md`
  - [ ] `test -s .omo/evidence/task-4-acceptance.txt`

  **QA Scenarios**:
  ```
  Scenario: Beginner can follow concrete local dataset layout
    Tool: Bash
    Steps: Run `grep -q "data/products/images/train" data/README.md && grep -q "data/products/labels/val" data/README.md && grep -q "class x_center y_center width height" data/README.md`
    Expected: Command exits 0.
    Evidence: .omo/evidence/task-4-data-guide.txt

  Scenario: Class mismatch risk is documented
    Tool: Bash
    Steps: Run `grep -qi "class order" data/README.md && grep -qi "Roboflow" data/README.md`
    Expected: Command exits 0.
    Evidence: .omo/evidence/task-4-class-mismatch.txt
  ```

  **Commit**: YES | Message: `docs(data): teach product image collection and annotation` | Files: [`data/README.md`, `notebooks/01_product_detection_yolo26n.ipynb`]

- [x] 5. Add CPU-safe training and validation tutorial cells

  **What to do**: Extend the notebook with CPU-safe training and validation cells. Use `data/product_dataset.yaml`, `model='yolo26n.pt'`, `epochs=1`, `imgsz=320` or another explicitly CPU-safe small setting, and a clear note that this is a smoke/learning run, not a quality model. Include validation cell and explain basic metrics at a beginner level. Include a preflight cell that runs `scripts/validate_yolo_dataset.py --data data/product_dataset.yaml` before training and explains `--allow-empty` is only for scaffold validation, not real training.
  **Must NOT do**: Do not use `epochs=100` in the executable CPU smoke cell. Do not promise usable accuracy from MVP data. Do not require GPU.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: connects notebook, validator, Ultralytics training, and CPU constraints.
  - Skills: [] - no specialized skill needed.
  - Omitted: [`debugging`] - no confirmed runtime bug.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 6, 7 | Blocked By: 2, 3, 4

  **References**:
  - Pattern: `scripts/validate_yolo_dataset.py` - must run before training.
  - API/Type: `data/product_dataset.yaml` - training data config.
  - External: `https://docs.ultralytics.com/tasks/detect` - train/val usage.

  **Acceptance Criteria**:
  - [ ] `grep -q "epochs=1" notebooks/01_product_detection_yolo26n.ipynb`
  - [ ] `grep -Eq "imgsz.?=.?(320|416)" notebooks/01_product_detection_yolo26n.ipynb`
  - [ ] `grep -q "validate_yolo_dataset.py" notebooks/01_product_detection_yolo26n.ipynb`
  - [ ] `grep -qi "smoke" notebooks/01_product_detection_yolo26n.ipynb && grep -qi "accuracy" notebooks/01_product_detection_yolo26n.ipynb`
  - [ ] `test -s .omo/evidence/task-5-acceptance.txt`

  **QA Scenarios**:
  ```
  Scenario: Training cell is CPU-safe smoke mode
    Tool: Bash
    Steps: Run `grep -q "epochs=1" notebooks/01_product_detection_yolo26n.ipynb && grep -Eq "imgsz.?=.?(320|416)" notebooks/01_product_detection_yolo26n.ipynb`
    Expected: Command exits 0.
    Evidence: .omo/evidence/task-5-cpu-smoke-settings.txt

  Scenario: Real training is blocked until dataset validates
    Tool: Bash
    Steps: Run `grep -q "validate_yolo_dataset.py" notebooks/01_product_detection_yolo26n.ipynb && grep -q -- "--allow-empty" notebooks/01_product_detection_yolo26n.ipynb`
    Expected: Notebook distinguishes scaffold validation from real training validation.
    Evidence: .omo/evidence/task-5-validation-preflight.txt
  ```

  **Commit**: YES | Message: `feat(notebook): add cpu-safe yolo26n training lesson` | Files: [`notebooks/01_product_detection_yolo26n.ipynb`]

- [x] 6. Add prediction, evidence, troubleshooting, and advanced YOLO26 note

  **What to do**: Extend the notebook with prediction cells that use the trained `runs/detect/train*/weights/best.pt` when available, otherwise explain how to use `yolo26n.pt` for a generic smoke prediction. Add command/cell that writes output under `runs/detect/predict*`. Add troubleshooting for missing dataset files, malformed labels, CPU slowness, similar-looking products, packaging changes, and class confusion. Add a short advanced note that YOLO26 defaults to end-to-end/NMS-free inference and that export/NMS tuning is out of MVP scope.
  **Must NOT do**: Do not add ONNX/TensorRT export as required MVP. Do not introduce manual NMS code.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: focused notebook extension and smoke artifact checks.
  - Skills: [] - no specialized skill needed.
  - Omitted: [`security-research`] - no security-sensitive surface.

  **Parallelization**: Can Parallel: NO | Wave 4 | Blocks: 7 | Blocked By: 3, 5

  **References**:
  - External: `https://docs.ultralytics.com/tasks/detect` - predict usage.
  - External: `https://docs.ultralytics.com/guides/end2end-detection` - end-to-end/NMS-free behavior.
  - External: `https://github.com/ultralytics/ultralytics/blob/5ed793530fc438f1105a88cecd75923332881397/docs/en/models/yolo26.md#L351-L355` - YOLO26 default end-to-end behavior.

  **Acceptance Criteria**:
  - [ ] `grep -q "predict" notebooks/01_product_detection_yolo26n.ipynb && grep -q "runs/detect/predict" notebooks/01_product_detection_yolo26n.ipynb`
  - [ ] `grep -qi "NMS" notebooks/01_product_detection_yolo26n.ipynb && grep -qi "advanced" notebooks/01_product_detection_yolo26n.ipynb`
  - [ ] `grep -qi "CPU" notebooks/01_product_detection_yolo26n.ipynb && grep -qi "troubleshooting" notebooks/01_product_detection_yolo26n.ipynb`
  - [ ] `python - <<'PY'
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
img = Path('.omo/evidence/task-6-prediction-input.jpg')
img.parent.mkdir(parents=True, exist_ok=True)
Image.new('RGB', (320, 320), color='white').save(img)
YOLO('yolo26n.pt').predict(source=str(img), save=True, project='.omo/evidence/task-6-prediction-smoke', name='predict', exist_ok=True)
assert any(Path('.omo/evidence/task-6-prediction-smoke/predict').glob('*'))
print('prediction artifact ok')
PY` succeeds, or `.omo/evidence/task-6-yolo26n-predict-blocker.txt` documents upstream model availability failure.
  - [ ] `test -s .omo/evidence/task-6-acceptance.txt`

  **QA Scenarios**:
  ```
  Scenario: Prediction artifact path is concrete
    Tool: Bash
    Steps: Run `grep -q "runs/detect/predict" notebooks/01_product_detection_yolo26n.ipynb && grep -q "best.pt" notebooks/01_product_detection_yolo26n.ipynb`, then run the prediction smoke Python command from Acceptance Criteria.
    Expected: Commands exit 0 and at least one file exists under `.omo/evidence/task-6-prediction-smoke/predict`, or `.omo/evidence/task-6-yolo26n-predict-blocker.txt` documents model availability failure.
    Evidence: .omo/evidence/task-6-predict-path.txt

  Scenario: Advanced export/NMS remains non-MVP
    Tool: Bash
    Steps: Run `grep -qi "NMS" notebooks/01_product_detection_yolo26n.ipynb && grep -qi "out of scope" notebooks/01_product_detection_yolo26n.ipynb`
    Expected: Command exits 0.
    Evidence: .omo/evidence/task-6-nms-guardrail.txt
  ```

  **Commit**: YES | Message: `docs(notebook): add prediction and troubleshooting lesson` | Files: [`notebooks/01_product_detection_yolo26n.ipynb`]

- [x] 7. Integrate `main.py`, run full smoke validation, and save evidence

  **What to do**: Replace the placeholder `main.py` with a minimal beginner-friendly launcher that prints the tutorial path and recommended smoke commands; it must not run training automatically. Run full smoke validation commands and save outputs under `.omo/evidence/`. If notebook execution cannot complete because no real dataset exists, the notebook must support a scaffold/smoke mode that executes non-training cells and documents the blocker clearly.
  **Must NOT do**: Do not make `main.py` a production inference app. Do not silently skip failing model-load or dataset checks.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: final integration and executable validation across files.
  - Skills: [] - no specialized skill needed.
  - Omitted: [`git-master`] - only use if user explicitly requests commits.

  **Parallelization**: Can Parallel: NO | Wave 5 | Blocks: Final Verification | Blocked By: 1, 2, 3, 4, 5, 6

  **References**:
  - Pattern: `main.py` - currently comment-only; replace with guidance-only entrypoint.
  - Test: commands listed in Definition of Done.
  - API/Type: `notebooks/01_product_detection_yolo26n.ipynb`, `data/product_dataset.yaml`, `scripts/validate_yolo_dataset.py`.

  **Acceptance Criteria**:
  - [ ] `python main.py > .omo/evidence/task-7-main-acceptance.txt && grep -q "notebooks/01_product_detection_yolo26n.ipynb" .omo/evidence/task-7-main-acceptance.txt && grep -q "python -m py_compile" .omo/evidence/task-7-main-acceptance.txt && grep -q "validate_yolo_dataset.py" .omo/evidence/task-7-main-acceptance.txt && grep -q "jupyter nbconvert" .omo/evidence/task-7-main-acceptance.txt`
  - [ ] `python -m py_compile main.py scripts/validate_yolo_dataset.py`
  - [ ] `python scripts/validate_yolo_dataset.py --data data/product_dataset.yaml --allow-empty`
  - [ ] `python -c "import ultralytics; print(ultralytics.__version__)"`
  - [ ] `jupyter nbconvert --to notebook --execute notebooks/01_product_detection_yolo26n.ipynb --output /tmp/yolo26n_product_smoke.ipynb` succeeds, or a documented blocker exists at `.omo/evidence/task-7-notebook-blocker.txt` caused only by upstream/model/dataset constraints explicitly explained in README.
  - [ ] Prediction smoke command from Definition of Done succeeds, or `.omo/evidence/prediction-smoke-blocker.txt` documents upstream `yolo26n.pt` availability failure.
  - [ ] `test -s .omo/evidence/task-7-acceptance.txt`

  **QA Scenarios**:
  ```
  Scenario: Repo smoke validation succeeds or has explicit blocker evidence
    Tool: Bash
    Steps: Run py_compile, validator allow-empty, ultralytics import, and notebook nbconvert command; save stdout/stderr to `.omo/evidence/task-7-full-smoke.txt`.
    Expected: Commands exit 0, except notebook/model-load may produce a documented blocker file if upstream `yolo26n.pt` availability fails.
    Evidence: .omo/evidence/task-7-full-smoke.txt

  Scenario: `main.py` does not trigger accidental training
    Tool: Bash
    Steps: Run `python main.py > .omo/evidence/task-7-main-output.txt` then verify output mentions notebook and does not create `runs/detect/train`.
    Expected: Output is guidance-only; no training directory is created by `main.py`.
    Evidence: .omo/evidence/task-7-main-output.txt
  ```

  **Commit**: YES | Message: `chore(smoke): wire beginner entrypoint and validation evidence` | Files: [`main.py`, `.omo/evidence/`, `README.md`, `notebooks/01_product_detection_yolo26n.ipynb`]

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
> Human approval is a post-verification completion gate only; implementation acceptance criteria and QA evidence above remain fully agent-executable with zero human intervention.
- [x] F1. Plan Compliance Audit — oracle
- [x] F2. Code Quality Review — unspecified-high
- [x] F3. Real Manual QA — unspecified-high
- [x] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit after each task only if the user explicitly wants commits during `/start-work` execution.
- Suggested commit sequence follows each task's `Commit` field.
- Do not commit `.venv`, downloaded model weights (`*.pt`), `runs/`, or temporary notebook outputs.

## Success Criteria
- Beginner can read `README.md`, install dependencies, open the notebook, understand the full product-detection workflow, and run CPU-safe smoke checks.
- Dataset docs teach collection, Roboflow annotation/export, YOLO labels, and local fallback without paid services.
- Notebook contains executable checks for environment, model availability, dataset validation, training smoke mode, validation, and prediction artifacts.
- All verification is agent-executable with concrete commands and evidence paths.
- No API/deployment/CI/custom-YOLO scope creep is introduced.
