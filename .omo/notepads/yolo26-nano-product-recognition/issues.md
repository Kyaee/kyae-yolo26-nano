## 2026-06-14 - Task 2 dataset config and validator

- No blocking issues. The requested notepad files were absent at task start, so this file was created while preserving the requested notepad directory.
- Validation ambiguity resolved locally: `path: data/products` is documented as a repo-relative beginner path, while `train` and `val` are enforced as relative paths below that root. Absolute `path` remains supported for temporary validation fixtures, but the committed project YAML does not use an absolute machine path.

## 2026-06-14 - Task 4 folder layout note

- Kept the beginner fallback layout anchored at `data/products/images/train`, `data/products/images/val`, `data/products/labels/train`, and `data/products/labels/val` so Roboflow exports and hand-made labels share one shape.
