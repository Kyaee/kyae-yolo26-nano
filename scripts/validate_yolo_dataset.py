from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
REQUIRED_KEYS = {"path", "train", "val", "names"}


class ValidationError(Exception):
    pass


@dataclass(frozen=True)
class DatasetConfig:
    yaml_path: Path
    dataset_root: Path
    train: str
    val: str
    names: list[str]


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")

    try:
        import yaml
    except ImportError:
        return parse_simple_yaml(text)

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValidationError("dataset YAML must be a mapping with path, train, val, and names")
    return loaded


def parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_list_key: str | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        stripped = line.strip()
        if stripped.startswith("-"):
            if current_list_key is None:
                raise ValidationError(f"line {line_number}: list item without a key")
            data.setdefault(current_list_key, []).append(stripped[1:].strip())
            continue

        current_list_key = None
        if ":" not in stripped:
            raise ValidationError(f"line {line_number}: expected 'key: value'")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            data[key] = []
            current_list_key = key
        elif value.startswith("[") and value.endswith("]"):
            data[key] = parse_inline_list(value, line_number)
        else:
            data[key] = value.strip("'\"")

    return data


def parse_inline_list(value: str, line_number: int) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        inner = value[1:-1].strip()
        parsed = [] if not inner else [part.strip().strip("'\"") for part in inner.split(",")]

    if not isinstance(parsed, list):
        raise ValidationError(f"line {line_number}: inline names value must be a list")
    return [str(item) for item in parsed]


def validate_config(yaml_path: Path) -> DatasetConfig:
    if not yaml_path.exists():
        raise ValidationError(f"dataset YAML not found: {yaml_path}")

    raw = load_yaml(yaml_path)
    missing = sorted(REQUIRED_KEYS - raw.keys())
    if missing:
        raise ValidationError(f"dataset YAML is missing required key(s): {', '.join(missing)}")

    path_value = require_string(raw["path"], "path")
    train = require_relative_path(raw["train"], "train")
    val = require_relative_path(raw["val"], "val")
    names = normalize_names(raw["names"])

    dataset_root = Path(path_value)
    if not dataset_root.is_absolute():
        dataset_root = dataset_root.resolve()

    return DatasetConfig(
        yaml_path=yaml_path,
        dataset_root=dataset_root,
        train=train,
        val=val,
        names=names,
    )


def require_string(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"'{key}' must be a non-empty string")
    return value.strip()


def require_relative_path(value: Any, key: str) -> str:
    path_text = require_string(value, key)
    path = Path(path_text)
    if path.is_absolute():
        raise ValidationError(f"'{key}' must be a relative path, not an absolute path: {path_text}")
    if ".." in path.parts:
        raise ValidationError(f"'{key}' must not contain '..': {path_text}")
    return path_text


def normalize_names(value: Any) -> list[str]:
    if isinstance(value, dict):
        try:
            ordered_items = sorted((int(key), name) for key, name in value.items())
        except (TypeError, ValueError) as exc:
            raise ValidationError("'names' mapping keys must be integer class IDs") from exc
        names = [str(name).strip() for _, name in ordered_items]
    elif isinstance(value, list):
        names = [str(name).strip() for name in value]
    else:
        raise ValidationError("'names' must be a list like [product_a, product_b] or an ID-to-name mapping")

    if not names:
        raise ValidationError("'names' must contain at least one class name")
    if any(not name for name in names):
        raise ValidationError("'names' must not contain empty class names")
    if len(set(names)) != len(names):
        raise ValidationError("'names' must not contain duplicate class names")
    return names


def validate_dataset(config: DatasetConfig, allow_empty: bool) -> int:
    image_count = 0
    for split_name, image_relative_dir in (("train", config.train), ("val", config.val)):
        image_dir = config.dataset_root / image_relative_dir
        label_dir = label_dir_for(config.dataset_root, image_relative_dir)
        image_count += validate_split(config, split_name, image_dir, label_dir, allow_empty)

    if image_count == 0 and not allow_empty:
        raise ValidationError("no images found; add images or rerun with --allow-empty for scaffold validation")
    return image_count


def label_dir_for(dataset_root: Path, image_relative_dir: str) -> Path:
    parts = Path(image_relative_dir).parts
    if not parts:
        raise ValidationError("image directory path is empty")
    if parts[0] != "images":
        raise ValidationError("train and val paths should start with 'images/' so labels can mirror them")
    return dataset_root.joinpath("labels", *parts[1:])


def validate_split(
    config: DatasetConfig,
    split_name: str,
    image_dir: Path,
    label_dir: Path,
    allow_empty: bool,
) -> int:
    if not image_dir.exists():
        if label_dir.exists() and any(label_dir.rglob("*.txt")):
            raise ValidationError(f"{split_name} labels exist but image directory is missing: {image_dir}")
        if allow_empty:
            return 0
        raise ValidationError(f"{split_name} image directory not found: {image_dir}")

    if not image_dir.is_dir():
        raise ValidationError(f"{split_name} image path is not a directory: {image_dir}")

    images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images and not allow_empty:
        raise ValidationError(f"no {split_name} images found in {image_dir}")

    for image_path in images:
        label_path = matching_label_path(image_dir, label_dir, image_path)
        if not label_path.exists():
            raise ValidationError(f"missing label file for image: {image_path} -> {label_path}")
        validate_label_file(label_path, len(config.names))

    validate_extra_labels(label_dir, image_dir)
    return len(images)


def matching_label_path(image_dir: Path, label_dir: Path, image_path: Path) -> Path:
    relative_image = image_path.relative_to(image_dir)
    return label_dir / relative_image.with_suffix(".txt")


def validate_extra_labels(label_dir: Path, image_dir: Path) -> None:
    if not label_dir.exists():
        return
    if not label_dir.is_dir():
        raise ValidationError(f"label path is not a directory: {label_dir}")

    image_suffixes_by_stem: dict[Path, set[str]] = {}
    for image_path in image_dir.rglob("*"):
        if image_path.suffix.lower() in IMAGE_SUFFIXES:
            image_suffixes_by_stem.setdefault(image_path.relative_to(image_dir).with_suffix(""), set()).add(
                image_path.suffix.lower()
            )

    for label_path in label_dir.rglob("*.txt"):
        label_stem = label_path.relative_to(label_dir).with_suffix("")
        if label_stem not in image_suffixes_by_stem:
            raise ValidationError(f"label file has no matching image: {label_path}")


def validate_label_file(label_path: Path, class_count: int) -> None:
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValidationError(f"{label_path}:{line_number}: expected 5 values: class x_center y_center width height")

        class_id = parse_class_id(parts[0], label_path, line_number)
        if class_id < 0 or class_id >= class_count:
            raise ValidationError(
                f"{label_path}:{line_number}: class ID {class_id} is outside 0-{class_count - 1}"
            )

        coordinates = parse_coordinates(parts[1:], label_path, line_number)
        for name, value in zip(("x_center", "y_center", "width", "height"), coordinates):
            if not 0.0 <= value <= 1.0:
                raise ValidationError(
                    f"{label_path}:{line_number}: {name}={value} is not a normalized coordinate in 0-1"
                )

        width, height = coordinates[2], coordinates[3]
        if width <= 0.0 or height <= 0.0:
            raise ValidationError(f"{label_path}:{line_number}: width and height must be greater than 0")


def parse_class_id(value: str, label_path: Path, line_number: int) -> int:
    try:
        class_id = int(value)
    except ValueError as exc:
        raise ValidationError(f"{label_path}:{line_number}: class ID must be an integer") from exc

    if str(class_id) != value:
        raise ValidationError(f"{label_path}:{line_number}: class ID must be an integer without decimals")
    return class_id


def parse_coordinates(values: list[str], label_path: Path, line_number: int) -> list[float]:
    coordinates: list[float] = []
    for value in values:
        try:
            coordinates.append(float(value))
        except ValueError as exc:
            raise ValidationError(f"{label_path}:{line_number}: coordinates must be numbers") from exc
    return coordinates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset YAML and label files.")
    parser.add_argument("--data", required=True, type=Path, help="Path to a YOLO dataset YAML file.")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow missing/empty image folders for starter scaffold validation only.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        config = validate_config(args.data)
        image_count = validate_dataset(config, args.allow_empty)
    except ValidationError as exc:
        print(f"dataset yaml error: {exc}", file=sys.stderr)
        return 1

    mode = "allow-empty scaffold" if args.allow_empty and image_count == 0 else "labels checked"
    print(
        "dataset yaml ok: "
        f"{len(config.names)} class(es), {image_count} image(s), root={config.dataset_root} ({mode})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
