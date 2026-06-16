# Product dataset guide

This folder is the local fallback dataset location for the YOLO26n product-recognition tutorial. You can annotate with Roboflow and export YOLO format, or you can build the same folder layout by hand without a paid Roboflow account.

## Dataset size

- MVP smoke dataset: collect **5 train + 2 val images per class**. With the placeholder classes in `product_dataset.yaml`, that means 15 training images and 6 validation images total.
- Recommended learning dataset: collect **30-50 images per class** before expecting useful behavior. Vary backgrounds, distance, angle, lighting, packaging condition, shelf position, and partial occlusion.

Tiny datasets are for checking that the tutorial, labels, and training command work. They do not prove real accuracy.

## Local fallback layout

Place files under this structure:

```text
data/products/
  images/
    train/
      product_a_001.jpg
    val/
      product_a_101.jpg
  labels/
    train/
      product_a_001.txt
    val/
      product_a_101.txt
```

`data/product_dataset.yaml` points YOLO at these relative paths:

```yaml
path: data/products
train: images/train
val: images/val
names:
  - product_a
  - product_b
  - product_c
```

Keep the order of `names` stable. Class ID `0` means `product_a`, `1` means `product_b`, and `2` means `product_c`.

## YOLO label format

Each image should have one matching `.txt` file in the mirrored `labels` folder. For example:

- Image: `data/products/images/train/product_a_001.jpg`
- Label: `data/products/labels/train/product_a_001.txt`

Each object gets one row:

```text
class x_center y_center width height
```

Example with one `product_a` box:

```text
0 0.512 0.488 0.310 0.420
```

Rules:

- `class` must be an integer from `0` to `2` for the placeholder classes.
- `x_center`, `y_center`, `width`, and `height` must be normalized numbers from `0` to `1`.
- `width` and `height` must be greater than `0`.
- One image can have multiple rows if it contains multiple products.
- A no-object/background image should still have a matching empty `.txt` file.

## Roboflow workflow, optional

Roboflow is useful for drawing boxes and exporting labels, but this tutorial does not require a paid account or API key. If you use Roboflow, export in YOLO detection format and copy the exported `images` and `labels` folders into `data/products/` so the layout matches the local fallback above.

Before training, compare the Roboflow class order with `data/product_dataset.yaml`. A class order mismatch silently teaches the model the wrong names.

## Common annotation mistakes

- Missing label file for an image. Use an empty `.txt` file for no-object images.
- Label file without a matching image.
- Six or more columns in a row, or fewer than five columns.
- Pixel coordinates instead of normalized `0-1` coordinates.
- Class IDs that are not integers, are negative, or are outside the `names` list.
- Boxes that are too loose, too tight, or drawn around multiple products when only one product should be labeled.
- Mixing train and validation images after labels were exported.
- Packaging changes are fine as long as the class stays the same and the box still fits the object.

Keep the notebook and any Roboflow export aligned with the same `data/products/images/...` and `data/products/labels/...` layout. That is the local fallback path this tutorial expects.

Run the validator after copying or editing labels:

```bash
python scripts/validate_yolo_dataset.py --data data/product_dataset.yaml
```

For the empty starter scaffold only, use:

```bash
python scripts/validate_yolo_dataset.py --data data/product_dataset.yaml --allow-empty
```

Do not use `--allow-empty` as proof that real training data is ready. It only confirms the YAML and folder rules are understandable before images exist.
