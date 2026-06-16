NOTEBOOK_PATH = "notebooks/01_product_detection_yolo26n.ipynb"

SCAFFOLD_COMMANDS = [
    "python -m py_compile main.py scripts/validate_yolo_dataset.py",
    "python scripts/validate_yolo_dataset.py --data data/product_dataset.yaml --allow-empty",
    "python -c \"import ultralytics; print(ultralytics.__version__)\"",
]

DATA_READY_COMMANDS = [
    "python scripts/validate_yolo_dataset.py --data data/product_dataset.yaml",
    "jupyter nbconvert --to notebook --execute notebooks/01_product_detection_yolo26n.ipynb --output /tmp/yolo26n_product_smoke.ipynb",
]


def main() -> None:
    print("YOLO26n product-recognition tutorial")
    print()
    print(f"Start with the notebook: {NOTEBOOK_PATH}")
    print()
    print("Safe scaffold checkpoint commands to copy and run before adding data:")
    for command in SCAFFOLD_COMMANDS:
        print(f"- {command}")
    print()
    print("After data/products has train and val images with matching labels, run:")
    for command in DATA_READY_COMMANDS:
        print(f"- {command}")
    print()
    print("The full notebook trains and predicts only after the strict dataset preflight passes.")
    print("This launcher is guidance-only: it does not train, download weights, or require a GPU.")


if __name__ == "__main__":
    main()
