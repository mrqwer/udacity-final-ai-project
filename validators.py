import os
def check_device(yes_or_no: str) -> None:
    if yes_or_no not in ("yes", "no"):
        raise ValueError("The --gpu argument must be either 'yes' or 'no'")

def check_arch(arch: str) -> None:
    if arch != "vgg16":
        raise ValueError("Unsupported Architecture")


def validate_image_path(image_path: str) -> None:
    if not os.path.isfile(image_path):
        raise FileNotFoundError("Error: Image file does not exist.")

    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more valid extensions if needed

    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        raise ValueError("Error: Invalid image file extension.")

def check_top_k(k: int) -> None:
    if k < 0 or k > 20:
        raise ValueError("Invalid range top k")

def check_category_json(filepath: str) -> None:
    if not os.path.isfile(filepath):
        raise FileNotFoundError("Error: Category file does not exist")

    valid_extensions = ['.json']  # Add more valid extensions if needed

    if not any(filepath.lower().endswith(ext) for ext in valid_extensions):
        raise ValueError("Error: Invalid file extension.")

def validate(args: dict) -> None:
    if "gpu" in args:
        check_device(args["gpu"])
    validate_image_path(args["image_path"])
    if "top_k" in args:
        check_top_k(args["top_k"])
    if "category_names" in args:
        check_category_json(args["category_names"])


