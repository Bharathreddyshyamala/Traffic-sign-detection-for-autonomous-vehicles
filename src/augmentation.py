# augmentation.py
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

def save_hyp_yaml(filepath):
    """
    Generate a YOLOv8-compatible hyp.yaml file with augmentation hyperparameters.
    """
    hyp = {
        'flipud': 0.0,        # Vertical flip probability
        'fliplr': 0.5,        # Horizontal flip probability
        'hsv_h': 0.015,       # Hue shift
        'hsv_s': 0.7,         # Saturation shift
        'hsv_v': 0.4,         # Brightness shift
        'mosaic': 1.0,        # Mosaic augmentation
        'rotate': 40.0,       # Rotation degrees
        'scale': 0.5,         # Scaling factor
        'translate': 0.1,     # Translation factor
        'shear': 0.0,         # Shearing factor
        'perspective': 0.0,   # Perspective distortion
        'blur': 0.1           # Motion blur
    }
    try:
        import yaml
        with open(filepath, 'w') as file:
            yaml.dump(hyp, file, default_flow_style=False)
        print(f"Saved hyp parameters to {filepath}")
    except ImportError:
        # Fallback: Write manually
        hyp_lines = "\n".join(f"{k}: {v}" for k, v in hyp.items())
        with open(filepath, 'w') as file:
            file.write(hyp_lines)
        print(f"Saved hyp parameters to {filepath} (manual write)")

def augment_image(image):
    """
    Apply Albumentations augmentation to an image (numpy array).
    Returns an augmented image tensor (for YOLOv8 consumption).
    """
    image = image.astype(np.float32) / 255.0
    augmentations = A.Compose([
        A.RandomBrightnessContrast(p=0.6),  # brightness changes
        A.Rotate(limit=40, p=0.5),         # random rotation
        A.HorizontalFlip(p=0.5),           # horizontal flip
        A.MotionBlur(p=0.1),               # motion blur
        ToTensorV2()                       # convert to tensor
    ])
    augmented = augmentations(image=image)
    return augmented['image']

def visualize_augmentation(image_path):
    """
    Read an image, apply augmentation, and plot original vs augmented.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    augmented_image = augment_image(image.copy())
    # Convert tensor to numpy for display
    augmented_image_np = augmented_image.permute(1, 2, 0).cpu().numpy()
    augmented_image_np = np.clip(augmented_image_np * 255, 0, 255).astype(np.uint8)

    plt.figure(figsize=(8,4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(augmented_image_np, cv2.COLOR_BGR2RGB))
    plt.title("Augmented Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augmentation utilities")
    parser.add_argument("--hyp-file", type=str, default="hyp.yaml", help="Path to save YOLOv8 hyp.yaml")
    parser.add_argument("--image", type=str, help="Path to an image to visualize augmentation")
    args = parser.parse_args()

    # Generate and save hyp.yaml
    save_hyp_yaml(args.hyp_file)

    # If an image is provided, visualize augmentation
    if args.image:
        if os.path.exists(args.image):
            visualize_augmentation(args.image)
        else:
            print(f"Image path {args.image} does not exist.")
