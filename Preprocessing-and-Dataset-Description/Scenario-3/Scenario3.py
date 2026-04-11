import os
import cv2
import numpy as np
from PIL import Image
import torch
import albumentations as A

# Device selection (for future PyTorch operations)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bone_suppression_processing(img_array: np.ndarray) -> np.ndarray:
    """
    Applies a simple simulation of Bone Suppression.
    Goal: To relatively darken bones and enhance soft tissue details (e.g., lungs).

    Steps:
      1. Convert to 8-bit grayscale format (0–255, uint8).
      2. Generate a "bone-weighted" component using Gaussian blur.
      3. Subtract a portion of this component from the original image to suppress bones.
      4. Apply CLAHE for local contrast enhancement.
      5. Return as float32 in the [0, 1] range.
    """

    # 1. Normalize to 8-bit grayscale format
    if img_array.dtype == np.uint16:
        # 16 bit [0, 65535] → 8 bit [0, 255]
        img_array = (img_array / 256).astype(np.uint8)
    elif img_array.dtype != np.uint8:
        # Scale to 0–255 range for other types (e.g., floats)
        max_val = float(img_array.max()) if img_array.size > 0 else 0.0
        if max_val > 0:
            img_array = (img_array / max_val * 255.0).astype(np.uint8)
        else:
            # If the image is all zeros, return a black 8-bit image
            img_array = np.zeros_like(img_array, dtype=np.uint8)

    # Convert to grayscale if the image is color, otherwise copy
    if img_array.ndim == 3:
        # Assuming typical color space; using BGR to GRAY conversion
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_array.copy()

    # 2. Gaussian blur: Simulates the low-frequency component of bright bone structures
    sigma_bone = 5.0
    # ksize=(0, 0) → kernel size is automatically selected based on sigma
    img_bone = cv2.GaussianBlur(img_gray, ksize=(0, 0), sigmaX=sigma_bone)

    # 3. Bone suppression simulation
    # img_gray - bone_weight * img_bone → Bright bone structures are partially darkened
    bone_weight = 0.5  # Suppression intensity
    img_suppressed = (
        img_gray.astype(np.float32) - bone_weight * img_bone.astype(np.float32)
    )

    # Clip to the 0–255 range and convert back to uint8
    img_suppressed = np.clip(img_suppressed, 0, 255).astype(np.uint8)

    # 4. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(12, 12))
    img_final = clahe.apply(img_suppressed)

    # 5. Convert to float32 output in the [0, 1] range
    img_float_output = img_final.astype(np.float32) / 255.0

    return img_float_output


# Augmentation pipeline that only performs resizing
transform_pipeline = A.Compose([
    A.Resize(224, 224, interpolation=cv2.INTER_LINEAR),
])


def process_image(image_path: str, output_path: str) -> None:
    """
    Loads a single image, applies Bone Suppression, resizes it to 224x224, and saves it.

    Steps:
      1. Load the image in grayscale (L) mode.
      2. Apply bone_suppression_processing.
      3. Resize to 224x224 (using Albumentations).
      4. Duplicate the channel to 3 channels (for ImageNet pre-trained CNNs).
      5. Save to disk in 0–255 uint8 format.
    """
    try:
        # 1. Load the image (grayscale)
        image_pil = Image.open(image_path).convert('L')
        image_np = np.array(image_pil)

        # 2. Apply bone suppression processing
        processed_image = bone_suppression_processing(image_np)

        # 3. Prepare for Albumentations: (H, W) → (H, W, 1)
        processed_image_expanded = np.expand_dims(processed_image, axis=-1)
        augmented_dict = transform_pipeline(image=processed_image_expanded)
        final_image = augmented_dict["image"]  # (224, 224, 1), float32, [0, 1]

        # 4. Expand to 3 channels (duplicate the same gray channel three times)
        final_image_rgb = np.concatenate(
            [final_image, final_image, final_image],
            axis=-1
        )  # (224, 224, 3)

        # 5. Convert to 0–255 uint8 and save
        final_image_uint8 = (final_image_rgb * 255.0).astype(np.uint8)
        output_image = Image.fromarray(final_image_uint8)
        output_image.save(output_path)

    except Exception as e:
        # Simple error handling (commented out print for quiet operation)
        # print(f"Error processing {image_path}: {e}")
        pass