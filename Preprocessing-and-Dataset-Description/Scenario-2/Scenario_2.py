import cv2
import numpy as np
from PIL import Image
import torch
import albumentations as A

# Set device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def alternative_radiology_processing(img_array: np.ndarray) -> np.ndarray:
    """
    Alternative processing chain using OpenCV/Numpy for radiological images:
      1. Normalization to 8-bit (0–255, uint8)
      2. Median filtering for noise reduction
      3. CLAHE for local contrast enhancement
      4. Sharpening with a simple high-pass kernel
      5. Normalization back to [0, 1] float32
    """

    # 1. Ensure 8-bit grayscale (0–255 range)
    if img_array.dtype != np.uint8:
        if img_array.dtype == np.uint16:
            # Scale 16-bit [0, 65535] → 8-bit [0, 255]
            img_array = (img_array / 256).astype(np.uint8)
        else:
            # Generic float / other types → scale by max
            max_val = float(img_array.max()) if img_array.size > 0 else 0.0
            if max_val > 0:
                img_array = (img_array / max_val * 255.0).astype(np.uint8)
            else:
                # Completely zero image
                img_array = np.zeros_like(img_array, dtype=np.uint8)

    # 2. Ensure single channel (convert from color to grayscale if necessary)
    if img_array.ndim == 3:
        # OpenCV expects BGR ordering; if array is RGB, this is an approximation
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_array

    # 3. Median filtering: good for salt-and-pepper noise, preserves edges relatively well
    img_denoised = cv2.medianBlur(img_gray, 3)

    # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_denoised)

    # 5. Simple sharpening using a high-pass-like kernel
    kernel = np.array(
        [
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1],
        ],
        dtype=np.float32,
    )
    img_sharpened = cv2.filter2D(img_clahe, ddepth=-1, kernel=kernel)

    # 6. Normalize result back to [0, 1] float32 for deep learning models
    img_float_output = img_sharpened.astype(np.float32) / 255.0

    return img_float_output


# Augmentation pipeline (Resize only)
transform_pipeline = A.Compose([
    A.Resize(224, 224, interpolation=cv2.INTER_LINEAR),
])


def process_image(image_path: str, output_path: str) -> None:
    """
    Loads, processes, resizes, and saves a single image for DL model input.

    Steps:
      1. Load as grayscale (PIL)
      2. Apply alternative_radiology_processing
      3. Resize to 224x224 with Albumentations
      4. Convert to 3-channel RGB (for ImageNet pretrained models)
      5. Save as uint8 image
    """
    try:
        # 1. Load image (grayscale)
        image_pil = Image.open(image_path).convert('L')
        image_np = np.array(image_pil)

        # 2. Apply image processing
        processed_image = alternative_radiology_processing(image_np)

        # 3. Resize (expand dims to (H, W, 1) for Albumentations)
        processed_image_expanded = np.expand_dims(processed_image, axis=-1)
        augmented_dict = transform_pipeline(image=processed_image_expanded)
        final_image = augmented_dict["image"]  # (224, 224, 1), float32 in [0, 1]

        # 4. Convert to 3-channel RGB
        final_image_rgb = np.concatenate(
            [final_image, final_image, final_image],
            axis=-1
        )  # (224, 224, 3)

        # 5. Convert back to 0–255 uint8 and save
        final_image_uint8 = (final_image_rgb * 255.0).astype(np.uint8)
        output_image = Image.fromarray(final_image_uint8)
        output_image.save(output_path)

    except Exception as e:
        # print(f"Error processing {image_path}: {e}")
        pass

