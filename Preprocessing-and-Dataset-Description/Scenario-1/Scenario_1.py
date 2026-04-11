import cv2
import numpy as np
from PIL import Image
from skimage import exposure, filters
import torch
import albumentations as A

# Set device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def radiology_friendly_processing(img_array: np.ndarray) -> np.ndarray:
    """
    Applies steps optimized for radiological image enhancement.
    Includes:
      1. Normalization to [0, 1]
      2. Gamma Correction
      3. CLAHE (adaptive histogram equalization)
      4. Unsharp Masking (sharpening)
    """

    # 1. Normalize to 0-1 range
    img_float = img_array.astype(np.float32) / 255.0

    # Ensure single channel (convert to grayscale average if input is color)
    if img_float.ndim == 3:
        img_float = np.mean(img_float, axis=2)

    # 2. Power Law (Gamma Correction)
    gamma = 1.2
    img_gamma = np.power(img_float, gamma)

    # 3. CLAHE (Adaptive Histogram Equalization)
    img_clahe = exposure.equalize_adapthist(img_gamma, clip_limit=0.01)

    # 4. Sharpening (Unsharp Masking)
    blurred = filters.gaussian(img_clahe, sigma=1)
    img_sharpened = img_clahe + (img_clahe - blurred) * 1.0

    # Clip values to stay in [0, 1]
    img_sharpened = np.clip(img_sharpened, 0.0, 1.0)

    return img_sharpened


# Augmentation Pipeline (Resize only)
transform_pipeline = A.Compose([
    A.Resize(224, 224, interpolation=cv2.INTER_LINEAR),
])


def process_image(image_path: str, output_path: str) -> None:
    """
    Loads, processes, resizes, and saves a single image for DL model input.

    Steps:
      1. Load as grayscale (PIL)
      2. If 16-bit, convert to 8-bit
      3. Apply radiology-friendly processing
      4. Resize to 224x224 with Albumentations
      5. Convert to 3-channel RGB (for ImageNet pretrained models)
      6. Save as uint8 image
    """
    try:
        # 1. Load Image (grayscale)
        image_pil = Image.open(image_path).convert('L')
        image_np = np.array(image_pil)

        # 16-bit to 8-bit conversion (if needed)
        if image_np.dtype == np.uint16:
            image_np = (image_np / 256).astype(np.uint8)

        # 2. Apply image processing
        processed_image = radiology_friendly_processing(image_np)

        # 3. Resize (expand dims to (H, W, 1) for Albumentations)
        processed_image_expanded = np.expand_dims(processed_image, axis=-1)
        augmented_dict = transform_pipeline(image=processed_image_expanded)
        final_image = augmented_dict["image"]  # (224, 224, 1), float32 in [0, 1]

        # 4. Convert to 3-channel RGB
        final_image_rgb = np.concatenate([final_image, final_image, final_image], axis=-1)

        # 5. Convert back to 0-255 uint8 and save
        final_image_uint8 = (final_image_rgb * 255).astype(np.uint8)
        output_image = Image.fromarray(final_image_uint8)
        output_image.save(output_path)

    except Exception as e:
        # TODO
        # print(f"Error processing {image_path}: {e}")
        pass

