import numpy as np
import cv2
from scipy.ndimage import convolve

def masked_blur(image, mask, ksize):
    if isinstance(ksize, int):
        ksize = (ksize, ksize)

    kernel = np.ones(ksize, dtype=np.float32)
    kernel /= np.sum(kernel)

    mask_f = mask.astype(np.float32)
    
    blurred = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        image_f = image[..., c].astype(np.float32)
        masked_image = image_f * mask_f

        sum_image = convolve(masked_image, kernel, mode='constant', cval=0.0)
        sum_mask = convolve(mask_f, kernel, mode='constant', cval=0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            channel_blur = sum_image / sum_mask
            channel_blur[sum_mask == 0] = 0

        blurred[..., c][mask] = channel_blur[mask]

    return blurred.astype(image.dtype)
