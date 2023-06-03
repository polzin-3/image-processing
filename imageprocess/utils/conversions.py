import cv2
import numpy as np

def conv_to_uint8(img: np.array) -> np.array:
    """Scales image pixel values to cover the range 0 -- 255, and
    converts from current dtype to uint8.

    Parameters
    ----------
    img : numpy.ndarray
        image to rescale and convert to uint8

    Returns
    -------
    numpy.ndarray
        rescaled uint8 image.

    """
    max_pix = np.percentile(img, 99.99)  # ignore bright outliers
    min_pix = img.min()
    img = 255 * (img.astype(np.float32) - min_pix) / (max_pix - min_pix)
    img = img.clip(0, 255).astype(np.uint8)
    return img

def conv_to_greyscale(img: np.array) -> np.array:
    """Flattens 3-dimensional BGR image to 2-dimensional greyscale.
    
    Parameters
    ----------
    img : numpy.ndarray
        image to rescale and convert to uint8

    Returns
    -------
    numpy.ndarray
        2-dimensional greyscale image.

    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)