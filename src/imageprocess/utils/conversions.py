import numpy as np

def conv_to_uint8(self, img: np.array) -> np.array:
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