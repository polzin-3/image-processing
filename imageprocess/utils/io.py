import cv2
import numpy as np

def read_image(img_path: str, bgr: bool = True) -> np.array:
    """Reads in image file at given file path.

    Parameters
    ----------
    img_path: str
        File path to image file
    bgr: bool, optional
        Return BGR channels of the image as separate array dimensions, or
        return single greyscale channel

    Returns
    -------
    numpy.ndarray
        array of values corresponding to image
    """
    img = cv2.imread(img_path, -1)
    if img.shape[2] == 4:
        # set masked pixels to zero
        img[img[:, :, 3] == 0] = 0
    if bgr and img.shape[2] > 1:
        img = img[:, :, :3]
    elif (not bgr) and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        pass

    return img