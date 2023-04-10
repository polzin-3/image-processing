import cv2
import numpy as np

def read_image(img_path: str, bgr: bool = True) -> np.array:
    """Reads in image file at given file path.

    Parameters
    ----------
    img_path : str
        File path to image file

    Returns
    -------
    numpy.ndarray
        array of values corresponding to image
    """
    img = cv2.imread(img_path, -1)
    if img.shape[2] == 4:
        # set masked pixels to zero
        img[img[: ,: ,3] == 0] = 0
    if bgr:
        img = img[:, :, :3]
    #ortho = cv2.cvtColor(ortho, cv2.COLOR_BGR2GRAY)

    return img