import numpy as np
from PIL import Image


def get_image_difference(image1: Image, image2: Image) -> Image:
    """
    Calculates the difference between images of the same size.

    :param image1: PIL Image object
    :param image2: PIL Image object
    :return: Difference factor
    """
    image1_arr = np.array(image1)
    image2_arr = np.array(image2)
    diff_arr = np.array(image1_arr - image2_arr)
    return Image.fromarray(np.abs(diff_arr))


def calculate_mean_squared_error(image1: Image, image2: Image) -> float:
    """
    Calculates the mean squared error of the images of the same size.

    :param image1: PIL Image object
    :param image2: PIL Image object
    :return: Difference factor
    """
    image1_arr = np.array(image1)
    image2_arr = np.array(image2)
    result = ((image1_arr - image2_arr) ** 2).mean()
    return float(result)
