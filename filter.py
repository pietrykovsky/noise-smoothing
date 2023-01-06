import numpy as np
from PIL import Image


def convolution_filter(image: Image, kernel: np.array) -> Image:
    """
    Apply a convolution filter to the image and return the result.
    Filter is not applied to the edges.

    :param image: Image to be filtered
    :param kernel: Convolution kernel used in filtering
    (size of the kernel should be odd and the shape should be a square e.g. 3x3)
    :return: Filtered image
    """
    pixel_array = np.array(image)
    offset = kernel.shape[0] // 2
    kernel_sum = np.sum(kernel)
    rows, cols, channels = pixel_array.shape
    result = pixel_array.copy()
    for channel in range(channels):
        for i in range(offset, rows - offset):
            for j in range(offset, cols - offset):
                pixel_subarray = np.array(
                    pixel_array[
                        i - offset : i + offset + 1,
                        j - offset : j + offset + 1,
                        channel,
                    ]
                )
                pixel_value = np.sum(pixel_subarray * kernel) // kernel_sum
                if pixel_value > 255:
                    pixel_value = 255
                elif pixel_value < 0:
                    pixel_value = 0
                result[i, j, channel] = pixel_value
    return Image.fromarray(result)


def median_filter(image: Image, kernel_size: int) -> Image:
    """
    Apply a median filter to the image and return the result.
    Filter is not applied to the edges.

    :param image: Image to be filtered
    :param kernel_size: Size of the median kernel (size of the kernel should be odd e.g. 3, 5, 9)
    :return: Filtered image
    """
    pixel_array = np.array(image)
    rows, cols, channels = pixel_array.shape
    result = pixel_array.copy()
    offset = kernel_size // 2
    for channel in range(channels):
        for i in range(offset, rows - offset):
            for j in range(offset, cols - offset):
                pixel_subarray = np.array(
                    pixel_array[
                        i - offset : i + offset + 1,
                        j - offset : j + offset + 1,
                        channel,
                    ]
                )
                pixel_value = int(np.median(pixel_subarray))
                result[i, j, channel] = pixel_value
    return Image.fromarray(result)


def bilateral_filter(image: Image, *args) -> Image:
    # to be implemented
    pass
