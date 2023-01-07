import numpy as np
import cv2
from PIL import Image

from filter import convolution_filter, median_filter
from utils import calculate_mean_squared_error, get_image_difference


if __name__ == "__main__":
    original = Image.open("original.jpg")
    img = Image.open("with-noise.jpg")

    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    blur = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
    gauss_blur3x3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    gauss_blur5x5 = np.array(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ]
    )
    unsharp_masking = np.array(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, -476, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ]
    )
    normalize = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    filters = {
        "sharpen": sharpen,
        "identity": identity,
        "blur": blur,
        "gauss_blur3x3": gauss_blur3x3,
        "gauss_blur5x5": gauss_blur5x5,
        "normalize": normalize,
        "unsharp_masking": unsharp_masking,
    }
    details = ["median", *filters, "bilateral"]
    # applying convolution filters with different kernels
    for name, kernel in filters.items():
        filtered_img = convolution_filter(image=img, kernel=kernel)
        filtered_img.save(f"filtered/{name}.jpg")
        details.append("name")

    # applying median filter
    filtered_img = median_filter(image=img, kernel_size=5)
    filtered_img.save("filtered/median.jpg")

    # applying bilateral filter
    cv2_img = cv2.imread("original.jpg")
    bilateral = cv2.bilateralFilter(cv2_img, 5, 2, 10)
    cv2.imwrite("filtered/bilateral.jpg", bilateral)

    # print mean squared error and save difference images
    for name in details:
        print(f"{name}:")
        filtered_img = Image.open(f"filtered/{name}.jpg")
        print(
            f"mean squared error: {calculate_mean_squared_error(original, filtered_img)}\n"
        )
        get_image_difference(filtered_img, original).save(f"difference/diff_{name}.jpg")
