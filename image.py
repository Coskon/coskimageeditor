import os
import imageio.v3 as iio
from numba import jit
from copy import deepcopy
import numpy as np
from PIL import Image
from PIL.ImageDraw import floodfill
from sklearn.cluster import KMeans
from collections import Counter
from scipy.signal import convolve2d
from cv2 import dilate
from typing import Union
from c_image import CDither, COutline

Image.MAX_IMAGE_PIXELS = None

randomplt = np.hstack((np.array([[np.random.randint(0, 255) for k in range(3)] for i in range(8)]), np.full((8, 1), 255)))
FILTERS = {    # [R, G, B, A, I] RGB: Red, green and blue values. A: Alpha channel. I: Invert image (0 or 1)
        "identity": [1.0, 1.0, 1.0, 1.0, 0.0],
        "cinematic": [0.8, 0.8, 1.0, 1.0, 0.0],
        "sepia": [(0.393, 0.769, 0.189), (0.349, 0.686, 0.168), (0.272, 0.534, 0.131), 1.0, 0.0],
        "sepia2": [0.8, 0.75, 0.45, 1.0, 0.0],
        "sunset": [0.85, 0.45, 0.2, 1.0, 0.0],
        "warm": [(1.0, 0.2, 0.0), (0.0, 1.0, 0.0), (0.0, -0.2, 1.0), 1.0, 0.0],
        "cold": [(1.0, 0.0, 0.0), (0.2, 1.0, 0.0), (0.2, 0.0, 1.0), 1.0, 0.0],
        "ocean": [0.5, 0.5, 1.0, 1.0, 0.0],
        "radioactive": [0.2, 0.8, 0.2, 1.0, 0.0],
        "overdrive": [0.8, 0.6, 0.2, 1.0, 1.0],
        "underwater": [0.5, 0.9, 1, 1.0, 0.0],
        "red": [0.9, 0.4, 0.4, 1.0, 0.0],
        "blue": [0.4, 0.4, 0.9, 1.0, 0.0],
        "green": [0.4, 0.9, 0.4, 1.0, 0.0],
        "inverse": [1.0, 1.0, 1.0, 1.0, 1.0],
        "changepalette": [1, 1, 1, 1, 0],
        "identity_tuple": [(1, 0, 0), (0, 1, 0), (0, 0, 1), 1, 0],
        "greyscale": [],
        "custom": []
    }
PALETTES = {
    "red": [[255, 255, 255, 255], [244, 104, 66, 255], [170, 47, 13, 255], [0, 0, 0, 255]],
    "green": [[255, 255, 255, 255], [196, 244, 65, 255], [109, 169, 12, 255], [0, 0, 0, 255]],
    "blue": [[255, 255, 255, 255], [40, 140, 244, 255], [20, 100, 210, 255], [0, 0, 0, 255]],
    "yellow": [[255, 255, 255, 255], [244, 235, 65, 255], [169, 164, 12, 255], [0, 0, 0, 255]],
    "red_mono": [[79, 20, 3, 255], [255, 227, 219, 255]],
    "green_mono": [[29, 56, 1, 255], [238, 255, 219, 255]],
    "blue_mono": [[2, 50, 79, 255], [219, 249, 255, 255]],
    "yellow_mono": [[35, 50, 1, 255], [255, 249, 185, 255]],
    "cmyk": [[0, 0, 0, 255], [255, 255, 255, 255], [230, 235, 10, 255], [10, 235, 255, 255], [235, 10, 255, 255]],
    "sepia_tri": [[60, 20, 10, 255], [160, 80, 40, 255], [140, 110, 60, 255], [230, 200, 150, 255]],
    "sepia": [[230, 200, 150, 255], [200, 170, 120, 255], [180, 150, 100, 255], [160, 80, 40, 255],
              [160, 130, 80, 255], [140, 110, 60, 255], [120, 90, 40, 255], [100, 70, 20, 255], [80, 50, 0, 255],
              [60, 20, 10, 255], [90, 50, 30, 255], [80, 40, 20, 255], [70, 30, 10, 255], [40, 10, 0, 255]],
    "purple": [[0, 0, 0, 255], [90, 30, 140, 255], [230, 220, 225, 255], [40, 7, 25, 255]],
    "gameboy": [[202, 220, 159, 255], [15, 56, 15, 255], [48, 98, 48, 255], [139, 172, 15, 255], [155, 188, 15, 255]],
    "random": randomplt.tolist(),
    "random1": [[70, 19, 5, 255], [194, 84, 249, 255], [43, 33, 111, 255], [196, 96, 138, 255], [111, 145, 149, 255],
                [190, 33, 114, 255], [38, 34, 113, 255], [107, 34, 27, 255], [102, 252, 214, 255], [37, 91, 188, 255]],
    "random2": [[136, 78, 13, 255], [242, 202, 202, 255], [213, 138, 117, 255], [2, 255, 96, 255], [214, 24, 109, 255],
                [191, 112, 48, 255], [133, 228, 109, 255], [79, 122, 243, 255], [10, 57, 28, 255], [15, 72, 35, 255]],
    "random3": [[50, 223, 225, 255], [219, 231, 20, 255], [22, 15, 2, 255], [47, 72, 54, 255], [32, 106, 180, 255],
                [222, 242, 205, 255], [251, 7, 172, 255], [188, 1, 200, 255], [72, 107, 58, 255], [87, 246, 104, 255]]
}
NEAREST = "nearest"
BILINEAR = "bilinear"
RESIZE_METHODS = [NEAREST, BILINEAR]
RESIZE_METHODS = [method.upper() for method in RESIZE_METHODS]

FILTER_LIST = list(FILTERS.keys())
PALETTE_LIST = list(PALETTES.keys())
RED, GREEN, BLUE, YELLOW, PURPLE, ORANGE, PINK, BROWN, BLACK, WHITE, GRAY, CYAN, MAGENTA, OLIVE, TEAL, LIME = \
    [255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255], \
    [128, 0, 128, 255], [255, 165, 0, 255], [255, 192, 203, 255], [139, 69, 19, 255], \
    [0, 0, 0, 255], [255, 255, 255, 255], [128, 128, 128, 255], [0, 255, 255, 255], \
    [255, 0, 255, 255], [128, 128, 0, 255], [0, 128, 128, 255], [50, 205, 50, 255]


def findClosest(target_color: Union[np.ndarray, list], color_palette: np.ndarray) -> np.ndarray:
    """
    Gets closest color to the target_color in the palette given

    :param target_color: Color to replace
    :param color_palette: Palette of colors
    :return: Color from the palette
    """
    return color_palette[np.argmin(np.linalg.norm(color_palette - np.array(target_color), axis=1))]


def findPalette(image_pixels: np.ndarray, num_colors: int, findType: str = 'kmeans') -> np.ndarray:
    """
    Given an image, tries to find the palette of num_colors colors that best represents it.

    Algorithms:
        -kmeans: Uses KMeans algorithm, best but slow, it's recommended to calculate the palette once.

        -popularity: Gets the most frequent colors with some extra tweaks.

    :param image_pixels: Image as a numpy array
    :param num_colors: Number of colors to grab from the image
    :param findType: Algorithm to use to find the best colors: kmeans, popularity
    :return:
    """
    h, w, hasAlpha = image_pixels.shape
    if hasAlpha != 4: image_pixels = np.dstack((image_pixels, np.full((h, w), 255, dtype=np.uint8)))
    if findType.lower() == 'kmeans':
        image_pixels = image_pixels.reshape(-1, 4)
        kmeans = KMeans(n_clusters=num_colors, n_init=10)
        kmeans.fit(image_pixels)
        cluster_centers = kmeans.cluster_centers_
        return cluster_centers.astype(int)
    else:
        lightness_values = (0.299 * image_pixels[:, :, 0] + 0.587 * image_pixels[:, :, 1] + 0.114 * image_pixels[:, :, 2]).flatten()
        selected_indices = np.argsort(lightness_values)[:num_colors]
        lightest_colors = [tuple(image_pixels.ravel()[i:i+3]) for i in selected_indices]
        image_pixels = image_pixels.reshape(-1, 4)
        color_counts = Counter(tuple(image_pixels[i]) for i in range(h))
        sorted_colors_by_frequency = sorted(color_counts, key=color_counts.get, reverse=True)
        lightest_colors = [color + (255,) for color in lightest_colors]
        optimized_palettes = np.concatenate((np.array(sorted_colors_by_frequency[:num_colors//2]), np.array(lightest_colors[:num_colors//2])))
        return optimized_palettes


def changePalette(target_array: np.ndarray, color_palette: Union[np.ndarray, list]) -> np.ndarray:
    """
    Changes the palette of an image to the given palette, without extra processing

    :param target_array: Image as a numpy array
    :param color_palette: New palette of colors
    :return: Modified image
    """
    color_palette = np.array(color_palette)
    a = np.linalg.norm(target_array[:, :, np.newaxis, :] - color_palette, axis=-1)
    return color_palette[np.argmin(a, axis=-1)].astype(np.uint8)


def randomPalette(N: int) -> list:
    """
    Makes a random palette

    :param N: Number of colors to include
    :return: Generated palette of colors
    """
    palette = []
    for i in range(N): palette.append([np.random.randint(0, 256) for _ in range(3)] + [255])
    return palette


def saturatePalette(colorList: Union[np.ndarray, list], sat: float) -> list:
    """
    Saturates the colors of the given palette

    :param colorList: Palette to modify
    :param sat: New saturation to apply
    :return: Palette with saturated colors
    """
    for i in range(len(colorList)):
        r, g, b = (colorList[i][k]/255 for k in range(3))
        luminance = (r+g+b)/3
        r = luminance + (r-luminance) * sat
        g = luminance + (g-luminance) * sat
        b = luminance + (b-luminance) * sat
        r, g ,b = max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b))
        colorList[i] = [255*r, 255*g, 255*b, 255]
    return colorList


def extendImage(image: Union[str, np.ndarray], N: int) -> np.ndarray:
    """
    Extends an image adding N pixels to each border

    :param image: Path to image or numpy array
    :param N: Pixels to add to each border
    :return: Extended image
    """
    if isinstance(image, str): image = iio.imread(image)
    h, w, hasAlpha = image.shape
    if hasAlpha != 4: image = np.dstack((image, np.full((h, w), 255, dtype=np.uint8)))

    height, width, channels = image.shape

    new_height = height + 2 * N
    new_width = width + 2 * N
    extended_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    extended_image[N:N+height, N:N+width] = image

    extended_image[:N, N:N+width] = image[0]
    extended_image[N+height:, N:N+width] = image[-1]
    extended_image[N:N+height, :N] = image[:, 0].reshape(-1, 1, channels)
    extended_image[N:N+height, N+width:] = image[:, -1].reshape(-1, 1, channels)

    for i in range(N):
        extended_image[i, :N] = image[0, 0]
        extended_image[i, N+width:] = image[0, -1]
        extended_image[N+height+i, :N] = image[-1, 0]
        extended_image[N+height+i, N+width:] = image[-1, -1]

    return extended_image


def resizeImage(image: Union[str, np.ndarray], scale: float = 1, size: tuple[int, int] = (None, None),
                resizeMethod: str = NEAREST) -> np.ndarray:
    """
    Resize an image to the selected scale/size (prioritizes size)

    :param image: Path to image or numpy array
    :param scale: Resizing scale
    :param size: (width, height) to resize the image into
    :param resizeMethod: Method of resizing to use: BILINEAR, NEAREST
    :return: Resized image
    """
    if isinstance(image, str): image = iio.imread(image)
    h, w, hasAlpha = image.shape
    if hasAlpha != 4: image = np.dstack((image, np.full((h, w), 255, dtype=np.uint8)))
    if not any(size): newh, neww = h*scale, w*scale
    else: newh, neww = size[1], size[0]
    x_scale, y_scale = neww / w, newh / h
    x, y = np.meshgrid(np.arange(neww), np.arange(newh))
    resizeMethod = resizeMethod.lower()
    if resizeMethod == BILINEAR:
        x1, y1 = (x / x_scale).astype(int), (y / y_scale).astype(int)
        x2, y2 = np.clip(x1 + 1, 0, w - 1), np.clip(y1 + 1, 0, h - 1)
        dx, dy = x / x_scale - x1, y / y_scale - y1
        interpolated_channels = [
            (1 - dx) * (1 - dy) * image[y1, x1, channel] +
            (1 - dx) * dy * image[y2, x1, channel] +
            dx * (1 - dy) * image[y1, x2, channel] +
            dx * dy * image[y2, x2, channel]
            for channel in range(4)
        ]
        return np.stack(interpolated_channels, axis=-1).astype(np.uint8)
    elif resizeMethod == NEAREST:
        return image[(y / y_scale).astype(int), (x / x_scale).astype(int)]
    raise ValueError(f"Please select a supported resize method: {', '.join([f'`{method}`' for method in RESIZE_METHODS])}")


def gaussianKernel(N: int, sigma: float) -> list:
    kernel = np.zeros((N, N))
    center = N // 2
    norm_factor = 2 * np.pi * sigma ** 2
    for i in range(N):
        for j in range(N):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / norm_factor
    return kernel.tolist()


def circleKernel(radius: int) -> list:
    d = 2*radius+1
    kernel = np.zeros((d, d))
    center = d // 2
    for i in range(d):
        for j in range(d):
            x, y = center - i, center - j
            if np.sqrt(x**2+y**2) <= radius+np.sqrt(center/(2*d)): kernel[i ,j] = 1
    return kernel.tolist()


def crossKernel(s: int) -> list:
    kernel = np.zeros((s, s))
    for i in range(s):
        for j in range(s):
            if i == j or i == s-j-1: kernel[i][j] = 1
    return kernel.tolist()


def motionKernel(N: int, direction: str, pos: int) -> list:
    kernel = np.zeros((N, N))
    if direction == "vertical":
        for i in range(N):
            kernel[i, pos] = 1
    if direction == "horizontal":
        for j in range(N):
            kernel[pos, j] = 1
    if direction == "diag1":
        for k in range(N):
            if k+pos >= N: break
            if k+pos >= 0: kernel[k+pos, k] = 1
    if direction == "diag2":
        for k in range(N):
            if k + pos >= N: break
            if k + pos >= 0: kernel[-k-1, k + pos] = 1
    return kernel.tolist()


def imageKernel(image: Union[str, np.ndarray], kernelType: str, customKernel: Union[np.ndarray, list] = np.ones((3,3)).tolist(),
                passes: int = 1, gaussian: Union[list[int, float], np.ndarray, None] = None, kernelSize: int = 3,
                motionDirection: str = "vertical", motionPosition: int = 1, power: int = None, invert: bool = False,
                fastMode: bool = True, transparent: bool = False,  keepTransparency: bool = True, thickness: int = None,
                sharpness: float = 0.0) -> np.ndarray:
    """
    Modify an image applying convolution operations with kernels (edge detection, blur, ...).

    List of kernels:
        -edge, inverse_edge, identity_edge, edge2: Edge detection kernels.

        -box_blur, circle_blur, cross_blur, gaussian_blur: Blur the image with different methods.

        -motion, high_motion: Movement effect.

        -identity: Leave the image as it is.

        -custom: Use a custom kernel.

    :param image: Path to image or numpy array
    :param kernelType: What kernel is to be applied (see list above)
    :param customKernel: Specify a custom kernel as a NxN matrix (N odd greater than 2, list or ndarray)
    :param kernelSize: Specify the kernelSize to be used for different kernels (odd greater than 2)
    :param passes: Applies the kernel the amount of times specified, the default is applying it one time
    :param gaussian: A list of two elements, the first element is the size of the matrix, the second is the
        standard deviation (sigma)
    :param motionDirection: Specify the direction of the movement effect: vertical, horizontal, diag1, diag2
    :param motionPosition: Specify the position of the ones of the movement matrix kernel
    :param power: Power of the edge detection algorithm
    :param invert: Inverts the kernel
    :param fastMode: Uses convolve2d function (faster)
    :param transparent: Allows transparency for some kernels
    :param keepTransparency: Keeps the transparency data of the original image and replaces the modified image with it
    :param thickness: Change the thickness of the edges. Leave None for the default thickness, else an integer greater than 0
    :param sharpness: Changes the sharpness of the edges. From 0.0 (most dull) to 1.0 (sharpest)
    :return: Modified image as a ndarray
    """
    if gaussian is None: gaussian = [3, 1.0]
    if not isinstance(kernelSize, int) or kernelSize % 2 == 0 or kernelSize < 3: raise ValueError(
        'Size of the kernel must be an odd number greater or equal than 3.')
    if not isinstance(customKernel, list) or not all(len(row) == len(customKernel) for row in customKernel) \
        or len(customKernel) % 2 == 0 or len(customKernel) < 3:
        raise ValueError("Kernel must be a NxN list with N an odd number greater or equal than 3.")
    if not isinstance(kernelType, str):
        raise ValueError("Kernel type must be a string defining one of the available types of kernels.")
    kernelType = kernelType.lower()
    kernels = {
        "edge": [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        "inverse_edge": [[0, 1, 0], [1, -5, 1], [0, 1, 0]],
        "identity_edge": [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
        "edge2": [[0, 0, -1, 0, 0], [0, -1, -8, -1, 0], [-1, -8, 42, -8, -1], [0, -1, -8, -1, 0], [0, 0, -1, 0, 0]],
        "box_blur": np.ones((kernelSize, kernelSize)).tolist(),
        "circle_blur": circleKernel(kernelSize // 2),
        "gaussian_blur": gaussianKernel(gaussian[0], gaussian[1]),
        "motion": motionKernel(kernelSize, motionDirection, motionPosition),
        "cross_blur": crossKernel(kernelSize),
        "high_motion": [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
        "identity": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "custom": customKernel
    }
    if kernelType not in kernels.keys(): raise NameError(f'"{kernelType}" is not a valid kernel type.')
    img = iio.imread(image) if isinstance(image, str) else image
    h, w, hasAlpha = img.shape
    if hasAlpha != 4: img = np.dstack((img, np.full((h, w), 255, dtype=np.uint8)))
    if keepTransparency: alpha_channel = img[:, :, 3]

    if not power and kernelType in ["edge", "identity_edge"]: power = 3
    elif not power: power = 1

    kernel = kernels[kernelType]
    max_val = sum(abs(value) for sublist in kernel for value in sublist)*power*((-1)**invert)
    kernel = [[value / max_val for value in sublist] for sublist in kernel]

    img = img[:, :, :3]
    kernel_len = len(kernel)
    N = kernel_len // 2
    cond1 = kernelType == 'edge' or invert
    for _ in range(passes):
        img = extendImage(img, N)
        if fastMode:
            result = np.zeros((img.shape[0], img.shape[1], 3))
            for channel in range(3):
                result[:, :, channel] = convolve2d(img[:, :, channel], kernel, mode='same', boundary='wrap')
            new_height, new_width, _ = result.shape
            top, bottom, left, right = N, new_height - N, N, new_width - N
            cropped_image = result[top:bottom, left:right]
            pixel = np.dstack((cropped_image, np.full((h, w, 1), 255, dtype=cropped_image.dtype))).astype(np.uint8)
        else:
            pixel = deepcopy(img)

            def convolution(i, j):
                conv = 0
                for k in range(kernel_len):
                    for l in range(kernel_len):
                        conv += img[i-N+k][j-N+l]*kernel[kernel_len-1-k][kernel_len-1-l]
                return conv
            for i in range(N, h+N):
                for j in range(N, w+N):
                    pixel[i][j] = convolution(i, j)

            new_height, new_width, _ = pixel.shape
            top, bottom, left, right = N, new_height - N, N, new_width - N
            cropped_image = pixel[top:bottom, left:right]

            pixel = np.dstack((cropped_image, np.full((h, w, 1), 255, dtype=cropped_image.dtype)))
        if thickness != None:
            k = np.array(circleKernel(round(thickness)))
            L = len(k)
            binary_mask = np.min(pixel, axis=-1) > 25 if cond1 else np.min(pixel, axis=-1) < 25
            at_least_one_black = convolve2d(binary_mask, k, mode='same', boundary='fill', fillvalue=0)
            if kernelType == 'inverse_edge':
                pixel[at_least_one_black > L * sharpness] = [255 * invert, 255 * invert, 255 * invert, 255]
            else:
                pixel[at_least_one_black > L * sharpness] = [255, 255, 255, 255]
        if transparent:
            cond = np.min(pixel, axis=-1) < 150 if cond1 else np.min(pixel, axis=-1) > 150
            pixel[cond] = [0, 0, 0, 0]
        if passes > 1 and _ != passes-1: img = deepcopy(pixel)
    if keepTransparency: pixel = np.dstack((pixel[:, :, 0], pixel[:, :, 1], pixel[:, :, 2], alpha_channel))
    return pixel


def imageFilter(image: Union[str, np.ndarray], filter_type: str, customFilter: Union[np.ndarray, list, None] = None,
                luminance: float = 1.0, colors: int = 4, ditherType: str = None, ditherColors: int = 2,
                saturation: float = 1.0, contrast: float = 1.0, pixelSize: int = 1, resizeMethod: str = 'nearest-neighbor',
                threshold: tuple[int, int] = None, customPalette: Union[np.ndarray, list, None] = None,
                paletteAlgorithm: str = 'kmeans', mirror: str = None, useCDither: bool = True,
                bayer: tuple[int, float] = (2, 16.0)) -> np.ndarray:
    """
    Modify an image by using filters, dithering, etc.

    List of color filters:
        -cinematic, sepia, sepia2, sunset, warm, cold, ocean, radioactive, overdrive, underwater,
        red, blue, green, inverse, greyscale.

    List of special filters:
        -identity, identity_tuple: Leaves the image as it is.

        -changepalette: Changes the colors to the custom palette specified.

        -custom: Uses the "customFilter" specified.

    :param image: Path to image or numpy array
    :param filter_type: What filter is to be applied (see list above)
    :param customFilter: Specify a custom filter of the form [R, G, B, A, I], RGBA values vary from 0.0 to 1.0,
        I is used to invert the colors of the image, either 0 or 1
    :param luminance: Change the luminance/brightness of the image
    :param colors: Quantity of greyscale colors to use with the greyscale filter or greyscale dithering
    :param ditherType: Type of dithering: halftone, floyd-steinberg, hor-floyd-steinberg, fast-floyd-steinberg.
        Halftone only works with greyscale images (use greyscale filter)
    :param ditherColors: Amount of colors of the palette to be extracted from the image for dithering (Floyd-Steinberg)
    :param useCDither: Whether to use or not a C implementation of dithering, being much faster.
    :param paletteAlgorithm: Algorithm to search for the best colors to grab based on ditherColors. Defaults to KMeans,
        anything else will use "color popularity" method
    :param customPalette: Specify a custom palette for dithering. You can also use pre-existing palettes from the
        "palette" dict
    :param saturation: Change the saturation. 1.0 is unmodified, 0.0 is no saturation
    :param contrast: Change the contrast. 1.0 is unmodified
    :param pixelSize: Defaults to 1. If changed it pixelates the image, 1 is the best resolution, then 2, 3, ...
    :param resizeMethod: When changing the pixelSize, you can change how it resizes: nearest-neighbor, bilinear
    :param mirror: Mirrors the image horizontally (h, hor, horizontal) or vertically (v, ver, vert, vertical)
    :param bayer: Tuple containing the size of the bayer matrix and strength of the effect. Used with halftone dithering.
    :param:
    :return: Modified image as a ndarray
    """
    if not isinstance(filter_type, str):
        raise ValueError("Filter type must be a string defining one of the available types of filter.")
    if customFilter == None: customFilter = [1.0, 1.0, 1.0, 1.0, 0.0]
    if isinstance(customPalette, str) and customPalette != "": customPalette = PALETTES[customPalette]
    if isinstance(customPalette, str) and customPalette == "": customPalette = PALETTES["red"],
    FILTERS['greyscale'] = [1.0, colors, 1, 1, 0]
    FILTERS['custom'] = customFilter
    filter_type = filter_type.lower()
    if filter_type not in FILTER_LIST: raise NameError(f'"{filter_type}" is not a valid filter type.')
    greyscale = False
    if not colors: byw = False
    else: byw = True
    if filter_type == "greyscale":
        greyscale = True
    filter = FILTERS[filter_type]
    if len(filter) < 3:
        raise ValueError("Invalid filter length (should be at least 3).")
    if len(filter) == 3:
        filter.append(1)
    if len(filter) == 4:
        filter.append(0)
    if filter_type in ['identity', 'identity_tuple', 'changepalette']: ident = True
    else: ident = False
    img = iio.imread(image) if isinstance(image, str) else image
    h, w, hasAlpha = img.shape
    sval = saturation
    if hasAlpha != 4: img = np.dstack((img, np.full((h, w), 255, dtype=np.uint8)))

    pixel = deepcopy(img)
    if pixelSize != 1:
        pixelSize = int(pixelSize)
        rh, rw = h // pixelSize, w // pixelSize
        pixel = pixel[:rh * pixelSize, :rw * pixelSize].reshape(rh, pixelSize, rw, pixelSize, 4).mean(axis=(1, 3)).astype(np.uint8)
        temph, tempw = h, w
        h, w = rh, rw
    if filter_type == 'changepalette' and not ditherType:
        pixel = changePalette(pixel, customPalette)

    if contrast != 1:
        pixel[..., :3] = np.clip((pixel[..., :3].astype(float) - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)

    if greyscale:
        r, g, b = pixel[:, :, 0].astype(np.int16), pixel[:, :, 1].astype(np.int16), pixel[:, :, 2].astype(np.int16)
        bri = (r+g+b)/3
        pixel = np.stack((bri, bri, bri, pixel[:, :, 3]), axis=-1).astype(np.uint8)
        if byw:
            colorDiv = np.arange(0, 255 + 255 / colors, 255 / colors).tolist()
            intervals = [[int(colorDiv[i]), int(colorDiv[i + 1])] for i in range(colors)]
            for k in range(len(intervals)):
                a = intervals[k][0]
                b = intervals[k][1]
                if k == 0: value = int(a)
                elif k == len(intervals) - 1: value = int(b)
                else: value = (a + b) / 2
                pixel = np.where((a <= pixel) & (pixel <= b), int(value), pixel)
    elif not saturation == 1 or not ident or luminance != 1:
        I = filter[4]
        saturation = (1/3)+saturation*(2/3)-1
        if isinstance(filter[0], tuple): rf1, rf2, rf3 = filter[0][0] + saturation, filter[0][1] - saturation / 2, filter[0][2] - saturation / 2
        else: rf1, rf2, rf3 = filter[0]+saturation, -saturation/2, -saturation/2
        if isinstance(filter[1], tuple): gf1, gf2, gf3 = filter[1][0] - saturation / 2, filter[1][1] + saturation, filter[1][2] - saturation / 2
        else: gf1, gf2, gf3 = -saturation/2, filter[1]+saturation, -saturation/2
        if isinstance(filter[2], tuple): bf1, bf2, bf3 = filter[2][0] - saturation / 2, filter[2][1] - saturation / 2, filter[2][2] + saturation
        else: bf1, bf2, bf3 = -saturation/2, -saturation/2, filter[2]+saturation
        r, g, b = pixel[:,:,0].astype(np.int16), pixel[:,:,1].astype(np.int16), pixel[:,:,2].astype(np.int16)
        r, g, b = np.clip(luminance*((255 * I) + ((-1) ** I)*(rf1*r+rf2*g+rf3*b)), 0, 255), \
                  np.clip(luminance*((255 * I) + ((-1) ** I)*(gf1*r+gf2*g+gf3*b)), 0, 255), \
                  np.clip(luminance*((255 * I) + ((-1) ** I)*(bf1*r+bf2*g+bf3*b)), 0, 255)
        pixel = np.stack((r, g, b, pixel[:,:,3]*filter[3]), axis=-1).astype(np.uint8)

    if threshold:
        pixel = clipBrightness(pixel, threshold)

    if ditherType:
        ditherType = ditherType.replace(" ", "-").lower()
        grey = greyscale or sval == 0
        if useCDither:
            try:
                if not customPalette: customPalette = findPalette(pixel, ditherColors, paletteAlgorithm)
            except:
                pass
            pixel = CDither(pixel, palette=np.array(customPalette), type=ditherType, bayerSize=bayer[0], bayerStrength=bayer[1])
        elif ditherType == 'halftone' and grey:
            if byw:
                colorDiv = np.arange(0, 255 + 255 / colors, 255 / colors).tolist()
                intervals = [[int(colorDiv[i]), int(colorDiv[i + 1])] for i in range(colors)]
                for k in range(len(intervals)):
                    a = intervals[k][0]
                    b = intervals[k][1]
                    if k == 0:
                        value = int(a)
                    elif k == len(intervals) - 1:
                        value = int(b)
                    else:
                        value = (a + b) / 2
                    pixel = np.where((a <= pixel) & (pixel <= b), int(value), pixel)

            oddh, oddw = 0, 0
            if h % 2 != 0: oddh = 1
            if w % 2 != 0: oddw = 1
            for i in range(0, h-1*oddh, 2):
                for j in range(0, w-1*oddw, 2):
                    brght = pixel[i][j][0]
                    if 0 <= brght < 0.2*255:
                        topleft, topright, bottomleft, bottomright = (0 for i in range(4))
                    elif 0.2*255 <= brght < 0.4*255:
                        topleft, topright, bottomright = (0 for i in range(3))
                        bottomleft = 255
                    elif 0.4*255 <= brght < 0.6*255:
                        topleft, bottomright = (0 for i in range(2))
                        bottomleft, topright = (255 for i in range(2))
                    elif 0.6*255 <= brght < 0.8*255:
                        topleft = 0
                        bottomleft, bottomright, topright = (255 for i in range(3))
                    else:
                        topleft, topright, bottomleft, bottomright = (255 for i in range(4))
                    pixel[i][j] = np.array([topleft, topleft, topleft, pixel[i][j][3]])
                    pixel[i+1][j] = np.array([bottomleft, bottomleft, bottomleft, pixel[i+1][j][3]])
                    pixel[i][j+1] = np.array([topright, topright, topright, pixel[i][j+1][3]])
                    pixel[i+1][j+1] = np.array([bottomright, bottomright, bottomright, pixel[i+1][j+1][3]])
        elif ditherType == 'halftone' and not grey: raise NameError('Halftone dithering only available for greyscale images.')
        else:
            if grey:
                cols = np.arange(0, 255 + 255 / (ditherColors - 1), 255 / (ditherColors - 1))
                palette = []
                for col in cols:
                    palette.append([col, col, col, 255])
                palette = np.array(palette)
                pixel = ditherImage(pixel, colors=ditherColors, rsz=0.5, palette=palette, type=ditherType)
            else:
                try:
                    if not customPalette: customPalette = findPalette(pixel, ditherColors, paletteAlgorithm)
                except: pass
                pixel = ditherImage(pixel, colors=ditherColors, rsz=0.5, palette=customPalette, type=ditherType)
    if pixelSize != 1:
        h, w = temph, tempw
        x_scale, y_scale = w / rw, h / rh
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        if resizeMethod == 'bilinear':
            x1, y1 = (x / x_scale).astype(int), (y / y_scale).astype(int)
            x2, y2 = np.clip(x1 + 1, 0, rw - 1), np.clip(y1 + 1, 0, rh - 1)
            dx, dy = x / x_scale - x1, y / y_scale - y1
            interpolated_channels = [
                (1 - dx) * (1 - dy) * pixel[y1, x1, channel] +
                (1 - dx) * dy * pixel[y2, x1, channel] +
                dx * (1 - dy) * pixel[y1, x2, channel] +
                dx * dy * pixel[y2, x2, channel]
                for channel in range(4)
            ]
            pixel = np.stack(interpolated_channels, axis=-1).astype(np.uint8)
        else:
            pixel = pixel[(y / y_scale).astype(int), (x / x_scale).astype(int)]
    if mirror:
        mirror = mirror.lower()
        if mirror in ['h', 'hor', 'horizontal']: pixel = pixel[:, ::-1, :]
        if mirror in ['v', 'ver', 'vert', 'vertical']: pixel = pixel[::-1, :, :]
    return pixel


def imageAddons(image: Union[str, np.ndarray], addonType: str, ellipse: tuple[int, int] = (None, None), radius = None,
                F: float = None, onlyAddon: bool = False, gradientType: str = 'horizontal', invert: bool = False,
                vignetteColor: Union[list, np.ndarray, None] = None, bloom: tuple[float, int, int] = (1.1, 8, 190)) -> np.ndarray:
    """
    Add extra things to an image (vignette, bloom, gradient)

    Addon types:
        -vignette: Adds a vignette of the given color and size.

        -gradient: Adds a horizontal or vertical gradient.

        -bloom: Adds bloom to the bright parts of the image, uses 'bloom' parameters: (strength, size, threshold).

    :param image: Path to image or numpy array
    :param addonType: Type of addon (see above)
    :param ellipse: Vignette ellipse shape
    :param radius: Vignette radius
    :param F: Falloff constant
    :param onlyAddon: Returns only the addon, ignoring the given image
    :param gradientType: Type of gradient (horizontal or vertical)
    :param invert: Invert gradient
    :param vignetteColor: Vignette color
    :param bloom: Bloom parameters (see above)
    :return:
    """
    if vignetteColor is None: vignetteColor = [0, 0, 0, 255]
    if isinstance(image, str): image = iio.imread(image)
    h, w, hasAlpha = image.shape
    if hasAlpha != 4: image = np.dstack((image, np.full((h, w), 255, dtype=np.uint8)))
    addonType = addonType.lower()
    r, g, b, _ = vignetteColor
    large_array = np.zeros((h, w, 4))
    large_array[..., 0], large_array[..., 1], large_array[..., 2], large_array[..., 3] = r, g, b, 255
    if addonType == 'vignette':
        center = h//2,w//2
        if not any(ellipse): ellipse = (h/w, w/h)
        if not radius:
            radius = min(center) * max(ellipse)
            if F != 0: radius *= 1.9

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        distance = 0.99*np.sqrt(ellipse[0]*(x - center[1]) ** 2 + ellipse[1]*(y - center[0]) ** 2)
        colors = np.zeros((h, w, 4), dtype=np.uint8)

        if F == None: falloff = -1.0598/(1+np.exp(2-(4*(distance-radius/3.6))/(radius/1.8)))+1.0184
        elif F == 0: falloff = radius
        else: falloff = 1 - (distance / radius ) ** F

        colors[..., 0] = (distance <= radius) * ((255-r)*falloff)
        colors[..., 1] = (distance <= radius) * ((255-g)*falloff)
        colors[..., 2] = (distance <= radius) * ((255-b)*falloff)
        colors[..., 3] = (distance <= radius) * np.clip((255*(1-falloff)), 0, 255)

        large_array[distance <= radius] = colors[distance <= radius]
        large_array[..., 0], large_array[..., 1], large_array[..., 2] = r, g, b
    if addonType == 'gradient':

        def custom_falloff(V, A):
            N = 255 * (1 - ((V / (A - 1)) ** F))
            return int(np.abs(N))

        gradientType = gradientType.lower()
        if not F: F = 1

        if gradientType == 'horizontal':
            U, B = np.arange(w), w
        elif gradientType == 'vertical':
            U, B = np.arange(h), h

        alpha_values = np.vectorize(lambda U: custom_falloff(U, B))(U)
        if invert: alpha_values = np.flip(alpha_values, axis=-1)
        large_array[..., 3] = alpha_values[:, np.newaxis] if gradientType == 'vertical' else alpha_values
    if addonType == 'bloom':
        def blm(img, i):
            binary_mask = np.sum(image[..., :3], axis=-1) / 3 > bloom[2]
            binary_mask = convolve2d(binary_mask, np.array(circleKernel(i)), mode='same', boundary='fill',
                                     fillvalue=0).astype(bool)
            imgtmp = np.where(binary_mask[..., np.newaxis], img, 0)
            imgtmp[binary_mask] = np.clip(imgtmp[binary_mask].astype(float) + 240, 0, 255)
            return imgtmp
        blmimage = blm(image, 0)
        custom_kernel = np.array(circleKernel(bloom[1]))
        blurred_image = deepcopy(blmimage)
        for channel in range(4):
            blurred_image[..., channel] = convolve2d(blmimage[..., channel], custom_kernel, mode='same', boundary='fill', fillvalue=0)
        imgtmp = blurred_image
        alpha_mask = blurred_image[..., 3] > 0
        imgtmp[..., :3] = 255
        imgtmp[alpha_mask, 3] = np.clip((255 - blurred_image[alpha_mask, 3].astype(float))*bloom[0], 0, 255).astype(np.uint8)
        large_array = deepcopy(imgtmp)

    if onlyAddon: return large_array.astype(np.uint8)
    image = imageInterp(image, large_array, 'alpha_compositing', interp1=1, interp2=1)
    return image.astype(np.uint8)


def imageOutline(image: Union[str, np.ndarray], thickness: int = 5, outlineColor: Union[list, np.ndarray, None] = None,
                 threshold: int = 200, outlineType: str = "circle") -> np.ndarray:
    """
    Adds an outline to an image.

    Outline types:
        -box/square: Sharp edges

        -circle: Smooth edges

        -star: Pointy edges

    :param image: Path to image or numpy array
    :param thickness: Thickness of the outline
    :param outlineColor: Color of the outline
    :param threshold: Transparency threshold to consider as a valid pixel to add outline
    :param outlineType: Type of outline (see above)
    :return: Outlined image
    """
    if outlineColor is None: outlineColor = [0, 0, 0, 255]
    img = iio.imread(image) if isinstance(image, str) else image

    thickness += 2

    def find_nonzero_bounds(arr):
        rows, cols = np.nonzero(arr)
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        return min_row, max_row, min_col, max_col

    if thickness % 2 == 0 or thickness < 0:
        raise ValueError("Thickness must be odd and > 0")

    if len(img.shape) > 3:
        raise ValueError("Invalid image.")

    h, w, alpha = img.shape
    if alpha != 4:
        img = np.dstack((img, np.full((h, w), 255, dtype=np.uint8)))

    new_h = h + 12 * thickness
    new_w = w + 12 * thickness
    resize_img = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    resize_img[6 * thickness:new_h - 6 * thickness, 6 * thickness:new_w - 6 * thickness, :] = img

    pixel = deepcopy(resize_img)

    threshold_mask = resize_img[:, :, 3] > threshold

    if thickness == 1:
        struct = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    else:
        thck_tuple = (thickness, thickness)
        struct = np.ones(thck_tuple)
        outlineType = outlineType.lower()
        if outlineType == "circle":
            center = (thickness - 1) / 2
            radius = thickness // 2

            indices = np.indices(thck_tuple)
            distances = np.sqrt((indices[0] - center) ** 2 + (indices[1] - center) ** 2)

            struct = (distances <= radius + np.sqrt(center/(32*radius)))
        elif outlineType == "star":
            indices = np.abs(np.arange(-thickness // 2 + 1, thickness // 2 + 1))
            struct = (indices[:, np.newaxis] + indices[np.newaxis, :]) <= thickness // 2
    outline_mask = dilate(threshold_mask.astype(np.uint8), np.array(struct).astype(np.uint8))
    outline_indices = np.where(outline_mask)
    pixel[outline_indices[0], outline_indices[1], :] = outlineColor
    pixel = imageInterp(pixel, resize_img, interpType='alpha_compositing', interp1=1, interp2=1)
    min_row, max_row, min_col, max_col = find_nonzero_bounds(pixel[:, :, 3])

    return pixel[min_row:max_row + 1, min_col:max_col + 1, :]


def noise(size: tuple[int, int], noiseType: str, pixelSize: int = 1, rainType: str = 'up_down', randomBias: str = 'linear1',
          perlinNoise: tuple[int, float, int, list[int, int]] = (4, 0.85, 2, [0, 0]), probabilities: tuple[int, int] = (50, 15),
          circles: tuple[Union[None, int], int] = (None, 3), npoly: float = 2, gridSize: int = None, seed: int = None,
          transparent: bool = False, invert: bool = False, serpent: tuple[int, int, int] = (100, 2, 5)) -> np.ndarray:
    """
    Generates random noise of various types

    Noise Types:
        -random: Generic white noise, can be biased with randomBias -> quadratic, quartic, triangular, sine,
        sine_spike, linear1, linear2, npoly, None.

        -colorrandom: Generic colored noise.

        -rain: Rain effect -> up, down, up_down

        -perlin: Perlin noise

        -circles: Randomly creates same radius circles.

        -serpent: Creates a serpent

    :param size: (w, h) Width and height of the image to generate
    :param noiseType: Noise type (see above)
    :param pixelSize: Resolution of the noise (higher: less resolution, 1 is best)
    :param rainType: Rain type (see 'rain' noise type)
    :param randomBias: Bias to apply to the randomness to get different patterns (see above)
    :param perlinNoise: Used in 'perlin' noise type: (octaves, persistence, frequency multiplier, center point)
    :param probabilities: Used in 'rain' type
    :param circles: Used in 'circles' type: (number of circles, radius)
    :param npoly: Used in 'npoly' randomBias
    :param gridSize: Perlin noise grid size
    :param seed: Seed to use for the random patterns (to repeat results)
    :param transparent: Makes some pixels transparent
    :param invert: Inverts the colors
    :param serpent: Used in 'serpent' noise type
    :return: Noise image
    """
    pixel = 255*(1-invert)*np.ones((size[1], size[0], 4)).astype(np.uint8)
    if transparent: pixel[..., 3] = 0
    else: pixel[..., 3] = 255
    if seed: np.random.seed(seed)
    noiseType = noiseType.lower()
    if noiseType == 'random':
        total = np.random.rand(size[1]//pixelSize, size[0]//pixelSize)
        if isinstance(randomBias, str):
            randomBias = randomBias.lower()
            if randomBias == 'quadratic':
                total = 8 * total * (total - 1) + 1
            elif randomBias == 'quartic':
                total = -(32/9) * total * (total - 1) * (total - 2) * (total + 1) + 1
            elif randomBias == 'triangular':
                total = 4*np.abs(total-0.5)-1
            elif randomBias == 'sine':
                total = -2*np.sin(np.pi * total)+1
            elif randomBias == 'sine_spike':
                total = 2*np.abs(np.sin(np.pi * (total-0.5)))-1
            elif randomBias == 'linear1':
                total = -2*total+1
            elif randomBias == 'linear2':
                total = 2*total-1
            elif randomBias == 'npoly':
                c = 2**npoly
                cond1 = total <= 0.5
                cond2 = total > 0.5
                total[cond1] = -c*np.power(total[cond1],npoly)+1
                total[cond2] = c*np.power(1-total[cond2],npoly)-1
        bri = np.clip((255 * (total + 1) / 2), 0, 255).astype(np.uint8)
        if invert: bri = 255 - bri
        tval = bri if transparent else 255
        pixel = resizeImage(np.stack((bri, bri, bri, np.full_like(bri, tval)), axis=-1), pixelSize)
    elif noiseType == 'colorrandom':
        pixel = resizeImage(np.clip(255 * np.random.random((size[1] // pixelSize, size[0] // pixelSize, 3)),
                                    0, 255), pixelSize)
    elif noiseType == 'rain':
        t = 255*invert
        for j in range(size[0]):
            n1 = np.random.randint(1, 1000)
            if n1 >= probabilities[0]:
                if rainType == 'up' or rainType == 'up_down':
                    for i in range(size[1]):
                        n2 = np.random.randint(1, 1000)
                        if n2 >= probabilities[1]: pixel[i][j] = [t, t, t, 255]
                        else: break
                if rainType == 'down' or rainType == 'up_down':
                    for k in range(size[1]):
                        n2 = np.random.randint(1, 1000)
                        if n2 >= probabilities[1]: pixel[-k-1][j] = [t, t, t, 255]
                        else: break
    elif noiseType == 'perlin':
        size = size[1], size[0]
        if not gridSize: gridSize = int(perlinNoise[2]**(perlinNoise[0]))
        gradient_grid = np.random.normal(size=(gridSize, gridSize, 2))
        gradient_grid = gradient_grid / np.linalg.norm(gradient_grid, axis=-1)[..., np.newaxis]
        octaves = perlinNoise[0]
        persistence = perlinNoise[1]

        def lattice_points(sample_points, gridSize):
            sample_points = np.floor(sample_points).astype(int)
            sample_points = np.clip(sample_points, 0, gridSize - 1)

            x0, y0 = sample_points[..., 0], sample_points[..., 1]
            x1, y1 = np.clip(x0 + 1, 0, gridSize - 1), np.clip(y0 + 1, 0, gridSize - 1)

            return (x0, y0), (x1, y0), (x0, y1), (x1, y1)
        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)
        def lerp(t, a, b):
            return a + t * (b - a)
        def perlin(sample_points):
            lattice_pts = lattice_points(sample_points, gradient_grid.shape[0])
            fractions = sample_points - np.stack(lattice_pts[0], axis=-1)
            dot_products = []
            for lattice_pt in lattice_pts:
                distance = sample_points - np.stack(lattice_pt, axis=-1)
                gradient = gradient_grid[lattice_pt]
                dot_products.append(np.sum(gradient * distance, axis=-1))

            t_x = fade(fractions[..., 0])
            ab = lerp(t_x, dot_products[0], dot_products[1])
            cd = lerp(t_x, dot_products[2], dot_products[3])

            t_y = fade(fractions[..., 1])
            return lerp(t_y, ab, cd)

        aspect_ratio = size[0] / size[1]
        if aspect_ratio <= 1: x, y = np.meshgrid(np.linspace(0, 1, size[1]), np.linspace(0, 1*aspect_ratio, size[0]))
        else: x, y = np.meshgrid(np.linspace(0, 1/aspect_ratio, size[1]), np.linspace(0, 1, size[0]))
        sample_points = np.stack((x, y), axis=-1)
        total = np.zeros_like(x)
        frequency, amplitude = 1, 1
        for _ in range(octaves):
            total += perlin((sample_points + perlinNoise[3]+[0.5, 0.5]) * frequency) * amplitude
            frequency *= perlinNoise[2]
            amplitude *= persistence
        total = np.clip((255*(total+1)/2), 0, 255).astype(np.uint8)
        if invert: total = np.clip(255 - total, 0, 255)
        pixel = np.stack((total, total, total, np.full_like(total, 255)), axis=-1)
        if transparent: pixel[:, :, 3] = pixel[:, :, 0]
    elif noiseType == 'circles':
        large_array = np.zeros((size[1], size[0]))
        num_circles = circles[0]
        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        if not num_circles: num_circles = (size[1]+size[0])
        for _ in range(num_circles):
            center = np.random.randint(0, size[0]), np.random.randint(0, size[1])
            distance = 0.88*np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            large_array[distance <= circles[1]] = 1
        k = -1*large_array if invert else 255 - 255 * large_array
        pixel = np.dstack((k, k, k, 255*np.ones(k.shape))).reshape((size[1], size[0], 4))
        if transparent: pixel[((pixel[:, :, :3] == 0).all(axis=-1), 3)] = 0
    elif noiseType == 'serpent':
        large_array = np.zeros((size[1], size[0]))
        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        center = np.random.randint(serpent[2]*serpent[1], size[0]-serpent[2]*serpent[1]), np.random.randint(serpent[2]*serpent[1], size[1]-serpent[2]*serpent[1])
        for _ in range(serpent[0]):
            distance = 0.88 * np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            large_array[distance <= serpent[1]] = 1
            A = 3*np.random.random(2)
            I = np.random.randint(-1, 2, 5)
            index = np.random.randint(0, 5, 2)
            center = center[0]+I[index[0]]*A[0], center[1]+I[index[1]]*A[1]
            large_array += 255/serpent[0]
        large_array = large_array.astype(np.uint8)
        k = 255 - large_array if invert else 255 - 255 * large_array
        pixel = np.dstack((k, k, k, 255 * np.ones(k.shape))).reshape((size[1], size[0], 4))
        if transparent: pixel[:, :, 3] = pixel[:, :, 0]

        if transparent: pixel[((pixel[:, :, :3] == 0).all(axis=-1), 3)] = 0

    return np.array(pixel).astype(np.uint8)


def imageInterp(image1: Union[str, np.ndarray], image2: Union[str, np.ndarray], interpType: str = 'alpha_compositing',
                interp1: float = 0.5, interp2: float = 0.5) -> np.ndarray:
    """
    Combines two images with different methods.

    Interpolation methods (interpType):
        -alpha_compositing: Default, uses alpha compositing which allows proper transparent combination.

        -add: Adds the RGBA values of both images together.

        -multiply: Multiplies the RGBA values of both images together.


    :param image1: Base image.
    :param image2: Secondary image.
    :param interpType: Type of interpolation (see above).
    :param interp1: From 0.0 to 1.0, strength of the first image.
    :param interp2: From 0.0 to 1.0, strength of the second image.
    :return: New image of the combination of the two input images, as a ndarray.
    """
    if isinstance(image1, str): img1 = iio.imread(image1)
    else: img1 = image1
    if isinstance(image2, str): img2 = iio.imread(image2)
    else: img2 = image2
    if isinstance(image1, str): w1, h1 = Image.open(image1).size
    else: w1, h1 = image1.shape[1], image1.shape[0]
    if isinstance(image2, str): w2, h2 = Image.open(image2).size
    else: w2, h2 = image2.shape[1], image2.shape[0]
    if w2 != w1 or h2 != h1: raise ValueError("Both images must be of the same size.")
    interpType = interpType.lower()
    pixel1 = deepcopy(img1).astype(float)
    pixel2 = deepcopy(img2).astype(float)
    if pixel1.shape[2] != 4: pixel1 = np.dstack((pixel1, np.full((h1, w1), 255, dtype=np.uint8)))
    if pixel2.shape[2] != 4: pixel2 = np.dstack((pixel2, np.full((h1, w1), 255, dtype=np.uint8)))
    if interpType == 'add':
        v = interp1*pixel1+interp2*pixel2
        pixel = np.clip(v, 0, 255)
    elif interpType == 'multiply':
        v = interp1*pixel1*interp2*pixel2
        pixel = np.clip(v, 0, 255)
    elif interpType == 'alpha_compositing':
        pixel = deepcopy(img1)
        if pixel.shape[2] != 4: pixel = np.dstack((pixel, np.full((h1, w1), 255, dtype=np.uint8)))
        alpha_channel = interp2*pixel2[:, :, 3]
        v1 = interp1 * (255 - alpha_channel)
        pixel[:, :, 0] = (pixel1[:, :, 0] * v1 + pixel2[:, :, 0] * alpha_channel) // 255
        pixel[:, :, 1] = (pixel1[:, :, 1] * v1 + pixel2[:, :, 1] * alpha_channel) // 255
        pixel[:, :, 2] = (pixel1[:, :, 2] * v1 + pixel2[:, :, 2] * alpha_channel) // 255
    return pixel.astype(np.uint8)


def unihanCharacters(unihanPath = r"./Unihan_IRGSources.txt"):
    from collections import defaultdict

    outdict = defaultdict(list)
    with open(unihanPath) as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            line = line.split("\t")
            if len(line) == 3 and line[1] == "kTotalStrokes":
                try:
                    stroke_count = int(line[2])
                except:
                    continue
                if stroke_count > 7 and any((i % 2 == 0, i % 3 == 0)) or 2825 < i < 2834:
                    continue
                character = chr(int(line[0].lstrip("U+").zfill(8), 16))
                outdict[stroke_count].append(character)
                i += 1

    ordered_characters = ""
    for stroke_count in sorted(outdict.keys()):
        ordered_characters += ''.join(outdict[stroke_count])
    return " " + ordered_characters


def imageToAscii(image: Union[str, np.ndarray], size: int, asciiString: str = "braille", customAscii: str = None,
                 contrast: float = 1, invert: bool = False, dither_type: Union[str, None] = None, smooth: bool = False,
                 accentuate: bool = False, thck: int = 1, replace_space: bool = True, space_strip: bool = False,
                 threshold: tuple = (0, 255), unihanCharacters = None) -> Union[tuple[str, np.ndarray]]:
    """
    Converts an image to text (ascii)

    Conversion types: braille|braille8, braille6, squares, simple, complex

    :param image: Path to image or numpy array
    :param size: Width of the image, aspect ratio is preserved
    :param asciiString: Type of conversion (see above)
    :param customAscii: Custom character density to use
    :param contrast: Change the contrast of the image before converting
    :param invert: Invert the image
    :param dither_type: Dither type to use (See `ditherImage` documentation for all types)
    :param smooth: Blurs the image
    :param accentuate: Tries to accentuate the edges
    :param thck: Edge accentuation thickness
    :param replace_space: Replaces empty characters with the next "smallest" character. Fixes uneven widths
    :param space_strip: Removes empty characters to the right
    :param threshold: Lower and upper bounds to clip the brightness of the image
    :return: The created string and the modified image provided
    """
    braille = 0
    density = customAscii
    if not customAscii:
        braille_dict = {
            '00,00,00,00': '⠀', '00,00,00,01': '⢀', '00,00,00,10': '⡀', '00,00,00,11': '⣀', '00,00,01,00': '⠠', '00,00,01,01': '⢠', '00,00,01,10': '⡠',
            '00,00,01,11': '⣠', '00,00,10,00': '⠄', '00,00,10,01': '⢄', '00,00,10,10': '⡄', '00,00,10,11': '⣄', '00,00,11,00': '⠤', '00,00,11,01': '⢤',
            '00,00,11,10': '⡤', '00,00,11,11': '⣤', '00,01,00,00': '⠐', '00,01,00,01': '⢐', '00,01,00,10': '⡐', '00,01,00,11': '⣐', '00,01,01,00': '⠰',
            '00,01,01,01': '⢰', '00,01,01,10': '⡰', '00,01,01,11': '⣰', '00,01,10,00': '⠔', '00,01,10,01': '⢔', '00,01,10,10': '⡔', '00,01,10,11': '⣔',
            '00,01,11,00': '⠴', '00,01,11,01': '⢴', '00,01,11,10': '⡴', '00,01,11,11': '⣴', '00,10,00,00': '⠂', '00,10,00,01': '⢂', '00,10,00,10': '⡂',
            '00,10,00,11': '⣂', '00,10,01,00': '⠢', '00,10,01,01': '⢢', '00,10,01,10': '⡢', '00,10,01,11': '⣢', '00,10,10,00': '⠆', '00,10,10,01': '⢆',
            '00,10,10,10': '⡆', '00,10,10,11': '⣆', '00,10,11,00': '⠦', '00,10,11,01': '⢦', '00,10,11,10': '⡦', '00,10,11,11': '⣦', '00,11,00,00': '⠒',
            '00,11,00,01': '⢒', '00,11,00,10': '⡒', '00,11,00,11': '⣒', '00,11,01,00': '⠲', '00,11,01,01': '⢲', '00,11,01,10': '⡲', '00,11,01,11': '⣲',
            '00,11,10,00': '⠖', '00,11,10,01': '⢖', '00,11,10,10': '⡖', '00,11,10,11': '⣖', '00,11,11,00': '⠶', '00,11,11,01': '⢶', '00,11,11,10': '⡶',
            '00,11,11,11': '⣶', '01,00,00,00': '⠈', '01,00,00,01': '⢈', '01,00,00,10': '⡈', '01,00,00,11': '⣈', '01,00,01,00': '⠨', '01,00,01,01': '⢨',
            '01,00,01,10': '⡨', '01,00,01,11': '⣨', '01,00,10,00': '⠌', '01,00,10,01': '⢌', '01,00,10,10': '⡌', '01,00,10,11': '⣌', '01,00,11,00': '⠬',
            '01,00,11,01': '⢬', '01,00,11,10': '⡬', '01,00,11,11': '⣬', '01,01,00,00': '⠘', '01,01,00,01': '⢘', '01,01,00,10': '⡘', '01,01,00,11': '⣘',
            '01,01,01,00': '⠸', '01,01,01,01': '⢸', '01,01,01,10': '⡸', '01,01,01,11': '⣸', '01,01,10,00': '⠜', '01,01,10,01': '⢜', '01,01,10,10': '⡜',
            '01,01,10,11': '⣜', '01,01,11,00': '⠼', '01,01,11,01': '⢼', '01,01,11,10': '⡼', '01,01,11,11': '⣼', '01,10,00,00': '⠊', '01,10,00,01': '⢊',
            '01,10,00,10': '⡊', '01,10,00,11': '⣊', '01,10,01,00': '⠪', '01,10,01,01': '⢪', '01,10,01,10': '⡪', '01,10,01,11': '⣪', '01,10,10,00': '⠎',
            '01,10,10,01': '⢎', '01,10,10,10': '⡎', '01,10,10,11': '⣎', '01,10,11,00': '⠮', '01,10,11,01': '⢮', '01,10,11,10': '⡮', '01,10,11,11': '⣮',
            '01,11,00,00': '⠚', '01,11,00,01': '⢚', '01,11,00,10': '⡚', '01,11,00,11': '⣚', '01,11,01,00': '⠺', '01,11,01,01': '⢺', '01,11,01,10': '⡺',
            '01,11,01,11': '⣺', '01,11,10,00': '⠞', '01,11,10,01': '⢞', '01,11,10,10': '⡞', '01,11,10,11': '⣞', '01,11,11,00': '⠾', '01,11,11,01': '⢾',
            '01,11,11,10': '⡾', '01,11,11,11': '⣾', '10,00,00,00': '⠁', '10,00,00,01': '⢁', '10,00,00,10': '⡁', '10,00,00,11': '⣁', '10,00,01,00': '⠡',
            '10,00,01,01': '⢡', '10,00,01,10': '⡡', '10,00,01,11': '⣡', '10,00,10,00': '⠅', '10,00,10,01': '⢅', '10,00,10,10': '⡅', '10,00,10,11': '⣅',
            '10,00,11,00': '⠥', '10,00,11,01': '⢥', '10,00,11,10': '⡥', '10,00,11,11': '⣥', '10,01,00,00': '⠑', '10,01,00,01': '⢑', '10,01,00,10': '⡑',
            '10,01,00,11': '⣑', '10,01,01,00': '⠱', '10,01,01,01': '⢱', '10,01,01,10': '⡱', '10,01,01,11': '⣱', '10,01,10,00': '⠕', '10,01,10,01': '⢕',
            '10,01,10,10': '⡕', '10,01,10,11': '⣕', '10,01,11,00': '⠵', '10,01,11,01': '⢵', '10,01,11,10': '⡵', '10,01,11,11': '⣵', '10,10,00,00': '⠃',
            '10,10,00,01': '⢃', '10,10,00,10': '⡃', '10,10,00,11': '⣃', '10,10,01,00': '⠣', '10,10,01,01': '⢣', '10,10,01,10': '⡣', '10,10,01,11': '⣣',
            '10,10,10,00': '⠇', '10,10,10,01': '⢇', '10,10,10,10': '⡇', '10,10,10,11': '⣇', '10,10,11,00': '⠧', '10,10,11,01': '⢧', '10,10,11,10': '⡧',
            '10,10,11,11': '⣧', '10,11,00,00': '⠓', '10,11,00,01': '⢓', '10,11,00,10': '⡓', '10,11,00,11': '⣓', '10,11,01,00': '⠳', '10,11,01,01': '⢳',
            '10,11,01,10': '⡳', '10,11,01,11': '⣳', '10,11,10,00': '⠗', '10,11,10,01': '⢗', '10,11,10,10': '⡗', '10,11,10,11': '⣗', '10,11,11,00': '⠷',
            '10,11,11,01': '⢷', '10,11,11,10': '⡷', '10,11,11,11': '⣷', '11,00,00,00': '⠉', '11,00,00,01': '⢉', '11,00,00,10': '⡉', '11,00,00,11': '⣉',
            '11,00,01,00': '⠩', '11,00,01,01': '⢩', '11,00,01,10': '⡩', '11,00,01,11': '⣩', '11,00,10,00': '⠍', '11,00,10,01': '⢍', '11,00,10,10': '⡍',
            '11,00,10,11': '⣍', '11,00,11,00': '⠭', '11,00,11,01': '⢭', '11,00,11,10': '⡭', '11,00,11,11': '⣭', '11,01,00,00': '⠙', '11,01,00,01': '⢙',
            '11,01,00,10': '⡙', '11,01,00,11': '⣙', '11,01,01,00': '⠹', '11,01,01,01': '⢹', '11,01,01,10': '⡹', '11,01,01,11': '⣹', '11,01,10,00': '⠝',
            '11,01,10,01': '⢝', '11,01,10,10': '⡝', '11,01,10,11': '⣝', '11,01,11,00': '⠽', '11,01,11,01': '⢽', '11,01,11,10': '⡽', '11,01,11,11': '⣽',
            '11,10,00,00': '⠋', '11,10,00,01': '⢋', '11,10,00,10': '⡋', '11,10,00,11': '⣋', '11,10,01,00': '⠫', '11,10,01,01': '⢫', '11,10,01,10': '⡫',
            '11,10,01,11': '⣫', '11,10,10,00': '⠏', '11,10,10,01': '⢏', '11,10,10,10': '⡏', '11,10,10,11': '⣏', '11,10,11,00': '⠯', '11,10,11,01': '⢯',
            '11,10,11,10': '⡯', '11,10,11,11': '⣯', '11,11,00,00': '⠛', '11,11,00,01': '⢛', '11,11,00,10': '⡛', '11,11,00,11': '⣛', '11,11,01,00': '⠻',
            '11,11,01,01': '⢻', '11,11,01,10': '⡻', '11,11,01,11': '⣻', '11,11,10,00': '⠟', '11,11,10,01': '⢟', '11,11,10,10': '⡟', '11,11,10,11': '⣟',
            '11,11,11,00': '⠿', '11,11,11,01': '⢿', '11,11,11,10': '⡿', '11,11,11,11': '⣿'
        }
        braille6_dict = {
            '00,00,00': '⠀', '00,00,01': '⠠', '00,00,10': '⠄', '00,00,11': '⠤',
            '00,01,00': '⠐', '00,01,01': '⠰', '00,01,10': '⠔', '00,01,11': '⠴',
            '00,10,00': '⠂', '00,10,01': '⠢', '00,10,10': '⠆', '00,10,11': '⠦', '00,11,00': '⠒', '00,11,01': '⠲',
            '00,11,10': '⠖', '00,11,11': '⠶',
            '01,00,00': '⠈', '01,00,01': '⠨', '01,00,10': '⠌', '01,00,11': '⠬', '01,01,00': '⠘', '01,01,01': '⠸',
            '01,01,10': '⠜', '01,01,11': '⠼',
            '01,10,00': '⠊', '01,10,01': '⠪', '01,10,10': '⠎', '01,10,11': '⠮', '01,11,00': '⠚', '01,11,01': '⠺',
            '01,11,10': '⠞', '01,11,11': '⠾',
            '10,00,00': '⠁', '10,00,01': '⠡', '10,00,10': '⠅', '10,00,11': '⠥', '10,01,00': '⠑', '10,01,01': '⠱',
            '10,01,10': '⠕', '10,01,11': '⠵',
            '10,10,00': '⠃', '10,10,01': '⠣', '10,10,10': '⠇', '10,10,11': '⠧', '10,11,00': '⠓', '10,11,01': '⠳',
            '10,11,10': '⠗', '10,11,11': '⠷',
            '11,00,00': '⠉', '11,00,01': '⠩', '11,00,10': '⠍', '11,00,11': '⠭', '11,01,00': '⠙', '11,01,01': '⠹',
            '11,01,10': '⠝', '11,01,11': '⠽',
            '11,10,00': '⠋', '11,10,01': '⠫', '11,10,10': '⠏', '11,10,11': '⠯', '11,11,00': '⠛', '11,11,01': '⠻',
            '11,11,10': '⠟', '11,11,11': '⠿'
        }

        densities = [
            r" .:-=+*#%@",
            r" ░▒▓█",
            r" .'` ^ ,:;Il!i><~+_-?][}{1)(|\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
        ]

        asciiString = asciiString.lower()
        if asciiString in {"braille", "braille8"}: braille = 1
        if asciiString in {"braille6"}: braille = 2
        elif asciiString == "squares": density = densities[1]
        elif asciiString == "simple": density = densities[0]
        elif asciiString in {"unihan", "chinese", "asian"}: density = unihanCharacters
        else: density = densities[2]
    if replace_space:
        braille_dict["00,00,00,00"] = '⠐'
        braille6_dict["00,00,00"] = '⠐'
        if braille == 0: density = density[1]+density[1:]
    img = iio.imread(image) if isinstance(image, str) else image
    h, w, hasAlpha = img.shape
    if hasAlpha != 4: img = np.dstack((img, np.full((h, w), 255, dtype=np.uint8)))
    if braille == 1: size_tuple = (size, round(h*size/w))
    else: size_tuple = (size, round(h*size/w))
    img = imageFilter(imageFilter(resizeImage(img, size=size_tuple), "identity", contrast=contrast, ditherType=dither_type,
                                  threshold=threshold), "greyscale", colors=16)
    if accentuate: img = imageKernel(img, "inverse_edge", thickness=thck, sharpness=1)
    if smooth: img = imageKernel(img, "box_blur", kernelSize=3)
    if invert: img = imageFilter(img, "inverse")
    newh, neww, _ = img.shape
    ascii_str = ""
    dens = len(density)-1 if not braille else None
    if braille == 1:
        for i in range(0, newh, 4):
            for j in range(0, neww, 2):
                brt = np.zeros((4,2))

                if j+1<neww:
                    brt[0][0], brt[0][1] = img[i][j][0], img[i][j+1][0]
                    if i+1<newh: brt[1][0], brt[1][1] = img[i+1][j][0], img[i+1][j+1][0]
                    if i+2<newh: brt[2][0], brt[2][1] = img[i+2][j][0], img[i+2][j+1][0]
                    if i+3<newh: brt[3][0], brt[3][1] = img[i+3][j][0], img[i+3][j+1][0]
                else:
                    brt[0][0] = img[i][j][0]
                    if i+1<newh: brt[1][0] = img[i+1][j][0]
                    if i+2<newh: brt[2][0] = img[i+2][j][0]
                    if i+3<newh: brt[3][0] = img[i+3][j][0]
                brt = np.round(brt/255).astype(np.uint8)
                txt = f"{brt[0][0]}{brt[0][1]},{brt[1][0]}{brt[1][1]},{brt[2][0]}{brt[2][1]},{brt[3][0]}{brt[3][1]}"
                ascii_str += braille_dict[txt]
            ascii_str += "\n"
    elif braille == 2:
        for i in range(0, newh, 3):
            for j in range(0, neww, 2):
                brt = np.zeros((4,2))

                if j+1<neww:
                    brt[0][0], brt[0][1] = img[i][j][0], img[i][j+1][0]
                    if i+1<newh: brt[1][0], brt[1][1] = img[i+1][j][0], img[i+1][j+1][0]
                    if i+2<newh: brt[2][0], brt[2][1] = img[i+2][j][0], img[i+2][j+1][0]
                else:
                    brt[0][0] = img[i][j][0]
                    if i+1<newh: brt[1][0] = img[i+1][j][0]
                    if i+2<newh: brt[2][0] = img[i+2][j][0]
                brt = np.round(brt/255).astype(np.uint8)
                txt = f"{brt[0][0]}{brt[0][1]},{brt[1][0]}{brt[1][1]},{brt[2][0]}{brt[2][1]}"
                ascii_str += braille6_dict[txt]
            ascii_str += "\n"
    else:
        for i in range(0, newh):
            for j in range(0, neww):
                brt = img[i][j][0]/255
                ascii_str += density[int(brt * dens)]
            ascii_str += "\n"
    if space_strip:
        ascii_str = '\n'.join([line.rstrip('⠀' if not replace_space or braille == 0 else '⠐') for line in ascii_str.splitlines()])
    return ascii_str, img


def profileGen(image: Union[str, np.ndarray], thickness: float = 1.0, color: Union[list, np.ndarray, None] = None,
               vignette: bool = True) -> np.ndarray:
    """
    Generates a profile picture with the given image, adding a colored circle around and an optional vignette

    :param image: Path to image or numpy array, must be a square image
    :param thickness: Outside circle thickness
    :param color: Outside circle color
    :param vignette: Set to True to add a black vignette
    :return: Modified image
    """
    if color is None: color = [255, 255, 255, 255]
    if isinstance(image, str): image = iio.imread(image)
    h, w, _ = image.shape
    if h != w:
        raise ValueError("Image width and height must match.")
        return None
    b = h // 1.5
    a1, b1, c1, d1 = 28.84416711, 0.0014620515, 2.51223573, -11.44728084
    c = h // 2 - thickness*(a1 * np.log(b1 * h + c1) + d1)
    if vignette: filtered = imageAddons(image, 'vignette', radius=b)
    else: filtered = image
    vignett = imageAddons(image, 'vignette', radius=c, onlyAddon=True, F=0, vignetteColor=color)
    filtered = imageInterp(filtered, vignett, 'alpha_compositing', interp1=1, interp2=1)

    center = h // 2, w // 2
    radius = h // 2 - 15
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    distance = 0.99 * np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    colors = np.zeros((h, w, 4), dtype=np.uint8)

    falloff = radius

    colors[..., 0] = (distance >= radius) * 0
    colors[..., 1] = (distance >= radius) * 0
    colors[..., 2] = (distance >= radius) * 0
    colors[..., 3] = (distance >= radius) * np.clip((255 * (1 - falloff)), 0, 255)

    filtered[distance >= radius] = colors[distance >= radius]
    return filtered.astype(np.uint8)


def remover(image: Union[str, np.ndarray], colorRm: Union[list, np.ndarray, None] = None,
            colorRp: Union[list, np.ndarray, None] = None, threshold: int = 0,
            rmType: str = 'remove') -> np.ndarray:
    """
    Replaces all pixels of a given color or similar to another color or erases them

    :param image: Path to image or numpy array
    :param colorRm: Color to remove/replace
    :param colorRp: Substitute color
    :param threshold: Maximum tolerable difference between colors
    :param rmType: Whether to remove or replace the given color
    :return: Image with the removed/replaced color
    """
    if colorRm is None: colorRm = [255, 255, 255, 255]
    if colorRp is None: colorRp = [127, 127, 127, 255]
    img = iio.imread(image) if isinstance(image, str) else image
    h, w, hasAlpha = img.shape
    if hasAlpha != 4: img = np.dstack((img, np.full((h, w), 255, dtype=np.uint8)))
    rmType = rmType.lower()
    if rmType == 'remove':
        diff = np.abs(img - colorRm)
        color_diff = np.sum(diff, axis=2)
        selected_pixels = np.where(color_diff <= threshold)
        img[selected_pixels[0], selected_pixels[1], 3] = 0
    elif rmType == 'replace':
        diff = np.abs(img - colorRm)
        color_diff = np.sum(diff, axis=2)
        selected_pixels = np.where(color_diff <= threshold)
        img[selected_pixels[0], selected_pixels[1]] = colorRp
    return img


def magicWand(img: Union[str, np.ndarray], selectedPixel: tuple[int, int],
              replaceColor: tuple[int, int, int, int] = (255, 0, 0, 255), tolerance: int = 20) -> np.ndarray:
    """
    Given a pixel, selects all contiguous similar pixels from it (floodfill)

    :param img: Path to image or numpy array
    :param selectedPixel: Start pixel
    :param replaceColor: Replaces the selected region with this color
    :param tolerance: Maximum tolerable difference between colors
    :return: Modified image
    """
    temp = False
    if isinstance(img, str): image2 = Image.open(img)
    else:
        Image.fromarray(img).save("temp_png.png")
        image2 = Image.open("temp_png.png")
        temp = True
    floodfill(image2, selectedPixel, replaceColor, thresh=tolerance)
    if temp: os.remove("temp_png.png")
    return np.array(image2).astype(np.uint8)


def paint(img: np.ndarray, selectedPixel: tuple, brushKernel: np.ndarray, color: Union[list, np.ndarray, None] = None,
          paintMode: str = "alpha_compositing") -> np.ndarray:
    """
    Function not finished!
    """
    if color is None: color = [0, 0, 0, 255]
    color = np.array(color)
    N = len(brushKernel[0])
    s0, s1 = selectedPixel
    r2, g2, b2, a2 = list(color)
    if paintMode == "replace":
        for i in range(N):
            for j in range(N):
                if brushKernel[i, j] != 0:
                    img[s0-i-N//2, s1-j-N//2] = np.array([r2*brushKernel[i, j], g2*brushKernel[i, j], b2*brushKernel[i, j], a2]).astype(np.uint8)
    elif paintMode == "alpha_compositing":
        for i in range(N):
            for j in range(N):
                if brushKernel[i, j] != 0:
                    r1, g1, b1, a1 = list(img[s0-i-N//2, s1-j-N//2])
                    a2 = int(color[3]*brushKernel[i,j])
                    v = 255-a2
                    img[s0-i-N//2, s1-j-N//2] =[(r1*v+r2*a2)//255, (g1*v+g2*a2)//255, (b1*v+b2*a2)//255, a1]
    elif paintMode in ["add", "addition"]:
        for i in range(N):
            for j in range(N):
                if brushKernel[i, j] != 0:
                    img[s0-i-N//2, s1-j-N//2] = np.clip(img[s0-i-N//2, s1-j-N//2]+color, 0, 255)
    elif paintMode == "substract":
        for i in range(N):
            for j in range(N):
                if brushKernel[i, j] != 0:
                    img[s0-i-N//2, s1-j-N//2] = np.clip(img[s0-i-N//2, s1-j-N//2]-color, 0, 255)
    elif paintMode == "multiply":
        for i in range(N):
            for j in range(N):
                if brushKernel[i, j] != 0:
                    img[s0-i-N//2, s1-j-N//2] = np.clip(img[s0-i-N//2, s1-j-N//2]*color, 0, 255)
    elif paintMode in ["average", "avg", "mean"]:
        for i in range(N):
            for j in range(N):
                if brushKernel[i, j] != 0:
                    img[s0-i-N//2, s1-j-N//2] = np.clip((img[s0-i-N//2, s1-j-N//2]+color)//2, 0, 255)
    elif paintMode == "erase":
        for i in range(N):
            for j in range(N):
                if brushKernel[i, j] != 0:
                    img[s0-i-N//2, s1-j-N//2, 3] *= int(255*brushKernel[i, j])
    else:
        for i in range(N):
            for j in range(N):
                if brushKernel[i, j] != 0:
                    r1, g1, b1, a1 = img[s0-i-N//2, s1-j-N//2]
                    img[s0-i-N//2, s1-j-N//2] = np.array([(1-a2) * r1 + a2 * r2, (1-a2) * g1 + a2 * g2, (1-a2) * b1 + a2 * b2, a1 + a2 * (1-a1)])*brushKernel[i, j]
    return img


@jit(nopython=True)
def jit_dither_floyd(pixel: np.ndarray, h: int, w: int, palette: np.ndarray) -> np.ndarray:
    for i in range(1, h+1):
        for j in range(1, w+1):
            oldpx = np.copy(pixel[i, j])
            min_dist = np.inf
            min_index = 0
            for idx, color in enumerate(palette):
                dist = np.sqrt(np.sum((color - oldpx) ** 2))
                if dist < min_dist:
                    min_dist = dist
                    min_index = idx
            newpx = palette[min_index]
            pixel[i, j] = newpx
            error = oldpx - newpx
            pixel[i, j + 1] = np.clip(pixel[i, j + 1] + error * (7 / 16), 0, 255)
            pixel[i + 1, j - 1] = np.clip(pixel[i + 1, j - 1] + error * (3 / 16), 0, 255)
            pixel[i + 1, j] = np.clip(pixel[i + 1, j] + error * (5 / 16), 0, 255)
            pixel[i + 1, j + 1] = np.clip(pixel[i + 1, j + 1] + error * (1 / 16), 0, 255)
    return pixel


@jit(nopython=True)
def jit_dither_atkinson(pixel: np.ndarray, h: int, w: int, palette: np.ndarray) -> np.ndarray:
    for i in range(2, h+1):
        for j in range(2, w+1):
            oldpx = np.copy(pixel[i, j])
            min_dist = np.inf
            min_index = 0
            for idx, color in enumerate(palette):
                dist = np.sqrt(np.sum((color - oldpx) ** 2))
                if dist < min_dist:
                    min_dist = dist
                    min_index = idx
            newpx = palette[min_index]
            pixel[i, j] = newpx
            error = oldpx - newpx
            pixel[i, j + 1] = np.clip(pixel[i, j + 1] + error * (1/8), 0, 255)
            pixel[i, j + 2] = np.clip(pixel[i, j + 2] + error * (1/8), 0, 255)
            pixel[i + 1, j - 2] = np.clip(pixel[i + 1, j - 2] + error * (1/8), 0, 255)
            pixel[i + 1, j - 1] = np.clip(pixel[i + 1, j - 1] + error * (1/8), 0, 255)
            pixel[i + 1, j] = np.clip(pixel[i + 1, j] + error * (1/8), 0, 255)
            pixel[i + 1, j + 1] = np.clip(pixel[i + 1, j + 1] + error * (1/8), 0, 255)
            pixel[i + 2, j] = np.clip(pixel[i + 2, j] + error * (1/8), 0, 255)
    return pixel


@jit(nopython=True)
def jit_dither_stucki(pixel: np.ndarray, h: int, w: int, palette: np.ndarray) -> np.ndarray:
    for i in range(2, h+1):
        for j in range(2, w+1):
            oldpx = np.copy(pixel[i, j])
            min_dist = np.inf
            min_index = 0
            for idx, color in enumerate(palette):
                dist = np.sqrt(np.sum((color - oldpx) ** 2))
                if dist < min_dist:
                    min_dist = dist
                    min_index = idx
            newpx = palette[min_index]
            pixel[i, j] = newpx
            error = oldpx - newpx
            pixel[i, j + 1] = np.clip(pixel[i, j + 1] + error * (8 / 42), 0, 255)
            pixel[i, j + 2] = np.clip(pixel[i, j + 2] + error * (4 / 42), 0, 255)
            pixel[i + 1, j - 2] = np.clip(pixel[i + 1, j - 2] + error * (2 / 42), 0, 255)
            pixel[i + 1, j - 1] = np.clip(pixel[i + 1, j - 1] + error * (4 / 42), 0, 255)
            pixel[i + 1, j] = np.clip(pixel[i + 1, j] + error * (8 / 42), 0, 255)
            pixel[i + 1, j + 1] = np.clip(pixel[i + 1, j + 1] + error * (4 / 42), 0, 255)
            pixel[i + 1, j + 2] = np.clip(pixel[i + 1, j + 2] + error * (2 / 42), 0, 255)
            pixel[i + 2, j - 2] = np.clip(pixel[i + 2, j - 2] + error * (1 / 42), 0, 255)
            pixel[i + 2, j - 1] = np.clip(pixel[i + 2, j - 1] + error * (2 / 42), 0, 255)
            pixel[i + 2, j] = np.clip(pixel[i + 2, j] + error * (4 / 42), 0, 255)
            pixel[i + 2, j + 1] = np.clip(pixel[i + 2, j + 1] + error * (2 / 42), 0, 255)
            pixel[i + 2, j + 2] = np.clip(pixel[i + 2, j + 2] + error * (1 / 42), 0, 255)
    return pixel


def ditherImage(image: Union[str, np.ndarray], colors: int = 6, rsz: float = 0.5,
                palette: Union[np.ndarray, list[list]] = None, type: str = "floyd-steinberg") -> np.ndarray:
    """
    Applies dithering to an RGB or RGBA image. Uses numba to improve performance.

    Available types: floyd-steinberg, atkinson, stucki

    :param image: Path to image or numpy array
    :param colors: Numbers of different colors to grab from the image
    :param rsz: How much to scale the image when searching for colors (might improve performance for big images)
    :param palette: Palette of colors to use instead of grabbing colors from the image
    :param type: Type of dithering
    :return: Dithered image
    """
    image = iio.imread(image) if isinstance(image, str) else image
    h, w, hasAlpha = image.shape
    type=type.lower()
    if hasAlpha != 4: image = np.dstack((image, np.full((h, w), 255, dtype=np.uint8)))
    try: palette = palette.tolist()
    except: pass
    N = 2 if type in ["atkinson", "stucki"] else 1
    image = extendImage(image, N)
    plt = np.array(findPalette(resizeImage(image, scale=rsz), colors)) if not palette else np.array(palette)
    if type == "atkinson":
        img = jit_dither_atkinson(image.astype(float), h=h, w=w, palette=plt)
    elif type == "stucki":
        img = jit_dither_stucki(image.astype(float), h=h, w=w, palette=plt)
    else:
        img = jit_dither_floyd(image.astype(float), h=h, w=w, palette=plt)
    img = img.astype(np.uint8)
    newh, neww, _ = img.shape
    top, bottom, left, right = N, newh - N, N, neww - N
    return img[top:bottom, left:right]


def clipBrightness(image: np.ndarray, threshold: Union[tuple, list]) -> np.ndarray:
    """
    Function not finished! For now it does what clipRGB does

    Clips the brightness of an image.

    :param image: RGB or RGBA Image as a numpy array
    :param threshold: Lower and upper bounds for the brightness
    :return: Clipped image
    """
    image[..., :3] = np.clip(image, threshold[0], threshold[1])[..., :3]
    return image


def clipRGB(image: np.ndarray, threshold: Union[tuple, list]) -> np.ndarray:
    """
    Clips the RGB values of an image.

    :param image: RGB or RGBA Image as a numpy array
    :param threshold: Lower and upper bounds
    :return: Clipped image
    """
    image[..., :3] = np.clip(image, threshold[0], threshold[1])[..., :3]
    return image


def hexToRgb(hex_color: int) -> list[int]:
    return [(hex_color >> 16) & 0xFF, (hex_color >> 8) & 0xFF, hex_color & 0xFF, 255]


def hexToRgba(hex_color: int) -> list[int]:
    return [(hex_color >> 24) & 0xFF, (hex_color >> 16) & 0xFF, (hex_color >> 8) & 0xFF, hex_color & 0xFF]


def rgbToHex(rgb: Union[list[int], tuple[int], np.ndarray]) -> int:
    return (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]


def rgbaToHex(rgba: Union[list[int], tuple[int], np.ndarray]) -> int:
    return (rgba[0] << 24) + (rgba[1] << 16) + (rgba[2] << 8) + rgba[3]

