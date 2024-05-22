from ctypes import *
from PIL import Image
from typing import Union
import numpy as np


def CDither(image: Union[str, np.ndarray], palette: np.ndarray, type: str = "",
            bayerSize: int = 2, bayerStrength: float = 1.0):
    if bayerSize > 5: return
    bayerSize = 2**bayerSize
    if type in {"ordered", "bayer", "halftone"}: type = "halftone"
    image = Image.open(image) if isinstance(image, str) else Image.fromarray(image)
    plt_size, _ = palette.shape
    cut_palette = palette[:, :3]
    dither_code = CDLL(f'dll/{type}.dll')
    dither_code.dither.argtypes = [c_char_p, c_int, c_int, c_int, ((c_uint8 * 3) * plt_size), c_int, c_int, c_float]
    dither_code.dither.restype = POINTER(c_char_p)
    w, h = image.size
    byte_data = image.tobytes()
    if len(byte_data) != w*h*4:
        new_img = np.dstack((image, np.full((h, w), 255, dtype=np.uint8)))
        byte_data = new_img.tobytes()
    c_palette = ((c_uint8 * 3) * plt_size)(*[
        (c_uint8 * 3)(*rgba) for rgba in cut_palette
    ])
    byte_out = dither_code.dither(byte_data, w, h, 4, c_palette, plt_size, bayerSize, bayerStrength)
    return np.array(Image.frombytes('RGBA', (w,h), cast(byte_out, POINTER(c_char_p * (w * h * 4))).contents))


def COutline(image: Union[str, np.ndarray], thickness: int = 15, outlineColor: Union[np.ndarray, list] = [0, 0, 0, 255],
                 threshold: int = 200, outlineType: str = "circle"):
    try: outlineColor.tolist()
    except: pass
    image = Image.open(image) if isinstance(image, str) else Image.fromarray(image)
    outline_code = CDLL(f'dll/outline.dll')
    outline_code.dither.argtypes = [c_char_p, c_int, c_int, c_int, c_uint8, (c_int16 * 4), c_int, c_char]
    outline_code.dither.restype = POINTER(c_char_p)
    w, h = image.size
    byte_data = image.tobytes()
    if len(byte_data) != w*h*4:
        new_img = np.dstack((image, np.full((h, w), 255, dtype=np.uint8)))
        byte_data = new_img.tobytes()
    c_outcol = (c_int16 * 4)(*outlineColor)
    c_outtype = bytes('c'.encode())
    if outlineType in {'square', 'box'}: c_outtype = bytes('q'.encode())
    elif outlineType in {'star'}: c_outtype = bytes('s'.encode())
    byte_out = outline_code.dither(byte_data, w, h, 4, threshold, c_outcol, thickness, c_outtype)
    return np.array(Image.frombytes('RGBA', (w,h), cast(byte_out, POINTER(c_char_p * (w * h * 4))).contents))

