# Library dealing with the reading, manipulation and display of images.

import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

def show_image(img:Image=None) -> None:
    """Primitive wrapper.
    TODO: Expand so tensor, pil, ndarray etc can be displayed without difficulty!

    Args:
        img (Image, optional): _description_. Defaults to None.
    """
    if(img):
        plt.imshow(img, cmap='gray')