import os
from PIL import Image
from skimage.io import imread, imsave
from itertools import product
import numpy as np

Image.MAX_IMAGE_PIXELS = None

name, ext = os.path.splitext('map.png')
img = Image.open('map.png')
w, h = img.size
d = 256 # tile size

grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
for i, j in grid:
    box = (j, i, j + d, i + d)
    out = os.path.join('cropped', f'{name}_{i}_{j}{ext}')
    img.crop(box).save(out)