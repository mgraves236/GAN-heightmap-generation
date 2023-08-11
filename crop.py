import os
from PIL import Image
from skimage.io import imread, imsave
from itertools import product
import numpy as np

Image.MAX_IMAGE_PIXELS = None

name, ext = os.path.splitext('A1.png')
img = Image.open('A1.png')
w, h = img.size
print(w, h)
d = 253 # tile size

grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
for i, j in grid:
    box = (j, i, j + d, i + d)
    out = os.path.join('croppedA1', f'{name}_{i}_{j}{ext}')
    img.crop(box).save(out)