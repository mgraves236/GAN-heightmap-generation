from PIL import Image
import numpy as np
from itertools import product
import os

w = 21600
h = 10800
d = 256
img_arr = []
freq = 0
count = 0
grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
for i, j in grid:
    box = (j, i, j + d, i + d)
    name = 'cropped2/map_' + str(i) + '_' + str(j) + '.png'
    img = Image.open(name)
    img_arr = np.array(img)
    img_arr = img_arr.flatten()
    freq = np.bincount(img_arr).argmax()
    freq = np.argmax(freq)
    count = np.bincount(img_arr)
    count = count[freq]
    rel = count / img_arr.size
    if rel >= 0.4:
        os.remove(name)
