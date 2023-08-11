from PIL import Image
import numpy as np
from itertools import product
import os

# name_arr = ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"]
name_arr = ["C1", "C2", "D1", "D2"]

w = 10800
h = 10800
d = 253
img_arr = []
freq = 0
count = 0
for filename in name_arr:
    print(filename)
    grid = product(range(0, h - h % d, d), range(0, w - w % d, d))

    for i, j in grid:
        print(i, j)
        box = (j, i, j + d, i + d)
        name = 'reduced2/' + filename + '_' + str(i) + '_' + str(j) + '.png'
        img = Image.open(name)
        img_arr = np.array(img)
        img_arr = img_arr.flatten()
        freq = np.bincount(img_arr)  # counts the occurrence of each element, find max index
        freq = np.argmax(freq)
        count = np.bincount(img_arr)
        count = count[freq]
        rel = count / img_arr.size
        if rel >= 0.2:
            os.remove(name)
