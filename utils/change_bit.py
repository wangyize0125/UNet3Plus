#! python3
# encoding: utf-8
"""
@author:    Yize Wang
@contact:    wangyize@hust.edu.cn
@file:    change_bit.py
@time:   2022/6/2 9:25
@description:    
"""

from PIL import Image
import numpy as np

filenames = ["resized_246", "resized_267", "resized_280"]

# file = Image.open("../data/train/masks/resized_061.png")
# file = file.convert("L")
# file.save("../data/train/masks/resized_061.png")

for filename in filenames:
    file = Image.open("../data/train/masks/" + filename + ".png")
    file = file.convert("L")
    file.save("../data/train/masks/" + filename + ".png")
