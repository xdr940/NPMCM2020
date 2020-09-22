#move files
from path import Path
import re
from utils.official import readlines
import os
from tqdm import  tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random


def frames2timestamp(frame_idx):
    m = 0
    s = 0
    if frame_idx < 471:
        m = int(frame_idx / 10)

        s = (frame_idx % 10 - 1) * 6 + 26

    if frame_idx > 500:
        m = 64
        s = (frame_idx - 500) * 6 + 11

    mm = int((m * 60 + s) / 60)
    timestamp = int(mm) + 960
    return timestamp


if __name__ == '__main__':

    ret = []
    for i in range(10,470):
        ret.append("color {:05d} {}".format(i,frames2timestamp(i)))
    for i in range(500,4600):
        ret.append("color {:05d} {}".format(i,frames2timestamp(i)))



    random.shuffle(ret)
    for item in ret:
        print(item)


