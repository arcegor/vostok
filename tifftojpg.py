import os
import re

import cv2 as cv
import argparse

import numpy as np
import tifffile as tiff
from tifffile import TiffFileError

parser = argparse.ArgumentParser(description='Great Description To Be Here')

parser.add_argument("-s",
                    "--source_path",
                    type=str)
parser.add_argument("-o",
                    "--out_path",
                    type=str)


def parseDirs(src, dst, testing=False):
    dirs = os.listdir(src)
    for dir in dirs:
        if os.path.isdir(src + '/' + dir):
            imgs = os.listdir(src + '/' + dir)
        else:
            imgs = dir.split()

        for j in imgs:
            res = re.findall('mask', j)
            l = len(res)
            if len(re.findall('cross', j)) > 0:
                rel_dir = '/cross'
            else:
                rel_dir = '/long'

            temp = convert(src + '/' + j)
            if temp is None:
                continue
            if testing:
                rel_dir = '/'
                dst = src
                for index, item in enumerate(temp):
                    path = dst + rel_dir + dir + '_' + re.sub('.tif', '', j) + '_' + str(index) + '.jpg'
                    cv.imwrite(path, item)
                continue
            if l == 0:
                for index, item in enumerate(temp):
                    path = dst + rel_dir + '/images/' + dir + '_' + re.sub('.tif', '', j) + '_' + str(index) + '.jpg'
                    cv.imwrite(path, item)
            else:
                for index, item in enumerate(temp):
                    item = cv.convertScaleAbs(item, alpha=255.0)
                    path = dst + rel_dir + '/masks/' + dir + '_' + re.sub('.tif', '', j) + '_' + str(index) + '_' \
                           + 'mask' + '.jpg'
                    cv.imwrite(path, item)


def convert(img):
    tmp = []
    try:
        temp = tiff.imread(img)
        # size = np.shape(temp)
        # if len(size) == 3:
        #     np.reshape(temp, (size[0], size[1], size[2], 3))
        for img in temp:
            size = np.shape(img)
            if len(size) == 2:
                img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
            tmp.append(img)
        return tmp
    except TiffFileError:
        print(img)


def main():
    parseDirs(r"c:\users\yvego\downloads\data", 'dataset')


if __name__ == '__main__':
    # args = parser.parse_args()
    main()
