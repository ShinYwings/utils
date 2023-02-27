import os
from datetime import datetime as dt
import numpy as np
from scipy.interpolate import interp1d
import OpenEXR
import Imath
import cv2
from enum import Enum
import matplotlib.pyplot as plt

CURR_PATH_PREFIX = os.getcwd()

class StrEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

def static_var(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_var(timestamp = dt.now().strftime("%Y-%m-%d-%H:%M:%S"))
def getTimestamp():
    return getTimestamp.timestamp

# If there is not input name, create the directory name with timestamp
def createNewDir(root_path, name=None):

    if name == None:
        print("[utils.py, createNewDir()] DirName is not defined in the arguments, define as timestamp")
        newpath = os.path.join(root_path, getTimestamp())
    else:
        newpath = os.path.join(root_path, name)

    """Create parent path if it doesn't exist"""
    if not os.path.isdir(newpath):
        os.mkdir(newpath)
    return newpath

def createTrainValidationDirpath(root_dir, createDir = False):
    
    if createDir == True:
        train_dir = createNewDir(root_dir, "train")
        val_dir = createNewDir(root_dir, "val")
    
    else:
        train_dir = os.path.join(root_dir, "train")
        val_dir = os.path.join(root_dir, "val") 

    return train_dir, val_dir

def writeHDR(arr, outfilename, imgshape):

    ext_name = outfilename.split(".")[1]
    if ext_name == "exr":
        '''Align_ratio (From HDRUNET)''' 
        # align_ratio = (2 ** 16 - 1) / arr.max()
        # arr = np.round(arr * align_ratio).astype(np.uint16)
        
        '''write HDR image using OpenEXR'''
        # Convert to strings
        R, G, B = [x.astype('float16').tostring() for x in [arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]]]

        im_height, im_width = imgshape

        HEADER = OpenEXR.Header(im_width, im_height)
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])

        out = OpenEXR.OutputFile(outfilename, HEADER)
        out.writePixels({'R': R, 'G': G, 'B': B})
        out.close()

    if ext_name == "hdr":
        cv2.imwrite(outfilename, arr.copy())


def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    filters = np.transpose(filters, [0, 3, 1, 2])
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], interpolation='nearest')
    plt.show()