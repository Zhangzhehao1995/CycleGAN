from __future__ import print_function

import SimpleITK as sitk
import numpy as np
import sys

def read_image(name, verbose=False):
    if(verbose):
        print('Reading image: ' + name)
    try:
        img = sitk.ReadImage(name)
    except:
        print("Error:", sys.exc_info()[1])
        exit()

    # fix NIFTI direction issue
    # img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    #img = sitk.GetArrayFromImage(img)
    # flip slices
    #img = img, axis=1)
    return img

def write_image(img, file_name, use_compression=True, verbose=False):
    if verbose:
        print('Writing image: ' + file_name)
    try:
        sitk.WriteImage(img, file_name)
    except:
        print("Error:", sys.exc_info()[1])
