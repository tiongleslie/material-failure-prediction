from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.io as sio
import numpy as np
from PIL import Image
from DML import DML


def read_image_as_tensor(filename):
    """
    This function converts JPG image to numpy matrix format
    :param filename: image path
    :return: 2D numpy matrix with normalization scaled
    """
    data_list = []
    img = Image.open(filename)
    img.thumbnail((150, 150), Image.AFFINE)
    arr = np.array(img).reshape((150, 150, 3))
    data_list.append(arr)
    img.close()

    data_X = np.array(data_list)

    return data_X / 255.


if __name__ == '__main__':
    print("-----------------------------------")
    x1 = read_image_as_tensor('DML-Inputs/x1.jpg')
    x2 = np.asarray(sio.loadmat('DML-Inputs/x2.mat')['x2'])

    model = DML()
    result = model.DML_model(path='pretrain', x_1=x1, x_2=x2)

    print("Fatigue fracture progress: {0:.2f}%".format(result))
    print("-----------------------------------")
