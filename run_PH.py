from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imageio
import numpy as np
import homcloud.interface as hc
import scipy.io as sio
import os
from CT_utils import CT_utils
from CT_utils import ImageTDA
from sklearn.preprocessing import MinMaxScaler


def crop_convert(folder, filename='', range_begin=1, range_end=1001, thres=128):
    """
    This function converts X-C images to binary format.

    :param folder: image folder
    :param filename: filename
    :param range_begin: the index number of starting image
    :param range_end: the index number of ending image
    :param thres: the threshold value for conversion
    """
    result_folder = 'Result/'+folder
    dataset_folder = 'Dataset/'+folder

    if os.path.isdir(result_folder) is False:
        os.mkdir(result_folder)
        os.mkdir(result_folder + '/CT Binary')
        os.mkdir(result_folder + '/CT Binary(NOBG)')
        os.mkdir(result_folder + '/PH Result')

    ct_object = CT_utils(ct_path=dataset_folder, result_path=result_folder, filename=filename, thres=thres)
    ct_object.image_process(range_begin=range_begin, range_end=range_end)


def TDA_analysis(folder, range_begin=1, range_end=126):
    """
    This functions calculates the void topology by using HomCloud API.

    :param folder: Sample folder path
    :param range_begin: starting range of input images
    :param range_end: ending range of input images
    """
    image_folder = 'Result/' + folder
    pict = np.stack([imageio.imread(image_folder + "/CT Binary(NOBG)/pict_{:04d}.jpg".format(n)) > 128
                     for n in range(range_begin, range_end)], axis=0)

    result_folder = 'Result/' + folder + '/PH Result'

    print('\n\nCalculating by Homcloud library...')
    hc.PDList.from_bitmap_levelset(hc.distance_transform(pict, signed=True),
                                   save_to=result_folder + '/PD.idiagram')

    hc.BitmapPHTreesPair.from_bitmap_levelset(hc.distance_transform(pict, signed=True),
                                              save_to=result_folder + '/PD.p2mt')
    print('Completed!\n\n')

    print('PH Analysis begins~')
    pdlist = hc.PDList(result_folder + '/PD.idiagram')
    pd1 = pdlist.dth_diagram(0)

    vmax = 50
    vmin = -6

    pd1.histogram((vmin - 2, vmax), 50).plot(colorbar={"type": "log"})

    phtreespair = hc.BitmapPHTreesPair(result_folder + '/PD.p2mt')
    phtree_0 = phtreespair.dim_0_trees()

    numbers_count = []
    bir = [-9, -8, -7, -6, -5, -4, -3, -2, -1]

    for k in range(len(bir)):
        for i in range(-20, 61):
            x = bir[k]
            y = i
            nodes_0 = phtree_0.pair_nodes_in_rectangle(x, x, y, y)
            if len(nodes_0) != 0:
                numbers_count.append([x, y, len(nodes_0)])

    result = np.array(numbers_count)
    min_b = np.min(result[:, [0]])
    max_d = np.max(result[:, [1]])
    numbers_count.append([min_b, max_d, 1.0])

    sio.savemat(result_folder + '/bin_count.mat', {'bin_count': np.array(numbers_count)})


def calc_PHmetric(filename='', vol=0.):
    """
    This function calculates the proposed PD metric.

    :param filename: Persistent Diagram metric
    :param vol: Total volume of input space
    :return: PD metric
    """
    count = sio.loadmat(filename)['bin_count']
    count = np.asarray(count)

    size = count.shape[0]
    xnew = np.unique(count[:, 0])

    bin_birth = []
    for i in range(xnew.shape[0]):
        temp = 0
        for j in range(size):
            if xnew[i] == count[j, 0]:
                temp = temp + count[j, 2]
        bin_birth.append(temp)

    xnew = abs(xnew) * 3
    xnew = np.flipud(xnew)
    bin_birth = np.flipud(bin_birth)

    metric = []
    total_vol_with_3 = 0
    total_vol_with_3_6 = 0
    total_vol_with_6 = 0
    for i in range(xnew.shape[0]):
        if xnew[i] <= 3:
            total_vol_with_3 += bin_birth[i]
        elif 3 < xnew[i] <= 6:
            total_vol_with_3_6 += bin_birth[i]
        elif xnew[i] > 6:
            total_vol_with_6 += bin_birth[i]

    metric.append(total_vol_with_3 / vol)
    metric.append(total_vol_with_3_6 / vol)
    metric.append(total_vol_with_6 / vol)

    sum_v = np.sum(count[:, 2])

    d_ = np.zeros((count.shape[0], 2))
    for i in range(count.shape[0]):
        d_[i, 0] = abs(count[i, 1] - count[i, 0])
        d_[i, 1] = count[i, 2]

    d = np.zeros((count.shape[0], 2))
    for i in range(count.shape[0]):
        d[i, 0] = d_[i, 0] * 3 - 1.5
        d[i, 1] = np.power(d[i, 0], count[i, 2]/sum_v)

    D = np.prod(d[:, 1])
    metric.append(D)

    v = np.zeros((count.shape[0], 2))
    for i in range(count.shape[0]):
        v[i, 0] = abs((d_[i, 0] * 3 - 1.5) - D)
        v[i, 1] = np.power(v[i, 0], count[i, 2] / sum_v)

    V = np.prod(v[:, 1])
    metric.append(V)

    return np.asarray(metric)


def normalize_metric(metric):
    """
    This function normalizes the PD metric by using Min-Max scaling.

    :param metric: PD metric
    :return: normalized PD metric
    """
    scaler = MinMaxScaler()
    sample_scaled_feat = [[89], [18628]]
    scaler.fit_transform(sample_scaled_feat)
    metric[0, 0] = scaler.transform(metric[0, 0].reshape(1, -1))

    sample_scaled_feat = [[0], [39]]
    scaler.fit_transform(sample_scaled_feat)
    metric[0, 1] = scaler.transform(metric[0, 1].reshape(1, -1))

    sample_scaled_feat = [[0], [95]]
    scaler.fit_transform(sample_scaled_feat)
    metric[0, 2] = scaler.transform(metric[0, 2].reshape(1, -1))

    sample_scaled_feat = [[9.453], [93.71]]
    scaler.fit_transform(sample_scaled_feat)
    metric[0, 3] = scaler.transform(metric[0, 3].reshape(1, -1))

    sample_scaled_feat = [[2.23], [19.59]]
    scaler.fit_transform(sample_scaled_feat)
    metric[0, 4] = scaler.transform(metric[0, 4].reshape(1, -1))

    return metric


if __name__ == '__main__':

    # Calculate Void topology
    print("-----------------------------------")
    folder = 'Sample'
    crop_convert(folder, filename='#20210311_#3-6_Fatigue', range_begin=309, range_end=809, thres=128)
    TDA_analysis(folder, range_begin=309, range_end=809)

    # Calculate PD Metric
    print("-----------------------------------")
    mat_file = 'Result/Sample/PH Result/bin_count.mat'
    vol = (3.000 * 500 / 1000) * 1.412 * 0.406
    metric = calc_PHmetric(filename=mat_file, vol=vol)
    print("PD Metric is created!\n")
    metric = np.array(metric).reshape((1, 5))
    metric = normalize_metric(metric)
    sio.savemat('DML-Inputs/x2.mat', {'x2': metric})

    # Generate PD Image
    print("-----------------------------------")
    pd_dim = 0
    R = [-10, 66]
    R_x = [-10, 2]
    R_y = [-10, 66]
    ph = ImageTDA(path="Result/Sample/PH Result/PD.idiagram", pd_dim=pd_dim)
    ph.plot_clean_pd_for_ml(pd_range=R, x_show=R_x, y_show=R_y)
    print("PD Image is created!\n\n")

    print("Done!")
    print("-----------------------------------")
