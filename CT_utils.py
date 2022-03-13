from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import imageio
import numpy as np
import homcloud.interface as hc
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class CT_utils(object):
    def __init__(self, ct_path, result_path, filename, thres=128):
        """
        Construstor

        :param ct_path: X-CT image path
        :param result_path: Result path
        :param filename: filename of image path
        """
        self.ct_path = ct_path
        self.result_path = result_path
        self.filename = filename
        self.thres = thres
        print('Creating CT...')

    def image_process(self, range_begin=1, range_end=1003):
        """
        This function processes the binary conversion.

        :param range_begin: the index number of starting image
        :param range_end: the index number of ending image
        :return: a binary format of numpy matrix
        """
        for i in range(range_begin, range_end):
            img = Image.open(self.ct_path + "/" + self.filename + "{:04d}.tiff".format(i))

            bw = np.asarray(img).copy()
            bw = np.true_divide(bw, [255.0], out=None)
            bw[bw >= self.thres] = 255
            bw[bw < self.thres] = 0
            bwfile = Image.fromarray(255-np.uint8(bw))
            bwfile.save(self.result_path + "/CT Binary/pict_{:04d}.jpg".format(i))

            no_bw = bw.copy()

            bwfile = Image.fromarray(np.uint8(no_bw))
            bwfile.save(self.result_path + "/CT Binary(NOBG)/pict_{:04d}.jpg".format(i))

        return np.stack([imageio.imread(self.result_path + "/CT Binary(NOBG)/pict_{:04d}.jpg".format(n)) > 128
                         for n in range(range_begin, range_end)], axis=0)

    @staticmethod
    def remove_bg(bw):
        """
        This function converts the background as 'black' and the voids as 'white'.

        :param bw: the input of image
        :return: the result of conversion
        """
        tempbw = bw.copy()
        result = np.zeros((bw.shape[0], bw.shape[1]))

        for i in range(bw.shape[1]):
            index_store_white = []
            if np.max(bw[:, i]) == 0:
                bw[:, i] = 255.
            else:
                for j in range(bw.shape[0]):
                    if bw[j, i] == 255:
                        index_store_white.append(j)

                for j in range(bw.shape[0]):
                    if not index_store_white:
                        bw[j, i] = 255.
                    else:
                        if j < index_store_white[0] or j > index_store_white[-1]:
                            bw[j, i] = 255.

        for i in range(tempbw.shape[0]):
            index_store_white = []
            if np.max(tempbw[i, :]) == 0:
                tempbw[i, :] = 255.
            else:
                for j in range(tempbw.shape[1]):
                    if tempbw[i, j] == 255:
                        index_store_white.append(j)

                for j in range(tempbw.shape[1]):
                    if not index_store_white:
                        tempbw[i, j] = 255.
                    else:
                        if j < index_store_white[0] or j > index_store_white[-1]:
                            tempbw[i, j] = 255.

        for i in range(tempbw.shape[0]):
            for j in range(tempbw.shape[1]):
                if tempbw[i, j] == bw[i, j] and tempbw[i, j] == 0:
                    result[i, j] = 0
                else:
                    result[i, j] = 255

        result = 255 - result

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if result[i, j] == 255:
                    result[i - 1:i + 1, j - 1:j + 1] = 255

        return result


class ImageTDA(object):
    def __init__(self, path, pd_dim=0):
        """
        Constructor

        :param path: image path
        :param pd_dim: we set as 0th PD [For more info, please refers to https://homcloud.dev/basic-usage.en.html]
        """
        self.img_path = path
        self.pd_dim = pd_dim

    def plot_clean_pd_for_ml(self, pd_range, x_show=None, y_show=None):
        """
        This function generates the final output of PD image with new range and colormap by using the Homcloud API.

        :param pd_range: the range of birth and death points
        :param x_show: x axis
        :param y_show: y axis
        """
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

        ''' idiag '''
        pdlist = hc.PDList('%s' % self.img_path)
        pd = pdlist.dth_diagram(self.pd_dim)

        pairs = pd.pairs()
        B = [pair.birth_time() for pair in pairs]
        D = [pair.death_time() for pair in pairs]

        B.append(10001)
        D.append(10001)
        B.append(10003)
        D.append(10002)
        bd_pair = np.vstack([B, D])
        g_kde = gaussian_kde(bd_pair)(bd_pair)

        ''' phtrees '''
        norm = colors.Normalize(vmin=1.0, vmax=2200)
        ax.scatter(B, D, c=g_kde * (1 / g_kde.min()), cmap='rainbow', s=40, marker='s', norm=norm, edgecolor='',
                   zorder=2)

        if x_show is None and y_show is None:
            ax.set_xlim(pd_range)
            ax.set_ylim(pd_range)
        else:
            ax.set_xlim(x_show)
            ax.set_ylim(y_show)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig('DML-Inputs/x1.jpg', bbox_inches='tight', pad_inches=0)
        plt.close()
