import os

import numpy as np

import matplotlib.pyplot as plt


path = os.path.dirname(os.path.abspath(__file__))


def test_plot_2D(fullQENSDataset):
    qens, res, ec, bkgd = fullQENSDataset
    qens[0].plot()
    data = plt.gcf().axes[0].lines[-1].get_xydata()
    ref = np.loadtxt(path + "/sample_data/plot_data_2D.txt")
    assert np.allclose(data, ref)


def test_plot_3D(fullQENSDataset):
    qens, res, ec, bkgd = fullQENSDataset
    qens.plot_3D()
    data = plt.gca().get_children()[0]._vec
    ref = np.loadtxt(path + "/sample_data/plot_data_3D.txt")
    assert np.allclose(data, ref)
