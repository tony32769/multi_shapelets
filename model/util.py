from sklearn.cluster import KMeans

from model import *


def to_tf_format(dataset, NUM_CHANNELS=2):
    """
    Format a multivariate dataset in a tf-friendly format
    :param dataset: 
    :return: 
    """
    new_d = []
    for i in range(0, len(dataset), NUM_CHANNELS):
        a = np.empty((dataset.shape[1], NUM_CHANNELS))
        for j in range(NUM_CHANNELS):
            a[:, j] = dataset[i + j, :]
        new_d.append(a)
    return np.array(new_d)


def multipage(filename, figs=None, dpi=200):
    """
    CREATES A PDF FILE CONTAINING MULTIPLE PLOTS
    :param filename: 
    :param figs: 
    :param dpi: 
    :return: 
    """
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

