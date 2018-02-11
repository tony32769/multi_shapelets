import matplotlib as mpl
import numpy as np

seed = 45   #seed for reproducibility
np.random.seed(seed)


import pandas as pd
import tensorflow as tf

tf.set_random_seed(seed)

from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Dense, BatchNormalization, Dropout, InputLayer, GaussianNoise, Activation
from keras.models import Sequential
from sklearn.preprocessing import scale
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import to_categorical
from model.util import to_tf_format
from mstamp_stomp import mstamp as mstamp_stomp
from keras import regularizers
from model.DistanceLayer import DistanceLayer

import matplotlib.pyplot as plt
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # this makes sure TF only uses the amount of RAM needed
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # this sets the upper band
set_session(tf.Session(config=config))

def discover_shapelets(data, N, lim=0.10, sh_len=0.25):
    """
    Motif-based initialization for shapelets
    :param data: originally formatted dataset
    :param lim: percentage of random instances to sample from each class
    :param sh_len: shapelet length
    :return: the initial guess for shapelets (top motifs)
    """
    shapelets = []
    sub_len = int((data.shape[1] - 3) * sh_len)
    num_channels = len(np.unique(data[data.columns[-2]]))


    for target_class in np.sort(np.unique(data['c{}'.format(data.shape[1] - 3)])):
        print('\n Target class: {}'.format(target_class))
        multi_sample = []
        data_from_class = data.query('c{}=={}'.format(data.shape[1] - 3, target_class))
        lim_samples = int(lim * data_from_class.shape[0] / num_channels)

        if lim_samples < 10:    #  use at least 10 samples
            lim_samples = 10

        if lim_samples > 30:    #  use at most 30 samples
            lim_samples = 30

        print('SAMPLES" {}'.format(lim_samples))

        if lim_samples < data_from_class[data_from_class.columns[-1]].nunique():
            random_sample = np.random.choice(data_from_class[data_from_class.columns[-1]].unique(),
                                             lim_samples, replace=False)
        else:
            random_sample = data_from_class[data_from_class.columns[-1]].unique()

        mask = data_from_class[data_from_class.columns[-1]].isin(random_sample)
        print(np.count_nonzero(mask))
        print(random_sample)

        data_from_class = data_from_class[mask]

        print(data_from_class.shape)
        check = data.shape[1]-3
        #check = int(val_lim/2)
        for ch in range(num_channels):
            data_from_target = data_from_class.query('c{}=={}'.format(data.shape[1] - 2, ch))
            print(data_from_target.shape)
            multi_sample.append(data_from_target.values[:, :check].flatten())

        why = np.vstack(multi_sample)
        mp, _ = mstamp_stomp(why.astype(np.float64), sub_len) # perform mSTAMP

        print(num_channels)
        print(N)
        dimensions = np.random.choice(range(num_channels), N, replace=False)

        for dim_spanned in dimensions:
            shapelet = []
            m = mp[dim_spanned, :].argsort()
            j = 0

            while (m[j] % (check) > (check)-sub_len): #checks that motifs are not overlapping samples
                print('\n {}'.format(m[j] % (check)))
                print((m[j] + sub_len) % (check))
                j += 1

            for i in range(num_channels):
                shapelet.append(why[i, m[j]:m[j] + sub_len])

            shapelets.append(shapelet)
        print('sh {}'.format(len(shapelets)))
    return np.swapaxes(np.array(shapelets), 1, 2)


def random_candidates(data, N, L):
    """
    Random sample N multivariate subsequences
    :param data: TF-formatted dataset
    :param N: number of shapelets
    :param L: length of shapelets
    :return: 
    """
    sample = (np.random.rand(N)*len(data)).astype(int)
    start = (np.random.rand(N)*(data.shape[1]-L)).astype(int)

    samples = []
    for n in range(N):
        samples.append(data[sample[n],start[n]:start[n]+L,:])

    return np.array(samples)




def read_data(FOLDER, fold):
    """
    Utility function to load .csv data
    :param FOLDER: folder containing the datasets
    :param fold: train/val/test fold
    :return: loaded pandas dataframe
    """
    TRAIN = 'new_train_val_test/train_df_{}.csv'.format(fold)
    VAL = 'new_train_val_test/val_df_{}.csv'.format(fold)
    TEST = 'new_train_val_test/test_df_{}.csv'.format(fold)

    df = pd.read_csv(
        '/home/sumo/rmedico/data/MTS/{}/{}'.format(FOLDER, TRAIN))#, index_col=0)
    dfval = pd.read_csv(
        '/home/sumo/rmedico/data/MTS/{}/{}'.format(FOLDER, VAL))#, index_col=0)
    dftest = pd.read_csv(
        '/home/sumo/rmedico/data/MTS/{}/{}'.format(FOLDER, TEST))#, index_col=0)

    """
    PREPROCESSING: scale to N(0,1) and add small random noise
    """
    df[df.columns[:-3]] = scale(df[df.columns[:-3]], axis=1) + np.random.random((df.shape[0],df.shape[1]-3))*0.1
    dfval[dfval.columns[:-3]] = scale(dfval[dfval.columns[:-3]], axis=1) + np.random.random((dfval.shape[0],
                                                                                             df.shape[1]-3))*0.1
    dftest[dftest.columns[:-3]] = scale(dftest[dftest.columns[:-3]], axis=1) + np.random.random((dftest.shape[0],
                                                                                                 df.shape[1]-3))*0.1
    return df, dfval, dftest

def keras_model(data, df, N, hidden=0, init=None, reg=10 ** -2, lr=10 ** -3, sh_len=0.25, classes=None):
    """
    Construct the keras model
    :param data: data in TF-format (N,L,C)
    :param df: data in original format
    :param N: number of shapelets
    :param hidden: number of hidden layers
    :param init: type of initialization
    :param reg: L2-regularization 
    :param lr: learning rate
    :param sh_len: length of shapelets
    :param classes: number of classes
    :return: keras model
    """
    model = Sequential()
    model.add(InputLayer(input_shape=(data.shape[1], data.shape[2])))
    model.add(BatchNormalization())

    if init == 'random_normal':
        model.add(DistanceLayer(N, random_candidates(data, N, L)))
    elif init == 'matrix_profile':
        model.add(DistanceLayer(N, discover_shapelets(df, int(N/classes), 0.10, sh_len=sh_len)))

    model.add(BatchNormalization())

    if hidden>0:
        for h in range(hidden):
            model.add(Dense(2*classes,
                            activation='relu',
                            kernel_regularizer=regularizers.l2(reg),
                            name='hidden_layer{}'.format(h),
                            ))
            model.add(BatchNormalization())
            model.add(Dropout(0.10))

    model.add(Dense(classes,
                    activation='softmax', trainable=True,
                    kernel_regularizer=regularizers.l2(reg),
                    name='output_layer'))


    adam = Adam(lr=lr)
    loss = 'binary_crossentropy' if classes == 2 else 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])

    return model


"""PLOTTING UTILITIES"""
def plot_learning_curve(h, init, hidden, acc):
    plt.figure(0)
    plt.title('Val Acc: {}'.format(acc))
    plt.subplot(211)
    plt.grid()
    plt.plot(h.history['loss'], label='Train Loss')
    plt.plot(h.history['val_loss'], label='Val Loss')

    plt.legend()
    plt.tight_layout()

    plt.subplot(212)
    plt.grid()

    plt.plot(h.history['acc'], label='Train Accuracy')
    plt.plot(h.history['val_acc'], label='Val Accuracy')

    plt.legend()
    plt.tight_layout()

    plt.savefig('{}_{}_{}_{}_learning_curve.pdf'.format(FOLDER.replace('/', '_').replace(' ', '_'), fold, init, hidden))

def plot_shapelet(f, n, n_channels, last, first):
    model.load_weights('weights/' + f)  # "weights/weights_{}_{}.0{}.hdf5".format('yo',fold, (f * period) - 1))

    for channel in range(n_channels):
        plt.subplot(n_channels, 1, channel + 1)
        # plt.ylim((-1,1))
        plt.box(on=None)
        print(len(model.layers[2].get_weights()))
        print(model.layers[2].get_weights()[0].shape)
        # print(model.layers[2].get_weights()[0][n])

        if last:
            plt.plot(model.layers[2].get_weights()[0][n][:, channel], c='b', linewidth=2, label='final')
        elif first:
            plt.plot(model.layers[2].get_weights()[0][n][:, channel], c='g', label='initial')
        else:
            plt.plot(model.layers[2].get_weights()[0][n][:, channel], alpha=.1, c='r')

    plt.legend(loc='best')

def shuffle_data(D, L):
    """ Random shuffle data """
    idx = np.random.permutation(len(D))
    return D[idx], L[idx]


class TestCallback(Callback):
    """ Callback to compute accuracy on test data """
    def __init__(self, test_data, batch_size):
        self.test_data = test_data
        self.batch_size = batch_size
        self.accs = []  # keep track of the scores

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0, batch_size=self.batch_size)
        self.accs.append(acc)
        print('\ntest_loss: {}, test_acc: {}\n'.format(loss, acc))


def get_data_fold(dataset_name, fold):
    """
    Get data for specific fold
    :param dataset_name: 
    :param fold: 
    :return: 
    """
    FOLDER = 'Aarons Official/{}'.format(dataset_name)

    print('Reading data for fold {}..'.format(fold))
    df, dfval, dftest = read_data(FOLDER, fold)
    C = len(df[df.columns[-2]].unique())
    train_labels = df.iloc[::C, -3].values.copy()
    val_labels = dfval.iloc[::C, -3].values.copy()
    test_labels = dftest.iloc[::C, -3].values.copy()
    all_labels = np.unique(np.array(list(train_labels) + list(test_labels) + list(val_labels)))

    N_CLASSES = len(all_labels)
    if np.min(all_labels) > 0:  # make sure labels are 0-based
        train_labels -= 1
        test_labels -= 1
        val_labels -= 1
        all_labels -= 1
    print(len(np.unique(all_labels)))

    train_labels = to_categorical(train_labels, len(np.unique(all_labels)))
    test_labels = to_categorical(test_labels, len(np.unique(all_labels)))
    val_labels = to_categorical(val_labels, len(np.unique(all_labels)))
    data, labels = df.iloc[:, :-3].values, train_labels
    val, val_labels = dfval.iloc[:, :-3].values, val_labels
    test, test_labels = dftest.iloc[:, :-3].values, test_labels
    df = df.rename(columns={c[1]: 'c{}'.format(c[0]) for c in enumerate(df.columns)})
    data = to_tf_format(data, NUM_CHANNELS=C)
    val = to_tf_format(val, NUM_CHANNELS=C)
    test = to_tf_format(test, NUM_CHANNELS=C)
    print(data.shape)
    print(val.shape)
    print(test.shape)
    data, labels = shuffle_data(data, labels)
    val, val_labels = shuffle_data(val, val_labels)
    test, test_labels = shuffle_data(test, test_labels)

    return data, val, test, labels, val_labels, test_labels, df, N_CLASSES,C

if __name__ == "__main__":
    for dataset_name in [
        'ArticularyWordUL',
        'ArticularyWordT1',
        'ArticularyWordLL',
        'uWaveGesture',
        'HandwritingAccelerometer',
        'HandwritingGyroscope',
        'ArabicDigit',
    ]:
        rows = []
        folds = 3                                       # number of train/val/test splits
        epochs = 500                                    # max epochs
        early_stopping = 25                             # patience
        bs = 64                                         # batch size

        period = 1
        print('Each fold is trained on for {} epochs'.format(epochs))

        """
        HYPER_PARAMS OPTIMIZATION
        """
        loaded = []
        for fold in range(folds):
            data, val, test, labels, val_labels, test_labels, df, N_CLASSES, C = get_data_fold(
                dataset_name, fold)
            for reg in [0.01,.1]:
                    for sh_len in [0.15,0.25]:
                        for num_shapelet in [3,2]:
                            for lr in [10 ** -3]:
                                for init in ['random_normal','matrix_profile']:
                                    for hidden in [1, 0]:

                                            N = N_CLASSES*num_shapelet
                                            L = int(sh_len * data.shape[1])

                                            print('Using {} shapelets'.format(N))
                                            print(L)
                                            print('Building model for fold {}..'.format(fold))
                                            model = keras_model(data, df, N, hidden=hidden,
                                                                init=init, lr=lr, reg=reg, sh_len=sh_len,
                                                                classes=N_CLASSES)

                                            callbacks = []
                                            save_model = False

                                            if save_model:
                                                import shutil

                                                try:
                                                    shutil.rmtree('weights_{}_{}'.format(init, hidden))
                                                except:
                                                    pass
                                                os.makedirs('weights_{}_{}'.format(init, hidden))

                                                callbacks.append(ModelCheckpoint('weights_' + '{}_{}'.format(init,
                                                                                                             hidden) + '/weights_best' + '_' + str(
                                                    fold) + '.{epoch:02d}-{val_acc:.3f}.hdf5',
                                                                                 monitor='val_acc',
                                                                                 verbose=0,
                                                                                 save_best_only=False,
                                                                                 mode='auto',
                                                                                 period=1
                                                                                 ))

                                            test_callback = TestCallback((test, test_labels), batch_size=bs)
                                            callbacks.append(test_callback)

                                            callbacks.append(EarlyStopping(monitor='val_loss',
                                                                           min_delta=10**-4,
                                                                           patience=early_stopping,
                                                                           verbose=0,
                                                                           mode='auto'))

                                            print('Training model for fold {}..'.format(fold))

                                            h = model.fit(data, labels, nb_epoch=epochs,
                                                          shuffle=True, batch_size=bs,
                                                          validation_data=(val, val_labels),
                                                          verbose=1,
                                                          callbacks=callbacks)

                                            print(np.argmax(h.history['val_acc']))
                                            stopped_at = np.argmin(h.history['val_loss'])
                                            test_acc = test_callback.accs[stopped_at]
                                            val_acc = h.history['val_acc'][stopped_at]
                                            train_acc = h.history['acc'][stopped_at]
                                            len_epochs = len(h.history['val_acc'])

                                            print('test_acc: {}'.format(test_acc))
                                            print('val_acc: {}'.format(val_acc))
                                            save_plots = False

                                            if save_plots:
                                                import os

                                                print('Plotting shapelets for fold {}..'.format(fold))

                                                n_epochs = epochs / period
                                                files = np.sort(
                                                    [f for f in os.listdir('weights') if '{}_{}'.format('best', fold) in f])
                                                print(files)
                                                for sh in range(N):
                                                    plt.figure(figsize=(30, 60))
                                                    tot = int(len(files) / 10)
                                                    for f in enumerate(files):  # range(1, len(files) + 1):
                                                        if f[0] % 10 == 0:
                                                            plot_shapelet(f[1], sh, C, tot == 0, f[0] == 0)
                                                        tot -= 1

                                                    plt.savefig('{}_{}_shapelet_#_{}.pdf'.format('yo', fold, sh),
                                                                fig=1)

                                            rows.append([fold,init, hidden, lr, reg, sh_len, N, val_acc,
                                                         test_acc,
                                                         train_acc,
                                                         len_epochs
                                                         ])
                                            res_df = pd.DataFrame(rows, columns=[
                                                                        'fold',
                                                                        'initialization',
                                                                        'hidden_layers',
                                                                        'learning_rate',
                                                                        'regularization',
                                                                        'shapelet_length',
                                                                        'num_shapelets',
                                                                        'val_acc',
                                                                        'test_acc',
                                                                        'train_acc',
                                                                        'epochs'
                                                                        ])

                                            res_df = res_df.sort_values(by='val_acc')[::-1]

                                            res_df.to_csv(
                                                'results/results_{}_dropout_val_split_seed_{}_fold_{}.csv'.format(
                                                    dataset_name.replace('/', '_').replace(' ', '_'), seed, fold))

