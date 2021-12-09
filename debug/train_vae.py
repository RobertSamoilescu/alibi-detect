import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import pickle as pkl
import tensorflow as tf

from alibi_detect.od import OutlierVAE
from alibi_detect.datasets import fetch_kdd
from alibi_detect.models.tensorflow.losses import elbo
from alibi_detect.utils.data import create_outlier_batch
from alibi_detect.utils.saving import save_detector
from sklearn.model_selection import train_test_split

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, help="gpu device to be used", default=0)
    parser.add_argument("--id", type=int, help="experiment id", required=True)
    parser.add_argument("--random-state", type=int, help="random state to be used", default=0)
    parser.add_argument("--epochs", type=int, help="number of epochs to train the od", default=30)
    parser.add_argument("--sim", type=float, help="std dev of output", default=0.01)
    args = parser.parse_args()
    return args


def get_kddcup(random_state: int = 0):
    # fetch only load 10% of the dataset
    kddcup = fetch_kdd(percent10=True, random_state=random_state)

    data_train, data_test, target_train, target_test = train_test_split(kddcup.data,
                                                                        kddcup.target,
                                                                        test_size=0.5,
                                                                        random_state=random_state)

    data_test, data_threshold, target_test, target_threshold = train_test_split(data_test,
                                                                                target_test,
                                                                                test_size=0.5,
                                                                                random_state=random_state)

    kddcup_train = {'data': data_train, 'target': target_train, 'feature_names': kddcup.feature_names}
    kddcup_test = {'data': data_test, 'target': target_test, 'feature_names': kddcup.feature_names}
    kddcup_threshold = {'data': data_threshold, 'target': target_threshold, 'feature_names': kddcup.feature_names}
    return kddcup_train, kddcup_test, kddcup_threshold


def get_dataset(args, perc_outlier_threshold: int = 5, perc_outlier: int = 10):
    # fetch only load 10% of the dataset
    kddcup_train, kddcup_outlier, kddcup_threshold = get_kddcup(random_state=0)

    # sample training dataset
    normal_batch = create_outlier_batch(kddcup_train['data'],
                                        kddcup_train['target'],
                                        n_samples=247010,
                                        perc_outlier=0,
                                        random_state=args.random_state)
    X_train, y_train = normal_batch.data.astype('float'), normal_batch.target
    mean, stdev = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / stdev

    # sample threshold dataset
    threshold_batch = create_outlier_batch(kddcup_threshold['data'],
                                           kddcup_threshold['target'],
                                           n_samples=1000,
                                           perc_outlier=perc_outlier_threshold)
    X_threshold, y_threshold = threshold_batch.data.astype('float'), threshold_batch.target
    X_threshold = (X_threshold - mean) / stdev

    # sample outlier dataset
    outlier_batch = create_outlier_batch(kddcup_outlier['data'],
                                         kddcup_outlier['target'],
                                         n_samples=1000,
                                         perc_outlier=perc_outlier)
    X_outlier, y_outlier = outlier_batch.data.astype('float'), outlier_batch.target
    X_outlier = (X_outlier - mean) / stdev

    return {
        'train': {'X': X_train, 'y': y_train},
        'outlier': {'X': X_outlier, 'y': y_outlier},
        'threshold': {'X': X_threshold, 'y': y_threshold}
    }


def train_od(X_train, filepath, args):
    n_features = X_train.shape[1]
    latent_dim = 2

    encoder_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(n_features,)),
            tf.keras.layers.Dense(20, activation=tf.nn.relu),
            tf.keras.layers.Dense(15, activation=tf.nn.relu),
            tf.keras.layers.Dense(7, activation=tf.nn.relu)
        ])

    decoder_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(7, activation=tf.nn.relu),
            tf.keras.layers.Dense(15, activation=tf.nn.relu),
            tf.keras.layers.Dense(20, activation=tf.nn.relu),
            tf.keras.layers.Dense(n_features, activation=None)
        ])

    # initialize outlier detector
    od = OutlierVAE(threshold=None,  # threshold for outlier score
                    score_type='mse',  # use MSE of reconstruction error for outlier detection
                    encoder_net=encoder_net,  # can also pass VAE model instead
                    decoder_net=decoder_net,  # of separate encoder and decoder
                    latent_dim=latent_dim,
                    samples=5)
    # train
    od.fit(X_train,
           loss_fn=elbo,
           cov_elbo=dict(sim=args.sim),
           epochs=args.epochs,
           verbose=True)

    # save the trained outlier detector
    save_detector(od, filepath)


if __name__ == "__main__":
    # parse arguments
    args = get_arguments()

    # set available gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(args.gpu, True)

    # load dataset
    data = get_dataset(perc_outlier_threshold=5, perc_outlier=10, args=args)

    # save dataset
    path = os.path.join("ckpts", f"id={args.id}, seed={args.random_state}, epochs={args.epochs}, sim={args.sim}")
    if not os.path.exists(path):
        os.makedirs(path)

    data_path = os.path.join(path, "data.pkl")
    with open(data_path, "wb") as fout:
        pkl.dump(data, fout)

    # train outlier detector
    train_od(X_train=data['train']['X'], filepath=path, args=args)