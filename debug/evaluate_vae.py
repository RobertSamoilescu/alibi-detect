import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import numpy as np
import pickle as pkl
import tensorflow as tf

from alibi_detect.utils.saving import load_detector
from alibi_detect.od import OutlierVAE
from sklearn.metrics import f1_score


def evaluate_od(od: OutlierVAE,
                X_threshold: np.ndarray,
                X_outlier: np.ndarray,
                y_outlier: np.ndarray,
                perc_outlier_threshold: int = 5,
                verbose: int = 0):

    # infer outlier detector threshold
    od.infer_threshold(X_threshold, threshold_perc=100 - perc_outlier_threshold)
    if verbose:
        print('New threshold: {}'.format(od.threshold))

    # detect outlier
    od_preds = od.predict(X_outlier,
                          outlier_type='instance',    # use 'feature' or 'instance' level
                          return_feature_score=True,  # scores used to determine outliers
                          return_instance_score=True)

    # compute f1 score
    y_pred = od_preds['data']['is_outlier']
    f1 = f1_score(y_outlier, y_pred)
    if verbose:
        print(f"F1 score: {f1}")

    return f1


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, help="gpu device to be used", default=0)
    parser.add_argument("--id", type=int, help="experiment id", required=True)
    parser.add_argument("--random-state", type=int, help="random state to be used", default=0)
    parser.add_argument("--epochs", type=int, help="number of epochs to train the od", default=30)
    parser.add_argument("--sim", type=float, help="std dev of output", default=0.01)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # parse arguments
    args = get_arguments()

    # set available gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(args.gpu, True)

    # load dataset
    root_dir = os.path.join("ckpts", f"id={args.id}, seed={args.random_state}, epochs={args.epochs}, sim={args.sim}")
    data_path = os.path.join(root_dir, "data.pkl")
    with open(data_path, 'rb') as fin:
        data = pkl.load(fin)

    # load detect
    od_path = os.path.join(root_dir)
    od = load_detector(od_path)

    # evaluate od
    evaluate_od(od=od,
                X_threshold=data['threshold']['X'],
                X_outlier=data['outlier']['X'],
                y_outlier=data['outlier']['y'],
                verbose=1)

