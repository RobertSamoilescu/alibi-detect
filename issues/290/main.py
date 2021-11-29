import os
import yaml
import numpy as np
import pickle as pkl
import timesynth as ts
import itertools
import random

from tqdm import tqdm
from typing import List, Dict, Union
from alibi_detect.od import SpectralResidual
from alibi_detect.utils.perturbation import inject_outlier_ts
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def generate_signal(n_points: int,
                    offset: float,
                    perc_outlier: int,
                    seed: int = 0) -> Dict[str, Union[np.ndarray, List[str]]]:
    """
    Generates a noised signal.

    Parameters
    ----------
    n_points:
        Number of points to be generated.
    offset:
        Signal offset.
    perc_outlier:
        Percentage of outliers contained in the signal.
    seed:
        Random seed.

    Returns
    -------
    Dictionary containing:
     - `time_samples` - regular time samples.
     - `X_outlier` - noised signal
     - `y_outlier` - target
     - `labels` - list of classification labels.
    """
    # timestamps
    time_sampler = ts.TimeSampler(stop_time=n_points // 4)
    time_samples = time_sampler.sample_regular_time(num_points=n_points)

    # harmonic time series with Gaussian noise
    sinusoid = ts.signals.Sinusoidal(frequency=0.25, ftype=np.cos)
    white_noise = ts.noise.GaussianNoise(std=0.1)
    ts_harm = ts.TimeSeries(signal_generator=sinusoid, noise_generator=white_noise)
    samples, signals, errors = ts_harm.sample(time_samples)

    # add some offset
    samples += offset
    signals += offset

    # inject outlier in the timeserie
    X = samples.reshape(-1, 1).astype(np.float32)
    data = inject_outlier_ts(X, perc_outlier=10, perc_window=10, n_std=2., min_std=1.)
    X_outlier, y_outlier, labels = data.data, data.target.astype(int), data.target_names
    return {
        "time_samples": time_samples,
        "X_outlier": X_outlier,
        "y_outlier": y_outlier,
        "labels": labels
    }


def compute_scores(y_pred: np.ndarray, y_outlier: np.ndarray):
    """
    Computes metrics for the given prediction.

    Parameters
    ----------
    y_pred:
        Outlier detector predictions.
    y_outlier:
        Ground-truth labels

    Returns
    -------
    F1 scores, accruacy and recall.
    """
    f1 = f1_score(y_outlier, y_pred)
    acc = accuracy_score(y_outlier, y_pred)
    rec = recall_score(y_outlier, y_pred)
    return {"f1": f1, "acc": acc, "rec": rec}


def run_experiment(time_samples: np.ndarray,
                   X_outlier: np.ndarray,
                   y_outlier: np.ndarray,
                   config_exp: Dict,
                   experiment_id: int,
                   experiment_number: int):

    if not os.path.exists("ckpts"):
        os.makedirs("ckpts")

    file_name = os.path.join("ckpts", str(experiment_id) + ".pkl")
    if os.path.exists(file_name):
        with open(file_name, "rb") as fin:
            results = pkl.load(fin)

        # skip if already computed
        if len(results[list(results.keys())[0]]) > experiment_number:
            return
    else:
        results = {}

    # define outlier detector
    od = SpectralResidual(threshold=None,
                          window_amp=config_exp["window_amp"],
                          window_local=config_exp["window_local"],
                          n_est_points=20,
                          use_old=config_exp["use_old"],
                          padding_amp_method=config_exp["padding_amp_method"],
                          padding_local_method=config_exp["padding_local_method"],
                          padding_amp_side=config_exp["padding_amp_side"])

    # infer threshold from the data
    od.infer_threshold(X_outlier[:10000], time_samples[:10000], threshold_perc=90)

    # run detector
    od_preds = od.predict(X_outlier, time_samples, return_instance_score=True)

    # compute scores
    scores = compute_scores(y_pred=od_preds['data']['is_outlier'], y_outlier=y_outlier)
    config_exp.update(scores)

    # update results
    for key in config_exp:
        results[key] = results.get(key, []) + [config_exp[key]]

    with open(file_name, "wb") as fin:
        pkl.dump(results, fin)


def run_all():
    num_experiments = 10
    n_points = [128, 256, 512, 1024]
    offset = [0, 10, 100]
    perc_outlier = [10, 20, 30]
    window_amp = [4, 8, 16, 32]
    window_local = [4, 8, 16, 32]
    use_old = [True, False]
    padding_amp_method = [None, 'constant', 'replicate', 'reflect']
    padding_local_method = [None, 'constant', 'replicate', 'reflect']
    padding_amp_side = ['bilateral']  # ["bilateral", "left", "right"]

    # generate all possible combinations of parameters
    combinations = itertools.product(
        n_points,
        offset,
        perc_outlier,
        window_amp,
        window_local,
        use_old,
        padding_amp_method,
        padding_local_method,
        padding_amp_side,
    )

    # filter out the where `use_old == True` and `padding_* != None`
    filtered_comb = []
    for comb in combinations:
        uo, pam, plm = comb[-4], comb[-3], comb[-2]
        # skip test
        if uo and (pam is not None or plm is not None):
            continue
        # skip test
        if (not uo) and (pam is None or plm is None):
            continue
        filtered_comb.append(comb)

    # run each experiment
    for i, comb in tqdm(enumerate(filtered_comb)):
        for j in range(num_experiments):
            # generate signal
            signal = generate_signal(n_points=comb[0], offset=comb[1], perc_outlier=comb[2])

            # unpack data
            time_samples = signal['time_samples']
            X_outlier = signal["X_outlier"]
            y_outlier = signal["y_outlier"]

            config_exp = {
                "n_points": comb[0],
                "offset": comb[1],
                "perc_outlier": comb[2],
                "window_amp": comb[3],
                "window_local": comb[4],
                "use_old": comb[5],
                "padding_amp_method": comb[6],
                "padding_local_method": comb[7],
                "padding_amp_side": comb[8] if not comb[5] else None,
            }
            run_experiment(time_samples=time_samples,
                           X_outlier=X_outlier,
                           y_outlier=y_outlier,
                           config_exp=config_exp,
                           experiment_id=i,
                           experiment_number=j)


if __name__ == "__main__":
    run_all()