import os
import sys
import shutil
import pickle
import argparse
import torch
import random
import itertools
import numpy as np
import pandas as pd
from utils import *
from mask import *
import time

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="METR-LA", help="dataset")
parser.add_argument(
    "--miss_mechanism", type=str, default="MCAR", help="missing mechanism"
)
parser.add_argument("--miss_ratio", type=float, default=0.5, help="missing ratio")
parser.add_argument("--seqlen", type=int, default=12, help="missing ratio")
parser.add_argument(
    "--opt", type=str, default="logistic", help="missing methods in MNAR"
)
parser.add_argument(
    "--p_obs",
    type=float,
    default=0.5,
    help="the probability of observed component when MAR or MNAR",
)
parser.add_argument(
    "--q",
    type=float,
    default=0.3,
    help="quantile level at which the cuts should occur when MNAR in quantile",
)
args = parser.parse_args()

dataset = args.dataset
miss_mechanism = args.miss_mechanism
miss_ratio = args.miss_ratio
mnar_opt = args.opt
mnar_p_obs = args.p_obs
mnar_q = args.q
seqlen = args.seqlen


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def produce_NA(X, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)  # numpy to tensor

    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs, False).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1 - p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    else:
        mask = (torch.rand(X.shape) < p_miss).double()

    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan
    return {"X_init": X.double(), "X_incomp": X_nas.double(), "mask": mask}


def generate_missing(data, miss_ratio):
    seed_torch(0)
    n_masks = data.shape[0]  # batch_size

    missdata_list = [
        produce_NA(data[i], miss_ratio, miss_mechanism, mnar_opt, mnar_p_obs, mnar_q)
        for i in range(n_masks)
    ]

    X_miss = []
    M_miss = []
    for missdata in missdata_list:
        X_miss.append(missdata["X_incomp"].numpy())
        M_miss.append(1 - missdata["mask"].numpy())

    return np.array(X_miss), np.array(M_miss)


def generate_dataset(dataset, data, index):
    add_position_encoding(dataset, data, add_time_in_week=True, seqlen=seqlen)

    train_index, valid_index, test_index = index["train"], index["valid"], index["test"]
    train_data, valid_data, test_data = [], [], []  # B,N,F
    for item in train_index:
        train_sample = data[item[0] : item[1], ...].squeeze()
        train_data.append(train_sample)
    for item in valid_index:
        valid_sample = data[item[0] : item[1], ...].squeeze()
        valid_data.append(valid_sample)
    for item in test_index:
        test_sample = data[item[0] : item[1], ...].squeeze()
        test_data.append(test_sample)
    train_data = np.array(train_data).transpose(0, 2, 1)
    valid_data = np.array(valid_data).transpose(0, 2, 1)
    test_data = np.array(test_data).transpose(0, 2, 1)
    print("missing rate: ", miss_ratio)
    print("train data shape: ", train_data.shape)
    print("valid data shape: ", valid_data.shape)
    print("test data shape: ", test_data.shape)

    train_set, train_mask = generate_missing(train_data, miss_ratio)
    valid_set, valid_mask = generate_missing(valid_data, miss_ratio)
    test_set, test_mask = generate_missing(test_data, miss_ratio)

    train_dict, valid_dict, test_dict = {}, {}, {}
    train_dict["data"], train_dict["mask"], train_dict["target"] = (
        train_set,
        train_mask,
        train_data,
    )
    valid_dict["data"], valid_dict["mask"], valid_dict["target"] = (
        valid_set,
        valid_mask,
        valid_data,
    )
    test_dict["data"], test_dict["mask"], test_dict["target"] = (
        test_set,
        test_mask,
        test_data,
    )

    output_dir = "datasets/{}/processed/{}/".format(dataset, miss_mechanism)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(
        output_dir
        + "train_{0}_ms{1}_seqlen_{2}.pkl".format(miss_mechanism, miss_ratio, seqlen),
        "wb",
    ) as f:
        pickle.dump(train_dict, f)
    with open(
        output_dir
        + "valid_{0}_ms{1}_seqlen_{2}.pkl".format(miss_mechanism, miss_ratio, seqlen),
        "wb",
    ) as f:
        pickle.dump(valid_dict, f)
    with open(
        output_dir
        + "test_{0}_ms{1}_seqlen_{2}.pkl".format(miss_mechanism, miss_ratio, seqlen),
        "wb",
    ) as f:
        pickle.dump(test_dict, f)


def load_pkl(pickle_file: str) -> object:
    """Load pickle data.

    Args:
        pickle_file (str): file path

    Returns:
        object: loaded objected
    """

    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def add_position_encoding(dataset, data, add_time_in_week, seqlen):
    num_nodes = data.shape[1]
    print("num nodes:", num_nodes)
    if add_time_in_week:
        # numerical time_in_week
        if seqlen == 12:
            time_inw = [i % 2016 / 2016 for i in range(data.shape[0])]
        elif seqlen == 6:
            time_inw = [i % 1008 / 1008 for i in range(data.shape[0])]
        else:
            assert "seqlen must be 6 or 12!"
        time_inw = np.array(time_inw)
        time_in_week = np.tile(time_inw[:, np.newaxis, np.newaxis], [1, num_nodes, 1])
    newdata = np.concatenate((data, time_in_week), axis=-1)
    with open("datasets/{}/data_pos.pkl".format(dataset), "wb") as fb:
        pickle.dump(newdata, fb)


if __name__ == "__main__":
    print("=" * 50, "Processing", dataset, miss_mechanism, "=" * 50)
    start_time = time.time()

    data = load_pkl("datasets/{}/processed/data.pkl".format(dataset))
    index = load_pkl("datasets/{}/processed/index_{}.pkl".format(dataset, seqlen))

    generate_dataset(dataset, data, index)
    print("spend_time: {:.2f}".format(time.time() - start_time))
    print("=" * 50, "Processing Finished", "=" * 50)
