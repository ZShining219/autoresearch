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

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='METR-LA', help="dataset")
args = parser.parse_args()


def standard_transform(data: np.array,
                       output_dir: str,
                       train_index: list,
                       seq_len: int,
                       norm_each_channel: int = False) -> np.array:
    """Standard normalization.

    Args:
        data (np.array): raw time series data.
        output_dir (str): output dir path.
        train_index (list): train index.
        seq_len (int): sequence length.
        norm_each_channel (bool): whether to normalization each channel.

    Returns:
        np.array: normalized raw time series data.
    """

    # data: L, N, C, C=1
    data_train = data[:train_index[-1][1], ...]
    if norm_each_channel:
        mean, std = data_train.mean(axis=0, keepdims=True), data_train.std(
            axis=0, keepdims=True)
    else:
        mean, std = data_train[..., 0].mean(), data_train[..., 0].std()

    print("mean (training data):", mean)
    print("std (training data):", std)
    normalization = {}
    normalization["func"] = "standard_transform"
    normalization["args"] = {"mean": mean, "std": std}
    # label to identify the scaler for different settings.
    with open(output_dir + "/normalization.pkl", "wb") as f:
        pickle.dump(normalization, f)

    def normalize(x):
        return (x - mean) / std

    data_norm = normalize(data)
    return data_norm


def generate_dataset(data, seqlen, mode):
    l, _, f = data.shape
    train_nums = round(l * train_ratio)
    valid_nums = round(l * valid_ratio)
    test_nums = l - train_nums - valid_nums
    print("timespan of training samples: {0}".format(train_nums))
    print("timespan of validation samples: {0}".format(valid_nums))
    print("timespan of test samples:{0}".format(test_nums))

    if mode == "overlap":
        train_data_index = []
        for t in range(seqlen, train_nums + 1):
            index = (t - seqlen, t)
            train_data_index.append(index)

        valid_data_index = []
        for t in range(train_nums + seqlen, train_nums + valid_nums + 1):
            index = (t - seqlen, t)
            valid_data_index.append(index)

        test_data_index = []
        for t in range(train_nums + valid_nums + seqlen,
                       train_nums + valid_nums + test_nums):
            index = (t - seqlen, t)
            test_data_index.append(index)
    else:
        train_num_segments = train_nums // seqlen
        train_data_index = [(i * seqlen, (i + 1) * seqlen)
                            for i in range(train_num_segments)]

        valid_num_segments = valid_nums // seqlen
        valid_data_index = [(train_nums + i * seqlen,
                             train_nums + (i + 1) * seqlen)
                            for i in range(valid_num_segments)]

        test_num_segments = test_nums // seqlen
        test_data_index = [(train_nums + valid_nums + i * seqlen,
                            train_nums + valid_nums + (i + 1) * seqlen)
                           for i in range(test_num_segments)]
    # print(train_data_index)
    # print(valid_data_index)
    # print(test_data_index)

    print('train data samples: {}, from {} to {}'.format(
        len(train_data_index), train_data_index[0][0],
        train_data_index[-1][-1]))
    print('valid data samples: {}, from {} to {}'.format(
        len(valid_data_index), valid_data_index[0][0],
        valid_data_index[-1][-1]))
    print('test data samples: {}, from {} to {}'.format(
        len(test_data_index), test_data_index[0][0], test_data_index[-1][-1]))

    index = {}
    index['train'] = train_data_index
    index['valid'] = valid_data_index
    index['test'] = test_data_index

    data_norm = standard_transform(data, output_dir, train_data_index, seqlen)

    with open(output_dir + "/data.pkl", "wb") as f:
        pickle.dump(data_norm, f)
    with open(output_dir + "/index_{}.pkl".format(seqlen), "wb") as f:
        pickle.dump(index, f)
    # return train_data_index,valid_data_index,test_data_index


if __name__ == "__main__":
    seq_len = 12  # sliding window size for generating sequence
    train_ratio = 0.7  # train dataset size
    valid_ratio = 0.1  # valid dataset size
    target_channel = [0]  # target channel(s)
    mode = None  # if overlap splitting then mode='overlap'

    DATASET_NAME = args.dataset
    output_dir = "./datasets/{}/processed".format(DATASET_NAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if DATASET_NAME == 'METR-LA' or DATASET_NAME == 'PEMS-BAY':
        data_file_path = "./datasets/{}/{}.h5".format(DATASET_NAME,
                                                      DATASET_NAME)
        df = pd.read_hdf(data_file_path)
        data = np.expand_dims(df.values, axis=-1)
        graph_file_path = "./datasets/{}/adj_mx.pkl".format(DATASET_NAME)
        if DATASET_NAME == 'PEMS-BAY':
            data = data[:51840, :, target_channel]  # 51840=288*180
        else:
            data = data[..., target_channel]
    elif DATASET_NAME == 'Seattle':
        data_path = './datasets/Seattle'
        file_name = 'Seattle.csv'
        graph_file_path = "./datasets/{}/adj_mx.pkl".format(DATASET_NAME)
        data = pd.read_csv(os.path.join(data_path, file_name)).values
        data = np.expand_dims(data, axis=-1)
    elif DATASET_NAME == 'Chengdu' or DATASET_NAME == 'Shenzhen':
        seq_len = 6  # sequence length in Chengdu or Shenzhen
        train_ratio = 0.6
        valid_ratio = 0.2
        data_file_path = "./datasets/{}/dataset.npy".format(DATASET_NAME)
        data = np.load(data_file_path)
        graph_file_path = "./datasets/{}/adj_mx.pkl".format(DATASET_NAME)
        data = data[..., target_channel]

    print('Processing dataset: ', DATASET_NAME)
    print("raw time series shape: {0}".format(data.shape))

    generate_dataset(data, seq_len, mode)