import os
import random
import numpy as np
import torch
import pickle as pk
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from utils import StandardScaler


def load_data(dataset, miss_mechanism, miss_ratio, seqlen):
    """
    read traffic data:
    """
    # get adjacency matrix
    with open("datasets/{}/adj_mx.pkl".format(dataset), "rb") as fb:
        A = pk.load(fb).astype(np.float32)
    # get normalization metric: mean, std
    with open("datasets/{}/processed/normalization.pkl".format(dataset), "rb") as fb:
        mean, std = pk.load(fb)["args"].values()

    # train
    with open(
        "datasets/{}/processed/{}/train_{}_ms{}_seqlen_{}.pkl".format(
            dataset, miss_mechanism, miss_mechanism, miss_ratio, seqlen
        ),
        "rb",
    ) as fb:
        train_data = pk.load(fb)
    # train_X = np.expand_dims(train_data['data'].astype(np.float32), 3)
    train_X = np.expand_dims(np.nan_to_num(train_data["data"]), 3).astype(
        np.float32
    )  # (B,N,L,1)
    train_M = np.expand_dims(train_data["mask"].astype(np.float32), 3)  # (B,N,L,1)
    train_Y = np.expand_dims(train_data["target"].astype(np.float32), 3)  # (B,N,L,1)

    # valid
    with open(
        "datasets/{}/processed/{}/valid_{}_ms{}_seqlen_{}.pkl".format(
            dataset, miss_mechanism, miss_mechanism, miss_ratio, seqlen
        ),
        "rb",
    ) as fb:
        valid_data = pk.load(fb)
    # valid_X = np.expand_dims(valid_data['data'].astype(np.float32), 3)
    valid_X = np.expand_dims(np.nan_to_num(valid_data["data"]), 3).astype(
        np.float32
    )  # (B,N,L,1)
    valid_M = np.expand_dims(valid_data["mask"].astype(np.float32), 3)  # (B,N,L,1)
    valid_Y = np.expand_dims(valid_data["target"].astype(np.float32), 3)  # (B,N,L,1)

    # test
    with open(
        "datasets/{}/processed/{}/test_{}_ms{}_seqlen_{}.pkl".format(
            dataset, miss_mechanism, miss_mechanism, miss_ratio, seqlen
        ),
        "rb",
    ) as fb:
        test_data = pk.load(fb)
    # test_X = np.expand_dims(test_data['data'].astype(np.float32), 3)
    test_X = np.expand_dims(np.nan_to_num(test_data["data"]), 3).astype(
        np.float32
    )  # (B,N,L,1)
    test_M = np.expand_dims(test_data["mask"].astype(np.float32), 3)  # (B,N,L,1)
    test_Y = np.expand_dims(test_data["target"].astype(np.float32), 3)  # (B,N,L,1)

    return (
        train_X,
        train_M,
        train_Y,
        valid_X,
        valid_M,
        valid_Y,
        test_X,
        test_M,
        test_Y,
        A,
        mean,
        std,
    )


def generate_miss_loader(
    dataset,
    miss_mechanism,
    miss_ratio,
    seqlen,
    batch_size,
    val_batch_size,
    test_batch_size,
):
    # split to train,valid,test
    with open("datasets/{}/processed/index_{}.pkl".format(dataset, seqlen), "rb") as fb:
        index = pk.load(fb)

    def add_pos_emb(mode, X, Y):
        pos_emb = []
        for item in index[mode]:
            pos = datapos[item[0] : item[1], :, 1]
            pos_emb.append(pos)
        pos_emb = np.array(pos_emb).transpose(0, 2, 1)
        pos_emb = np.expand_dims(pos_emb, 2)  # (B,N,1,L)
        X = X.transpose(0, 1, 3, 2)  # (B,N,1,L)
        Y = Y.transpose(0, 1, 3, 2)  # (B,N,1,L)
        X = np.concatenate([X, pos_emb], axis=2).astype(np.float32)
        Y = np.concatenate([Y, pos_emb], axis=2).astype(np.float32)
        return X, Y

    data = {}

    (
        train_X,
        train_M,
        train_Y,
        valid_X,
        valid_M,
        valid_Y,
        test_X,
        test_M,
        test_Y,
        A,
        mean,
        std,
    ) = load_data(dataset, miss_mechanism, miss_ratio, seqlen)

    # get data with position information
    with open("datasets/{}/data_pos.pkl".format(dataset), "rb") as fb:
        datapos = pk.load(fb)

    # train
    train_M = np.repeat(train_M, 2, axis=-1)
    train_M = train_M.transpose(0, 1, 3, 2)  # (B,N,2,L)
    train_X, train_Y = add_pos_emb("train", train_X, train_Y)  # (B,N,2,L)

    # valid
    valid_M = np.repeat(valid_M, 2, axis=-1)
    valid_M = valid_M.transpose(0, 1, 3, 2)  # (B,N,2,L)
    valid_X, valid_Y = add_pos_emb("valid", valid_X, valid_Y)  # (B,N,2,L)

    # test
    test_M = np.repeat(test_M, 2, axis=-1)
    test_M = test_M.transpose(0, 1, 3, 2)  # (B,N,2,L)
    test_X, test_Y = add_pos_emb("test", test_X, test_Y)  # (B,N,2,L)

    print(
        "train X: {} train M: {} train Y: {}".format(
            train_X.shape, train_M.shape, train_Y.shape
        )
    )
    print(
        "valid X: {} valid M: {} valid Y: {}".format(
            valid_X.shape, valid_M.shape, valid_Y.shape
        )
    )
    print(
        "test X: {} test M: {} test Y: {}".format(
            test_X.shape, test_M.shape, test_Y.shape
        )
    )

    train_dataset = TensorDataset(
        torch.from_numpy(train_X), torch.from_numpy(train_M), torch.from_numpy(train_Y)
    )

    valid_dataset = TensorDataset(
        torch.from_numpy(valid_X), torch.from_numpy(valid_M), torch.from_numpy(valid_Y)
    )

    test_dataset = TensorDataset(
        torch.from_numpy(test_X), torch.from_numpy(test_M), torch.from_numpy(test_Y)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=val_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

    return train_loader, valid_loader, test_loader, mean, std, A


class MissDataLoader(object):

    def __init__(
        self, xs, ms, ys, batch_size, pad_with_last_sample=True, shuffle=False
    ):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            m_padding = np.repeat(ms[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ms = np.concatenate([ms, m_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ms, ys = xs[permutation], ms[permutation], ys[permutation]
        self.xs = xs
        self.ms = ms
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                m_i = self.ms[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, m_i, y_i)
                self.current_ind += 1

        return _wrapper()

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return torch.Tensor(self.xs[idx]), (
            torch.Tensor(self.ys[idx]),
            torch.Tensor(self.ms[idx]),
        )


def load_missdataset(
    dataset, miss_mechanism, miss_ratio, seqlen, batch_size, test_batch_size
):
    data = {}

    (
        train_X,
        train_M,
        train_Y,
        valid_X,
        valid_M,
        valid_Y,
        test_X,
        test_M,
        test_Y,
        A,
        mean,
        std,
    ) = load_data(dataset, miss_mechanism, miss_ratio, seqlen)

    train_X = train_X.transpose(0, 2, 1, 3)  # (B,L,N,1)
    train_M = train_M.transpose(0, 2, 1, 3)  # (B,L,N,1)
    train_Y = train_Y.transpose(0, 2, 1, 3)  # (B,L,N,1)

    valid_X = valid_X.transpose(0, 2, 1, 3)  # (B,L,N,1)
    valid_M = valid_M.transpose(0, 2, 1, 3)  # (B,L,N,1)
    valid_Y = valid_Y.transpose(0, 2, 1, 3)  # (B,L,N,1)

    test_X = test_X.transpose(0, 2, 1, 3)  # (B,L,N,1)
    test_M = test_M.transpose(0, 2, 1, 3)  # (B,L,N,1)
    test_Y = test_Y.transpose(0, 2, 1, 3)  # (B,L,N,1)

    scaler = StandardScaler(mean, std)
    data["m_train"] = train_M
    data["m_val"] = valid_M
    data["m_test"] = test_M
    data["y_test"] = test_Y
    data["train_loader"] = MissDataLoader(
        train_X, train_M, train_Y, batch_size, pad_with_last_sample=False, shuffle=True
    )
    data["val_loader"] = MissDataLoader(
        valid_X,
        valid_M,
        valid_Y,
        test_batch_size,
        pad_with_last_sample=False,
        shuffle=False,
    )
    data["test_loader"] = MissDataLoader(
        test_X,
        test_M,
        test_Y,
        test_batch_size,
        pad_with_last_sample=True,
        shuffle=False,
    )
    data["scaler"] = scaler
    return data
