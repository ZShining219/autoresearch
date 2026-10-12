"""
Description: 
Author: Jianping Zhou
Email: jianpingzhou0927@gmail.com
Date: 2023-08-23 17:56:01
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
import pickle as pk
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import argparse
import wandb
import math
import time
import yaml
from utils import *
from models.model import *
from load_data import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", type=str, default="configs/METR-LA.yaml", help="config filepath"
)
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--learnable", type=int, default=1, help="1: learnable; 0: fixed")
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)


class Learnable_Missing_Encoding(nn.Module):

    def __init__(self, hidden_dim):
        super(Learnable_Missing_Encoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim, 1))
        nn.init.uniform_(self.mask_token, -0.02, 0.02)

    def forward(self, input_emb, m):
        B, N, D, L = input_emb.shape  # B,N,D,L

        mask = m[:, :, :1, :].expand(B, N, D, L)

        observed_token = input_emb * mask

        missed_token = self.mask_token.expand(B, N, D, L) * (1 - mask)

        learnable_input_emb = observed_token + missed_token
        learnable_input_emb = learnable_input_emb.transpose(-1, -2)  # (B,N,L,D)
        return learnable_input_emb


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=0.1)

    def getPE(self, x):
        return self.pe[:, : x.size(1)]

    def forward(self, input_emb, position):
        B, N, L, D = input_emb.shape  # B,N,L,D

        # position emb
        pe = self.getPE(position.view(B * N, L, -1).long())  # (B*N,L,1,D)
        input_emb = input_emb.view(B * N, L, -1) + pe
        input_emb = self.dropout(input_emb.view(B, N, L, D))  # (B,N,L,D)
        return input_emb


class Temporal_Positional_Embedding(nn.Module):

    def __init__(self, hidden_dim, max_len=1000, dropout=0.1):
        super(Temporal_Positional_Embedding, self).__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.pe = nn.Parameter(torch.empty(max_len, hidden_dim), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, input_emb, position):
        B, N, L, D = input_emb.shape  # B,N,L,D

        # position emb
        pe = self.pe[position.view(B * N, L, -1).long(), :]  # (B*N,L,1,D)
        learnable_pos_emb = input_emb + pe.view(B, N, L, -1)
        learnable_pos_emb = self.dropout(learnable_pos_emb)  # (B,N,L,D)
        return learnable_pos_emb


class Adaptive_Missing_Spatial_Temporal_Encoder(nn.Module):

    def __init__(
        self,
        in_channel,
        hidden_dim,
        learnable=True,
        pe_learnable=True,
    ):
        super(Adaptive_Missing_Spatial_Temporal_Encoder, self).__init__()
        self.learnable = learnable
        self.pe_learnable = pe_learnable
        self.in_channel = in_channel
        self.hidden_dim = hidden_dim

        self.observation_encoding = nn.Conv2d(
            in_channel, hidden_dim, kernel_size=(1, 1), stride=(1, 1)
        )
        self.maskemb = Learnable_Missing_Encoding(hidden_dim)
        if self.learnable:
            self.posemb = Temporal_Positional_Embedding(hidden_dim)
        else:
            self.posemb = PositionalEmbedding(hidden_dim)

    def forward(self, x, m):
        if self.pe_learnable:
            position = x[:, :, 1, :].unsqueeze(-2)
        x = x[:, :, : self.in_channel, :]
        B, N, F_in, L = x.shape  # B,N,F,L

        input = x.unsqueeze(-1)  # B, N, F, L, 1
        input = input.reshape(B * N, F_in, L, 1)  # B*N, F, L, 1

        # learnable missing encoding
        input_emb = self.observation_encoding(input)  # B*N,  d, L, 1
        input_emb = input_emb.squeeze(-1).view(B, N, self.hidden_dim, -1)  # B,N,d,L
        Learnable_Missing_Encoding = self.maskemb(input_emb, m)  # B,N,L,D

        # temporal positional embedding
        H = self.posemb(
            Learnable_Missing_Encoding, position.view(B * N, L, -1).long()
        )  # B,N,L,D
        return H


class MagiNet(nn.Module):

    def __init__(
        self,
        device,
        num_nodes,
        seqlen,
        in_channels,
        hidden_dim,
        st_block,
        K,
        n_heads,
        d_model,
        adj_mx,
        learnable=True,
        pe_learnable=True,
    ):
        super(MagiNet, self).__init__()

        self.num_nodes = num_nodes
        self.seqlen = seqlen
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        self.AMSTenc = Adaptive_Missing_Spatial_Temporal_Encoder(
            in_channels, hidden_dim, learnable=learnable, pe_learnable=pe_learnable
        )

        self.MASTdec = make_model(
            DEVICE=device,
            st_block=st_block,
            in_channels=in_channels,
            K=K,
            nb_chev_filter=hidden_dim,
            nb_time_filter=hidden_dim,
            adj_mx=adj_mx,
            num_of_timesteps=seqlen,
            num_of_nodes=num_nodes,
            d_model=d_model,
            d_k=32,
            d_v=32,
            n_heads=n_heads,
        )

    def forward(self, x, m):

        # AMSTenc
        H = self.AMSTenc(x, m)

        # MASTdec
        output = self.MASTdec(H.transpose(-1, -2), m, H.transpose(-1, -2))

        return output  # (B,N,1,L)


def predict(
    model,
    device,
    best_save_path,
    result_path,
    test_loader,
    mean,
    std,
    dataset,
    miss_ratio,
    miss_mechanism,
    seed,
):
    model.load_state_dict(torch.load(best_save_path))
    # test
    test_maes, test_rmses, test_mapes = [], [], []
    model.eval()
    miss_data = []
    predict_results = []
    groundtruths = []
    with torch.no_grad():
        for _, (x, m, y) in enumerate(test_loader):
            x = x.to(device)  # (B,N,2,L)
            m = m.to(device)  # (B,N,2,L)
            y = y[:, :, :1, :].detach().cpu().numpy()
            x_hat = model(x, m).detach().cpu().numpy()
            unnorm_x = unnormalization(x[:, :, :1, :].detach().cpu().numpy(), mean, std)
            unnorm_x_hat = unnormalization(x_hat, mean, std)
            unnorm_y = unnormalization(y, mean, std)
            mask = m.detach().cpu().numpy()
            mae, rmse, mape = missed_eval_np(unnorm_x_hat, unnorm_y, mask)
            predict_data = (
                unnorm_x_hat * (1 - mask[:, :, :1, :]) + unnorm_x * mask[:, :, :1, :]
            )
            unnorm_x = np.where(mask[:, :, :1, :] == 0, np.nan, unnorm_x)
            miss_data.append(unnorm_x)
            predict_results.append(predict_data)
            groundtruths.append(unnorm_y)
            test_maes.append(mae)
            test_rmses.append(rmse)
            test_mapes.append(mape)
        test_mae = np.mean(test_maes)
        test_rmse = np.mean(test_rmses)
        test_mape = np.mean(test_mapes)
    print(
        "Test result: MAE {} RMSE {} MAPE {}".format(
            test_mae, test_rmse, test_mape * 100
        )
    )
    result = {}
    result["missed_data"] = np.concatenate(miss_data, axis=0)  # B,N,1,L
    result["imputed_data"] = np.concatenate(predict_results, axis=0)  # B,N,1,L
    result["groundtruth"] = np.concatenate(groundtruths, axis=0)  # B,N,1,L
    print(result["missed_data"].shape)
    print(result["imputed_data"].shape)
    print(result["groundtruth"].shape)
    result_path = result_path + "{}/".format(dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(
        result_path
        + "result_{}_ms{}_seed{}.pkl".format(miss_mechanism, miss_ratio, seed),
        "wb",
    ) as fb:
        pk.dump(result, fb)


def main(args):
    if args.learnable == 1:
        learnable = True
    else:
        learnable = False
    config_filename = args.config_path
    with open(config_filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    dataset = config["data"]["dataset"]
    miss_mechanism = config["data"]["miss_mechanism"]
    # miss_pattern = config["data"]["miss_pattern"]
    miss_ratio = float(config["data"]["miss_ratio"])
    batch_size = int(config["data"]["batch_size"])
    val_batch_size = int(config["data"]["val_batch_size"])
    test_batch_size = int(config["data"]["test_batch_size"])
    seqlen = int(config["model"]["seqlen"])
    num_nodes = int(config["model"]["num_nodes"])
    st_block = int(config["model"]["st_block"])
    in_channel = int(config["model"]["in_channel"])
    hidden_size = int(config["model"]["hidden_size"])
    K = int(config["model"]["K"])
    d_model = int(config["model"]["d_model"])
    n_heads = int(config["model"]["n_heads"])
    epochs = int(config["train"]["epochs"])
    lr = float(config["train"]["lr"])
    save_path = config["train"]["save_model_path"]
    result_path = config["train"]["result_path"]
    seed = args.seed

    device = torch.device(
        "cuda:{}".format(int(config["train"]["cuda"]))
        if torch.cuda.is_available()
        else "cpu"
    )

    seed_torch(seed)
    torch.set_num_threads(10)

    train_loader, valid_loader, test_loader, mean, std, A = generate_miss_loader(
        dataset,
        miss_mechanism,
        # miss_pattern,
        miss_ratio,
        seqlen,
        batch_size,
        val_batch_size,
        test_batch_size,
    )

    # wandb.init(
    #     project="MagiNet",
    #     name="{}_lr{}_hiddensize{}_batchsize{}_seed{}".format(
    #         dataset, lr, hidden_size, batch_size, seed
    #     ),
    # )

    adj_mx = weight_matrix(A)

    model = MagiNet(
        device=device,
        num_nodes=num_nodes,
        seqlen=seqlen,
        in_channels=in_channel,
        hidden_dim=hidden_size,
        st_block=st_block,
        K=K,
        d_model=d_model,
        n_heads=n_heads,
        adj_mx=adj_mx,
        learnable=learnable,
    ).to(device)
    loss_function = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    patience = 0
    best_val_mae = 999
    for epoch in range(epochs):
        epoch_time = time.time()
        # train
        loss_epoch = []
        model.train()
        for _, (x, m, y) in enumerate(train_loader):
            x = x.to(device)  # (B,N,2,L)
            m = m.to(device)  # (B,N,2,L)
            y = y.to(device)  # (B,N,2,L)
            x_hat = model(x, m)
            loss = loss_function(x_hat, y[:, :, :1, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())

        # valid
        valid_maes, valid_rmses, valid_mapes = [], [], []
        model.eval()
        with torch.no_grad():
            for _, (x, m, y) in enumerate(valid_loader):
                x = x.to(device)  # (B,N,2,L)
                m = m.to(device)  # (B,N,2,L)
                y = y[:, :, :1, :].detach().cpu().numpy()
                x_hat = model(x, m).detach().cpu().numpy()
                unnorm_x_hat = unnormalization(x_hat, mean, std)
                unnorm_y = unnormalization(y, mean, std)
                mask = m.detach().cpu().numpy()
                mae, rmse, mape = missed_eval_np(unnorm_x_hat, unnorm_y, mask)
                valid_maes.append(mae)
                valid_rmses.append(rmse)
                valid_mapes.append(mape)
            valid_mae = np.mean(valid_maes)
            valid_rmse = np.mean(valid_rmses)
            valid_mape = np.mean(valid_mapes)
            if valid_mae < best_val_mae:
                patience = 0
                best_val_mae = valid_mae
                if not os.path.exists(save_path + "{}".format(dataset)):
                    os.makedirs(save_path + "{}".format(dataset))
                best_save_path = (
                    save_path
                    + "{}".format(dataset)
                    + "/best_model_{}_ms{}_seed{}.pth".format(
                        miss_mechanism, miss_ratio, seed
                    )
                )
                torch.save(model.state_dict(), best_save_path)
            # else:
            #     patience += 1
            #     if patience > 20:
            #         print("Early Stop!")
            #         break
        # wandb.log({"train loss": loss.item(), "valid loss": valid_mae})
        print(
            "Epoch [{}/{}] : ".format(epoch, epochs),
            "loss = ",
            np.mean(loss_epoch),
            "epoch_time: {}".format(time.time() - epoch_time),
        )
    # wandb.finish()

    # test
    predict(
        model,
        device,
        best_save_path,
        result_path,
        test_loader,
        mean,
        std,
        dataset,
        miss_ratio,
        miss_mechanism,
        seed,
    )


if __name__ == "__main__":
    start_time = time.time()
    main(args)
    print("Spend Time: {}".format(time.time() - start_time))
