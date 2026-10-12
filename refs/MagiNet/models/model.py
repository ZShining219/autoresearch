import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import configparser
from scipy.sparse.linalg import eigs
from utils import batch_cosine_similarity, batch_dot_similarity


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] -
                                cheb_polynomials[i - 2])

    return cheb_polynomials


class ScaledDotProductSAtt(nn.Module):

    def __init__(self, d_k):
        super(ScaledDotProductSAtt, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(
                attn_mask > 0.5, -1e9
            )  # Fills elements of self tensor with value where mask is True.
        return scores


class ScaledDotProductTAtt(nn.Module):

    def __init__(self, d_k):
        super(ScaledDotProductTAtt, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask, res_att):
        '''
        Q: [B, N, n_heads, len_q, d_k]
        K: [B, N, n_heads, len_k, d_k]
        V: [B, N, n_heads, len_v, d_v]
        attn_mask: [B, N, n_heads, len_v, len_v]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k) + res_att  # scores : [B, N, n_heads, len_v, len_v]
        if attn_mask is not None:
            scores.masked_fill_(
                attn_mask > 0.5, -1e9
            )  # Fills elements of self tensor with value where mask is True.
        attn = F.softmax(scores, dim=3)  # B,N,n_heads,T,T
        context = torch.matmul(attn, V)  # [B, N, n_heads, len_v, d_v]
        return context, scores


class Mask_Spatial_Attention(nn.Module):

    def __init__(self, DEVICE, d_model, d_k, d_v, n_heads, num_of_nodes):
        super(Mask_Spatial_Attention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_nodes = num_of_nodes
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)

    def forward(self, input_Q, input_K, attn_mask):
        '''
        input_Q: [B,L,N,F]
        input_K: [B,L,N,F]
        input_V: [B,L,N,F]
        attn_mask: [B,L,N,1]
        '''
        residual, batch_size, N = input_Q, input_Q.size(0), input_Q.size(2)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(
            batch_size, self.num_of_nodes, self.n_heads,
            self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(
            batch_size, self.num_of_nodes, self.n_heads,
            self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        # if attn_mask is not None:
        #     attn_mask = attn_mask.unsqueeze(1).expand(
        #         batch_size, self.num_of_nodes, self.n_heads, N,
        #         N)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        attn = ScaledDotProductSAtt(self.d_k)(Q, K, attn_mask)
        return attn


class Mask_Temporal_Attention(nn.Module):

    def __init__(self, DEVICE, d_model, d_k, d_v, n_heads, num_of_nodes):
        super(Mask_Temporal_Attention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_d = num_of_nodes
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask, res_att):
        '''
        input_Q: [B,N,L,F]
        input_K: [B,N,L,F]
        input_V: [B,N,L,F]
        attn_mask: [B,N,L,1]
        '''
        residual, batch_size, L = input_Q, input_Q.size(0), input_Q.size(2)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, self.num_of_d, -1, self.n_heads,
                                   self.d_k).transpose(
                                       2, 3)  # Q: [B,N,n_heads,L,d_k]
        K = self.W_K(input_K).view(batch_size, self.num_of_d, -1, self.n_heads,
                                   self.d_k).transpose(
                                       2, 3)  # K: [B,N,n_heads,L,d_k]
        V = self.W_V(input_V).view(batch_size, self.num_of_d, -1, self.n_heads,
                                   self.d_v).transpose(
                                       2, 3)  # V: [B,N,n_heads,L,d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(2).expand(
                batch_size, self.num_of_d, self.n_heads, L,
                L)  # attn_mask: [B,N,n_heads,L,L]
        # context: [B, N, n_heads, len_q, d_v], attn: [B, N, n_heads, len_q, len_k]
        context, res_attn = ScaledDotProductTAtt(self.d_k)(Q, K, V, attn_mask,
                                                           res_att)

        context = context.transpose(2, 3).reshape(
            batch_size, self.num_of_d, -1, self.n_heads *
            self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]

        return nn.LayerNorm(self.d_model).to(self.DEVICE)(output +
                                                          residual), res_attn


class Mask_Aware_Spatial_Temporal_Attention(nn.Module):

    def __init__(self, DEVICE, num_of_d, d_model, num_of_timesteps, K, d_k,
                 d_v, n_heads, num_of_nodes):
        super(Mask_Aware_Spatial_Temporal_Attention, self).__init__()
        self.MTA = Mask_Temporal_Attention(DEVICE, num_of_d, d_k, d_v, n_heads,
                                           num_of_nodes)
        self.MSA = Mask_Spatial_Attention(DEVICE, d_model, d_k, d_v, K,
                                          num_of_nodes)
        self.Aggr_node_emb = nn.Conv2d(num_of_timesteps,
                                       d_model,
                                       kernel_size=(1, num_of_d))
        self.SpatialPosEmb = SpatialPositionEmbedding(num_of_nodes, d_model,
                                                      num_of_d, DEVICE)
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, H, m, res_att):
        batch_size, num_of_nodes, num_of_features, num_of_timesteps = H.shape  # B,N,F,T
        H1 = H.permute(0, 1, 3, 2)  # B,N,T,F
        att_mask = m[:, :, :1, :].expand(batch_size, num_of_nodes, 1,
                                         num_of_timesteps)  # B,N,1,T
        att_mask = att_mask.permute(0, 1, 3, 2)  # B,N,T,1
        Y, Tatt = self.MTA(H1, H1, H1, att_mask,
                           res_att)  # B,N,T,F; B,N,Ht,T,T

        H_matt = self.Aggr_node_emb(Y.permute(0, 2, 1,
                                              3))[:, :, :,
                                                  -1].permute(0, 2,
                                                              1)  # B,N,d_model

        # adj_mgl = adj_mx # use underlying graph
        # missing spatial graph learning
        adj_mgl = (F.softmax(F.relu(
            torch.matmul(H_matt, H_matt.transpose(-1, -2))),
                             dim=2) > 0.5).float()  # B,N,N

        # Mask spatial attention
        H2 = self.SpatialPosEmb(H_matt, batch_size)  # B,N,d_model
        SpatialPosEmb_H = self.dropout(H2)  # B,N,d_model
        MASTatt = self.MSA(SpatialPosEmb_H, SpatialPosEmb_H, None)  # B,Hs,N,N
        return MASTatt, Tatt, adj_mgl


class Mask_Aware_Spatial_Conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels,
                 num_of_nodes):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(Mask_Aware_Spatial_Conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.relu = nn.ReLU(inplace=True)
        self.Theta = nn.ParameterList([
            nn.Parameter(
                torch.FloatTensor(in_channels, out_channels).to(self.DEVICE))
            for _ in range(K)
        ])
        self.mask = nn.ParameterList([
            nn.Parameter(
                torch.FloatTensor(num_of_nodes, num_of_nodes).to(self.DEVICE))
            for _ in range(K)
        ])

    def forward(self, x, spatial_attention, adj_mgl):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :param spatial_attention: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_nodes, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size,
                                 num_of_nodes, self.out_channels).to(
                                     self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)
                mask = self.mask[k]

                myspatial_attention = spatial_attention[:,
                                                        k, :, :] + adj_mgl.mul(
                                                            mask)
                myspatial_attention = F.softmax(myspatial_attention, dim=1)

                T_k_with_at = T_k.mul(
                    myspatial_attention)  # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(
                    graph_signal
                )  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(
                    theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return self.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class SpatialPositionEmbedding(nn.Module):

    def __init__(self, nb_seq, d_Em, num_of_features, DEVICE):
        super(SpatialPositionEmbedding, self).__init__()
        self.nb_seq = nb_seq
        self.num_of_features = num_of_features
        self.pos_embed = nn.Embedding(nb_seq, d_Em)
        self.norm = nn.LayerNorm(d_Em)
        self.DEVICE = DEVICE

    def forward(self, x, batch_size):
        pos = torch.arange(self.nb_seq, dtype=torch.long).to(self.DEVICE)
        pos = pos.unsqueeze(0).expand(batch_size, self.nb_seq)
        embedding = x + self.pos_embed(pos)
        Emx = self.norm(embedding)
        return Emx


class GTU(nn.Module):

    def __init__(self, in_channels, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels,
                                 2 * in_channels,
                                 kernel_size=(1, kernel_size),
                                 stride=(1, 1))

    def forward(self, x):
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, :self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x_gtu


class ST_Block(nn.Module):

    def __init__(self, DEVICE, num_of_d, in_channels, K, nb_chev_filter,
                 nb_time_filter, cheb_polynomials, adj_mx, num_of_nodes,
                 num_of_timesteps, d_model, d_k, d_v, n_heads):
        super(ST_Block, self).__init__()
        self.DEVICE = DEVICE
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.num_of_nodes = num_of_nodes
        self.num_of_timesteps = num_of_timesteps

        # self.pre_conv = nn.Conv2d(num_of_timesteps,
        #                           d_model,
        #                           kernel_size=(1, num_of_d))

        # self.EmbedT = SpatialPositionEmbedding(num_of_timesteps, num_of_nodes,
        #                                 num_of_d, 'T', DEVICE)
        # self.EmbedS = SpatialPositionEmbedding(num_of_nodes, d_model, num_of_d,
        #                                        'S', DEVICE)

        # self.MTA = Mask_Temporal_Attention(DEVICE, num_of_d, d_k, d_v, n_heads,
        #                                    num_of_nodes)
        # self.MSA = Mask_Spatial_Attention(DEVICE, d_model, d_k, d_v, K,
        #                                   num_of_nodes)

        self.MASTatt = Mask_Aware_Spatial_Temporal_Attention(
            DEVICE, num_of_d, d_model, num_of_timesteps, K, d_k, d_v, n_heads,
            num_of_nodes)

        self.MASconv = Mask_Aware_Spatial_Conv(K, cheb_polynomials,
                                               nb_chev_filter, nb_chev_filter,
                                               num_of_nodes)
        self.gtu3 = GTU(nb_time_filter, 3)
        self.gtu5 = GTU(nb_time_filter, 5)
        self.gtu7 = GTU(nb_time_filter, 7)
        self.pooling = torch.nn.MaxPool2d(kernel_size=(1, 2),
                                          stride=None,
                                          padding=0,
                                          return_indices=False,
                                          ceil_mode=False)

        self.residual_conv = nn.Conv2d(in_channels,
                                       nb_time_filter,
                                       kernel_size=(1, 1),
                                       stride=(1, 1))

        self.dropout = nn.Dropout(p=0.05)
        self.fcmy = nn.Sequential(
            nn.Linear(3 * num_of_timesteps - 12, num_of_timesteps),
            nn.Dropout(0.05),
        )
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, H, m, res_att):
        '''
        :param H: (Batch_size, N, F_in, T)
        :param res_att: (Batch_size, N, F_in, T)
        :return: (Batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_nodes, num_of_features, num_of_timesteps = H.shape  # B,N,F,T

        # # Mask temporal attention
        # TEmx = H.permute(0, 1, 3, 2)  # B,N,T,F
        # att_mask = m[:, :, :1, :].expand(batch_size, num_of_nodes, 1,
        #                                  num_of_timesteps)  # B,N,1,T
        # att_mask = att_mask.permute(0, 1, 3, 2)  # B,N,T,1
        # TATout, re_At = self.MTA(TEmx, TEmx, TEmx, att_mask,
        #                          res_att)  # B,N,T,F; B,N,Ht,T,T

        # x_TAt = self.pre_conv(TATout.permute(0, 2, 1,
        #                                      3))[:, :, :,
        #                                          -1].permute(0, 2,
        #                                                      1)  # B,N,d_model

        # # adj_mgl = adj_mx # use underlying graph
        # # missing spatial graph learning
        # adj_mgl = (F.softmax(F.relu(
        #     torch.matmul(x_TAt, x_TAt.transpose(-1, -2))),
        #                      dim=2) > 0.5).float()

        # # Mask spatial attention
        # SEmx_TAt = self.EmbedS(x_TAt, batch_size)  # B,N,d_model
        # SEmx_TAt = self.dropout(SEmx_TAt)  # B,N,d_model
        # STAt = self.MSA(SEmx_TAt, SEmx_TAt, None)  # B,Hs,N,N

        MASTatt, T_att, adj_mgl = self.MASTatt(H, m, res_att)

        # graph convolution in spatial dim
        spatial_gcn = self.MASconv(H, MASTatt, adj_mgl)  # B,N,F,T，这里F=32

        # convolution along the time axis
        X = spatial_gcn.permute(0, 2, 1, 3)  # B,F,N,T，这里F=32
        x_gtu = []
        x_gtu.append(self.gtu3(X))  # B,F,N,T-2
        x_gtu.append(self.gtu5(X))  # B,F,N,T-4
        if self.num_of_timesteps == 12:
            x_gtu.append(self.gtu7(X))  # B,F,N,T-6
        time_conv = torch.cat(x_gtu, dim=-1)  # B,F,N,3T-12
        time_conv = self.fcmy(time_conv)  # B,F,N,T

        if num_of_features == 1:
            time_conv_output = self.relu(time_conv)
        else:
            time_conv_output = self.relu(X + time_conv)  # B,F,N,T

        # residual shortcut
        if num_of_features == 1:
            x_residual = self.residual_conv(H.permute(0, 2, 1, 3))
        else:
            x_residual = H.permute(0, 2, 1, 3)
        x_residual = self.ln(
            F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(
                0, 2, 3, 1)  # B,N,F,T，这里F为32

        return x_residual, T_att


class MagiNet_submodule(nn.Module):

    def __init__(self, DEVICE, st_block, in_channels, K, nb_chev_filter,
                 nb_time_filter, cheb_polynomials, adj_mx, num_of_timesteps,
                 num_of_nodes, d_model, d_k, d_v, n_heads):
        '''
        :param st_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param cheb_polynomials:
        '''

        super(MagiNet_submodule, self).__init__()

        self.BlockList = nn.ModuleList([
            ST_Block(DEVICE, nb_time_filter, in_channels, K, nb_chev_filter,
                     nb_time_filter, cheb_polynomials, adj_mx, num_of_nodes,
                     num_of_timesteps, d_model, d_k, d_v, n_heads)
        ])

        self.BlockList.extend([
            ST_Block(DEVICE, nb_time_filter, nb_chev_filter, K, nb_chev_filter,
                     nb_time_filter, cheb_polynomials, adj_mx, num_of_nodes,
                     num_of_timesteps, d_model, d_k, d_v, n_heads)
            for _ in range(st_block - 1)
        ])

        self.final_conv = nn.Conv2d(int((num_of_timesteps) * st_block),
                                    128,
                                    kernel_size=(1, nb_time_filter))
        self.final_fc = nn.Linear(256 * 2, num_of_timesteps)
        self.fc_hid = nn.Sequential(
            nn.Conv2d(nb_time_filter, 128, (1, 1), (1, 1)),
            nn.Linear(num_of_timesteps, 1))
        self.DEVICE = DEVICE

        self.rnn = nn.RNN(256, 256, 1, bidirectional=True)

        self.to(DEVICE)

    def forward(self, H, m, hidden_states):
        '''
        :param H: (B, N_nodes, F_in, T_in)
        :param hidden_states: (B, N_nodes, D_dim, T_in)
        :return: (B, N_nodes, T_out)
        '''
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        need_concat = []
        res_attT = 0
        for block in self.BlockList:
            H, res_attT = block(H, m, res_attT)
            need_concat.append(H)

        final_x = torch.cat(need_concat, dim=-1)  # B,N,F,T*4
        output1 = self.final_conv(final_x.permute(0, 3, 1,
                                                  2))[:, :, :,
                                                      -1].permute(0, 2,
                                                                  1)  # B,N,128
        hidden_states = self.fc_hid(hidden_states.permute(
            0, 2, 1, 3))[:, :, :, -1].permute(0, 2, 1)  # B,N,128

        output1 = self.rnn(torch.cat((output1, hidden_states), dim=2))[0]

        output = self.final_fc(output1).unsqueeze(-2)  # B,N,1,L
        return output


def make_model(DEVICE, st_block, in_channels, K, nb_chev_filter,
               nb_time_filter, adj_mx, num_of_timesteps, num_of_nodes, d_model,
               d_k, d_v, n_heads):
    '''
    :param DEVICE: cuda device
    :param st_block: number of ST blocks
    :param in_channels: input channels
    :param K: K-th order of Cheb Convolution
    :param nb_chev_filter: hidden dimension of cheb conv
    :param nb_time_filter: hidden dimension of temporal conv
    :param adj_mx: adjacency matrix
    :param num_of_timesteps: seqlen
    :param num_of_nodes: number of nodes
    :param d_model: hidden size
    :param d_k: hidden dimension of key in attention calculation
    :param d_v: hidden dimension of value in attention calculation
    :return:
    '''
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [
        torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE)
        for i in cheb_polynomial(L_tilde, K)
    ]
    model = MagiNet_submodule(DEVICE, st_block, in_channels, K, nb_chev_filter,
                              nb_time_filter, cheb_polynomials, adj_mx,
                              num_of_timesteps, num_of_nodes, d_model, d_k,
                              d_v, n_heads)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model