# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import igraph as ig
from evaluate import *
import sys
import time
import omega_index

start = time.perf_counter()

base_path = str(sys.argv[1])
# markov_time = str(sys.argv[2])
markov_time = 1
# cishu = int(sys.argv[3])
cishu = 1

gml_file_path = base_path + '\\network.gml'
community_file_path = base_path + '\\community_v2.txt'
rcam_file = base_path + '\\RCAM.txt'
g = ig.Graph.Read_GML(gml_file_path)
num_node = len(g.vs)
adj_m = g.get_adjacency(type=2, eids=False)
adj = []
for row in adj_m:
    for item in row:
        adj.append(item)
adj = np.mat(np.array(adj).reshape(num_node, num_node))
RCAM, num_community = getRCAM_and_CN(num_node, community_file_path, rcam_file)
num_hidden = 128
threshold = np.sqrt(-np.log(1 - (2 * num_node / (num_node * (num_node - 1)))))
lr = 0.01
dropout_on = True
weight_decay_on = True
nmi_result_file = base_path + '\\NMI_CDMSG_MT_' + str(markov_time) + '.txt'

def m_pow(m, t):
    m_p = torch.eye(len(m))
    for i in range(int(t)):
        m_p = torch.mm(m_p, m)
    return m_p


def pre_loss_func():
    t = markov_time
    A = torch.from_numpy(adj).float()  # 邻接矩阵
    # np.savetxt('A.txt', A.detach().numpy())
    m = torch.sum(A) / 2
    d = torch.sum(A, 1)
    pai = d / (2 * m.item())
    L = torch.diag(pai)  # pai的对角阵
    D = torch.diag(d).float()
    D_ = torch.inverse(D)
    M = torch.mm(D_, A)
    firstmd = torch.mm(L, m_pow(M, t))
    return A, m, d, pai, L, D, D_, M, firstmd


A, m, d, pai, L, D, D_, M, firstmd = pre_loss_func()


def norm(adj):
    adj += np.eye(adj.shape[0])
    degree = np.array(adj.sum(1))
    degree = np.diag(np.power(degree, -0.5))
    adj_n = degree.dot(adj).dot(degree)
    adj_n = (torch.from_numpy(adj_n)).float()
    return adj_n


class GCNLayer(nn.Module):
    def __init__(self, infeature_size, outfeature_size):
        super(GCNLayer, self).__init__()
        self.weight = Parameter(torch.FloatTensor(infeature_size, outfeature_size))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(norm(adj), support)
        if active:
            # output = F.relu(output)
            output = F.relu(output) + 0.0001
        return output


class GCN(nn.Module):
    def __init__(self, input_size, output_size, num_community):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(input_size, output_size)
        self.dropout = torch.nn.Dropout(0.5)
        self.gcn2 = GCNLayer(output_size, num_community)

    def forward(self, features, adj):
        h = self.gcn1(features, adj)
        do_h = self.dropout(h)
        o = self.gcn2(do_h, adj)
        return o

def regularize(H, k, n):
    res = np.sqrt(k) / n
    res = res * torch.norm(torch.sum(H, dim = 0)) - 1
    return res


def loss_func(H, pai):
    # A = torch.from_numpy(adj).float()  # 邻接矩阵
    # # np.savetxt('A.txt', A.detach().numpy())
    # m = torch.sum(A) / 2
    # d = torch.sum(A, 1)
    # pai = d / (2 * m.item())
    # L = torch.diag(pai)  # pai的对角阵
    # D = torch.diag(d).float()
    # D_ = torch.inverse(D)
    # M = torch.mm(D_, A)
    # firstmd = torch.mm(L, m_pow(M, t))
    pai = pai.reshape(1, len(pai))
    secondmd = torch.mm(pai.T, pai)
    med = firstmd - secondmd
    R = torch.mm(torch.mm(H.t(), med), H)
    loss = -torch.trace(R)
    # loss = -F.relu(torch.trace(R))
    # regularization = regularize(H, len(H[0]), len(H))
    # loss = loss + regularization
    return loss


def norm(H):
    return (H.T / H.sum(axis=1)).T
def stop(prev_losses, cur_loss):
    if(len(prev_losses) < 10):
        return False
    if(np.mean(prev_losses[len(prev_losses) : len(prev_losses) - 4 : -1]) >= cur_loss):
        return True
    return False

def f1(ground_truth_m, embed_m):
    n = (ground_truth_m.dot(embed_m.T)).astype(float)  # cg * cd
    p = n / np.array(embed_m.sum(axis=1)).clip(min=1).reshape(-1)
    r = n / np.array(ground_truth_m.sum(axis=1)).clip(min=1).reshape(-1, 1)
    f1 = 2 * p * r / (p + r).clip(min=1e-10)

    f1_s1 = f1.max(axis=1).mean()
    f1_s2 = f1.max(axis=0).mean()
    f1_s = (f1_s1 + f1_s2) / 2
    return f1_s


def changeCfRtoD(_CAM):
    communities = {}
    c_index = 0
    for c in _CAM.T:
        n_index = 0
        temp = []
        for at in c:
            if(at == 1):
                temp.append(n_index)
            n_index = n_index + 1
        communities[c_index] = temp
        c_index = c_index + 1
    return communities

def run():
    model = GCN(num_node, num_hidden, num_community)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    adj_torch = torch.from_numpy(adj).float()
    prev_losses = []
    prev_nmi_score = 0
    prev_running_time = 0
    epoch = 0
    label = False
    while True:
        H = model(adj_torch, adj_torch)
        H = norm(H)
        loss = loss_func(H, pai)
        optimizer.zero_grad()
        loss.backward()
        PCAM = getPredictCAM(H, threshold)
        np.savetxt(base_path + "\\CDMSG_CAM_"+ "MT_" + str(markov_time) + "_CS_" + str(cishu) + ".txt", PCAM.detach().numpy())
        # print(RCAM.shape, PCAM.shape)
        nmi_score = overlapping_nmi(RCAM.float(), PCAM.float())
        # omega_score = omega_index.Omega(changeCfRtoD(PCAM.detach().numpy()), changeCfRtoD(RCAM.numpy()))
        runningtime = time.perf_counter()-start
        label = stop(prev_losses, np.abs(loss.item()))
        if(label):
            break;
        else:
            print('Epoch:{}  Loss:{}  NMI:{}  Time:{}'.format(str(epoch), str(loss.item()), str(nmi_score.item()), str(runningtime)))
        prev_losses.append(np.abs(loss.item()))
        prev_running_time = runningtime
        prev_nmi_score = str(nmi_score.item()) + " Time:" + str(runningtime)
        # 保存结果
        optimizer.step()
        epoch = epoch + 1
    save_nmi(nmi_result_file, prev_nmi_score)
    omega_score = omega_index.Omega(changeCfRtoD(PCAM.detach().numpy()), changeCfRtoD(RCAM.numpy()))
    print("End\nOmega Index: " + str(omega_score))
    return

if __name__ == '__main__':
    run()
