import numpy as np
import networkx as nx
from model.const import Const

# 位相の計算
def calculate_phases(u, v):
    return np.arctan2(v, u)

# 同期度rの計算
def calculate_r(phases, N):
    return np.abs(np.sum(np.exp(1j * phases)) / N)

# 平均ノード次数 or 平均ノード強度を計算する関数
def calc_avg_node_strength(A, N):
    avg_node_strength = 0.0
    for i in range(N):
        node_i_strength = A[i].sum()
        avg_node_strength += node_i_strength
    avg_node_strength /= N
    return avg_node_strength

# クラスタリング係数を計算する関数
def calc_clustering_coeff(G, weight):
    return nx.average_clustering(G, weight=weight)

# 平均最短経路長を計算する関数
def calc_shortest_path_length(G):
    return nx.average_shortest_path_length(G)

# 高同期の割合を計算する関数
def calc_high_synchro_rate(r_log, threshold):
    high_synchro = np.sum(r_log > threshold)
    return (high_synchro / r_log.size) * 100