import numpy as np
import networkx as nx

# 動的位相の計算
def calc_dynamical_phases(u, v):
    print(f"7. {np.random.random()}")
    path_lukas="Transformation_geom/transf_a0.50.dat"
    lukas=np.loadtxt(path_lukas, delimiter=' ')
    x=lukas[:,0]
    y=lukas[:,1]
    z=np.polyfit(x,y,5)
    p=np.poly1d(z)
    dynamical_phases = p(np.arctan2(v[:][:],u[:][:]))

    return dynamical_phases

# 同期度rの計算
def calc_r(dynamical_phases, N, t):
    print(f"8. {np.random.random()}")
    r = np.array([np.abs(np.sum(np.exp(1j * dynamical_phases[:, i]))) / N for i in range(len(t))])
    return r


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