import matplotlib.pyplot as plt
import numpy as np
import os

# パスが無かったら作成する関数
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 実行時のパラメータを保存する関数
def save_simulation_param(path, const):
    with open(f"{path}/simulation_parameter.txt", mode="w") as f:
        params = ["N", "epsilon", "sigma", "a", "phi", "start_time", "finish_time", "step_width"]
        for param in params:
            f.write(f"{param}\t\t: {eval('const.' + param)}\n")

# 使用したネットワークの情報を保存する関数
def save_network_param(path, iteration,const, fhn):
    with open(f"{path}/{iteration}_network_parameter.txt", mode="w") as f:
        params1 = ["network_name", "k", "p"]
        params2 = ["clustering_coeff", "shortest_path_length", "S"]
        for param in params1:
            f.write(f"{param}\t\t: {eval('const.' + param)}\n")
        for param in params2:
            f.write(f"{param}\t\t: {eval('fhn.' + param)}\n")

# ネットワーク構造を保存する関数
def save_network_structure(path, iteration, A):
    np.savetxt(f"{path}/{iteration}_network_structure.txt", A)

# 解を保存する関数 ※視認性を挙げるために転置しています, N行t列 -> t行N列
def save_solution(path, iteration, u_sol, v_sol):
    np.savetxt(f"{path}/{iteration}_u_sol.txt", u_sol.T)
    np.savetxt(f"{path}/{iteration}_v_sol.txt", v_sol.T)

# 同期度をプロットする関数
def plot_synchro(path, iteration, t, r_values, const):
    plt.figure(figsize=(10, 8))
    plt.plot(t, r_values, linewidth=0.05)
    plt.ylim(0.0, 1.0)
    plt.title('Time evolution of the global Kuramoto order parameter r(t)')
    plt.xlabel('Time')
    plt.ylabel('r(t)')
    plt.grid(True)
    plt.savefig(f'{path}/{iteration}_synchronization_{const.start_time}-{const.finish_time}-{const.step_width}.png')

# 膜電位uをプロットする関数
def plot_u(path, iteration, t, u_sol, const):
    plt.figure(figsize=(10, 6))
    plt.plot(t, u_sol.T)
    plt.title('Membrane potentials of the FHN oscillators')
    plt.xlabel('Time')
    plt.ylabel('u_k')
    plt.savefig(f'{path}/{iteration}_u_{const.start_time}-{const.finish_time}-{const.step_width}.png')

# 隣接行列Aを可視化する関数
def plot_adjacency_matrix(path, iteration, A, const):
    plt.figure(figsize=(8, 8))
    #plt.imshow(A, cmap='YlOrRd', interpolation='none', norm=LogNorm())
    plt.imshow(A, cmap='YlOrRd', interpolation='none')
    plt.colorbar(label='Link Strength')
    plt.title('Adjacency Matrix')
    plt.xlabel('Node index')
    plt.ylabel('Node index')
    plt.savefig(f'{path}/{iteration}_adjacency-matrix_{const.start_time}-{const.finish_time}-{const.step_width}.png')
