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
def save_network_param(path, iteration, const, all_clus_coeff, all_shortest_len, all_S, all_link):
    with open(f"{path}/network_parameter.txt", mode="w") as f:
        f.write(f"network_name\t\t: {const.network_name}\n")
        if const.network_name == "ws-network":
            f.write(f"k\t\t: {const.k}\n")
            f.write(f"p\t\t: {const.p}\n")
        f.write(f"all_clustering_coeff\t\t: {all_clus_coeff}\n")
        f.write(f"all_shortest_length\t\t: {all_shortest_len}\n")
        f.write(f"all_S\t\t: {all_S}\n")
        f.write(f"all_link\t\t: {all_link}\n")
        f.write(f"[mean] all_clustering_coeff\t\t: {np.mean(all_clus_coeff)}\n")
        f.write(f"[mean] all_shortest_length\t\t: {np.mean(all_shortest_len)}\n")
        f.write(f"[mean] all_S\t\t: {np.mean(all_S)}\n")
        f.write(f"[mean] all_link\t\t: {np.mean(all_link)}\n")
            
# シミュレーション結果を保存する関数
def save_simulation_eval(path, all_mean_r, all_delta_r, all_high_synchro):
    with open(f"{path}/evaluation.txt", mode="w") as f:
        f.write(f"all_mean_r\t\t: {all_mean_r}\n")
        f.write(f"all_delta_r\t\t: {all_delta_r}\n")
        f.write(f"all_high_synchro\t\t: {all_high_synchro}\n")
        f.write(f"[mean] all_mean_r\t\t: {np.mean(all_mean_r)}\n")
        f.write(f"[mean] all_delta_r\t\t: {np.mean(all_delta_r)}\n")
        f.write(f"[mean] all_high_synchro\t\t: {np.mean(all_high_synchro)}\n")

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
    plt.plot(t, r_values, linewidth=0.1)
    plt.ylim(0.0, 1.0)
    plt.title('Time evolution of the global Kuramoto order parameter r(t)')
    plt.xlabel('Time')
    plt.ylabel('r(t)')
    plt.grid(True)
    plt.savefig(f'{path}/{iteration}_synchronization_{const.start_time}-{const.finish_time}-{const.step_width}.png')
    plt.close()

# 膜電位uをプロットする関数
def plot_u(path, iteration, t, u_sol, const):
    plt.figure(figsize=(10, 6))
    plt.plot(t, u_sol.T)
    plt.title('Membrane potentials of the FHN oscillators')
    plt.xlabel('Time')
    plt.ylabel('u_k')
    plt.savefig(f'{path}/{iteration}_u_{const.start_time}-{const.finish_time}-{const.step_width}.png')
    plt.close()

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
    plt.close()
