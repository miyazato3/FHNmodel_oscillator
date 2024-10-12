"""
[memo]
変更点：solve_ivpを使用
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.integrate import solve_ivp
import networkx as nx
import datetime
from numba import njit
import time

""" 1. 実験パラメータの設定 """
# 定数の設定
N = 90  # 脳領域の数
epsilon = 0.05
sigma = 0.1
a = 0.5
phi = np.pi/2 - 0.1

# 時間の設定
start_time = 0
finish_time = 60
step_width = 601
#finish_time = 10800
#step_width = 108001

# カップリング行列B
B = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

""" 2. ネットワーク構造の定義 """
""" (現段階ではWSネットワークのみ実装) """
# ネットワーク作成時のパラメータの定義
k = 6   # 平均次数 (各ノードが持つ隣接ノードの数)
p = 1   # 再配線確率 (p = 1 で完全なランダムネットワーク)
num_links = 270  # 全リンク数
link_weight = 1  # リンク強度
desired_clustering_coeff = 0.05  # 目標クラスタリング係数
desired_shortest_path_length = 2.7  # 目標平均最短経路長

# ワッツ-ストロガッツ・ネットワークを生成
G = nx.watts_strogatz_graph(N, k, p)
#A = np.random.rand(N, N)

# 隣接行列 A を生成し、リンクの重みを設定
A = nx.to_numpy_array(G) * link_weight

# Aの対角成分を0で初期化
for i in range(N): A[i][i] = 0.0

# 現在のネットワークのクラスタリング係数と平均最短経路長を計算
clustering_coeff = nx.average_clustering(G, weight=None)
shortest_path_length = nx.average_shortest_path_length(G)
print(f'Clustering Coefficient: {clustering_coeff}')
print(f'Shortest Path Length: {shortest_path_length}')

""" 3. 関数の定義 """
# FHNモデルの定義
# def fhn_ode(t, X, N, epsilon, sigma, a, A, B):
#     u = X[:N]
#     v = X[N:]
#     du = np.zeros(N)
#     dv = np.zeros(N)
#     for k in range(N):
#         sum_u = sum(A[k, j] * (B[0, 0] * (u[j] - u[k]) + B[0, 1] * (v[j] - v[k])) for j in range(N))
#         sum_v = sum(A[k, j] * (B[1, 0] * (u[j] - u[k]) + B[1, 1] * (v[j] - v[k])) for j in range(N))
#         du[k] = (u[k] - u[k]**3 / 3 - v[k]) / epsilon + sigma * sum_u
#         dv[k] = u[k] + a + sigma * sum_v
#     return np.concatenate([du, dv])

@njit
def fhn_ode(t, X, N, epsilon, sigma, a, A, B):
    u = X[:N]
    v = X[N:]
    du = np.zeros(N)
    dv = np.zeros(N)
    for k in range(N):
        sum_u = np.sum(A[k, :] * (B[0, 0] * (u - u[k]) + B[0, 1] * (v - v[k])))
        sum_v = np.sum(A[k, :] * (B[1, 0] * (u - u[k]) + B[1, 1] * (v - v[k])))
        du[k] = (u[k] - u[k]**3 / 3 - v[k]) / epsilon + sigma * sum_u
        dv[k] = u[k] + a + sigma * sum_v
    
    result = np.zeros(2 * N)
    result[:N] = du
    result[N:] = dv
    return result
    
# 位相の計算
def calculate_phases(u, v):
    return np.arctan2(v, u)

# 同期度rの計算
def calculate_r(phases):
    return np.abs(np.sum(np.exp(1j * phases)) / N)

""" 4. 初期化 """
# 初期条件の設定
u0 = np.random.rand(N)
v0 = np.random.rand(N)
X0 = np.concatenate([u0, v0])

# 時間の設定
t = np.linspace(start_time, finish_time, step_width)

""" 5. 数値計算の実行 """
# ODE solverをsolve_ivpに変更
dbg_solve_start_time = time.time()
sol = solve_ivp(fhn_ode, [start_time, finish_time], X0, method='LSODA', t_eval=t, args=(N, epsilon, sigma, a, A, B))
dbg_solve_end_time = time.time()

# 解(u,v)の取得
u_sol = sol.y[:N, :]
v_sol = sol.y[N:, :]

""" 6. 同期度の計算 """
# 時間経過に伴う位相の変遷を計算
phases = np.array([calculate_phases(u_sol[:, i], v_sol[:, i]) for i in range(len(t))])

# 各時刻での同期度rを計算
r_values = np.array([calculate_r(phases[i, :]) for i in range(len(t))])


""" 7. プロット処理 """
# プロットに必要な変数の定義
current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

# 実行パラメータを保存(未実装)

# 同期度のプロット
plt.figure(figsize=(20, 6))
plt.plot(t, r_values)
plt.ylim(0.0, 1.0)
plt.title('Time evolution of the global Kuramoto order parameter r(t)')
plt.xlabel('Time')
plt.ylabel('r(t)')
plt.grid(True)
plt.savefig(f'results/{current_time}-synchronization.png')

## 膜電位uのプロット
#plt.figure(figsize=(10, 6))
#plt.plot(t, u_sol.T)
#plt.title('Membrane potentials of the FHN oscillators')
#plt.xlabel('Time')
#plt.ylabel('u_k')
#plt.savefig(f'results/{current_time}-u.png')

# 隣接行列Aの可視化
plt.figure(figsize=(8, 8))
#plt.imshow(A, cmap='YlOrRd', interpolation='none', norm=LogNorm())
plt.imshow(A, cmap='YlOrRd', interpolation='none')
plt.colorbar(label='Link Strength')
plt.title('Adjacency Matrix')
plt.xlabel('Node index')
plt.ylabel('Node index')
plt.savefig(f'results/{current_time}-adjacency-matrix.png')

""" 実行時間の表示 """
print(f"[solve] {dbg_solve_end_time - dbg_solve_start_time}")