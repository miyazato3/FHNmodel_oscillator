"""
[memo]
変更点：solve_ivpを使用
"""
import numpy as np
from matplotlib.colors import LogNorm
from scipy.integrate import solve_ivp
from numba import njit
import time
import datetime
from model.const import Const
from model import network
from model import analyze
from model import export

import pdb

# FHN振動子モデルの定義
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

class FHNmodel_oscillator:
    def __init__(self, const):
        # 初期値の設定
        self.const = const
        self.solve_time = 0.0
        self.t = np.linspace(self.const.start_time,
                             self.const.finish_time,
                             self.const.step_width)
        # ネットワーク生成
        self.A,\
        self.clustering_coeff,\
        self.shortest_path_length,\
        self.S = network.make_network(const.network_name,
                                      const.N, const.k, const.p)
        
    # 解を求める関数
    def solver(self):
        # 初期設定
        u0 = np.random.rand(self.const.N)
        v0 = np.random.rand(self.const.N)
        X0 = np.concatenate([u0, v0])

        solve_start_time = time.time()
        sol = solve_ivp(fhn_ode,
                        [self.const.start_time, self.const.finish_time],
                        X0,
                        method='LSODA',
                        t_eval=self.t,
                        args=(self.const.N, self.const.epsilon,
                              self.const.sigma, self.const.a,
                              self.A, self.const.B))

        solve_end_time = time.time()
        self.solve_time = solve_end_time - solve_start_time

        return sol

def experiment(time_setting, network_name, k, p, num_iterations):
    const = Const(time_setting, network_name, k, p, num_iterations)

    # 結果の出力先を設定
    root_dir = "results"
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_path = f"{root_dir}/{current_time}"
    export.check_path(save_path)

    # 実行時のパラメータを保存
    export.save_simulation_param(save_path, const)

    all_sol = []
    for n in range(const.num_iterations):
        np.random.seed(n)
        fhn = FHNmodel_oscillator(const)
        
        """ 解を求める """
        sol = fhn.solver()
        u_sol = sol.y[:const.N, :]
        v_sol = sol.y[const.N:, :]
        all_sol.append(sol)
        
        """ 同期度を求める """
        phases = np.array([analyze.calculate_phases(u_sol[:, i], v_sol[:, i]) for i in range(len(fhn.t))])  # 時間経過に伴う位相の変遷を計算
        r_values = np.array([analyze.calculate_r(phases[i, :], const.N) for i in range(len(fhn.t))])        # 各時刻での同期度rを計算

        """ プロット関係、ログを記録する """
        export.save_network_param(save_path, n, const, fhn)         # 使用したネットワークの情報を保存
        export.save_network_structure(save_path, n, fhn.A)          # ネットワーク構造を保存
        export.save_solution(save_path, n, u_sol.T, v_sol.T)        # 解を保存(視認性を挙げるために転置する, N行t列 -> t行N列)
        export.plot_synchro(save_path, n, fhn.t, r_values, const)   # 同期度のプロット
        export.plot_adjacency_matrix(save_path, n, fhn.A, const)    # 隣接行列Aの可視化

        print(f"solve time: {fhn.solve_time}s")
        print(f"Finished {n + 1}/{const.num_iterations}")

    """ シミュレーションnum_iteration回分の平均を計算する """
    all_u_sol = all_sol.y[:const.N, :]
    all_v_sol = all_sol.y[const.N:, :]
    mean_u_sol = np.mean(all_u_sol, axis=0)
    mean_v_sol = np.mean(all_v_sol, axis=0)

    # 解を保存(視認性を挙げるために転置する, N行t列 -> t行N列)
    export.save_solution(save_path, "all", mean_u_sol.T)
    export.save_solution(save_path, "all", mean_v_sol.T)

    # 時間経過に伴う位相の変遷を計算
    mean_phases = np.array([analyze.calculate_phases(mean_u_sol[:, i], mean_v_sol[:, i]) for i in range(len(fhn.t))])

    # 各時刻での同期度rを計算
    mean_r_values = np.array([analyze.calculate_r(mean_phases[i, :], const.N) for i in range(len(fhn.t))])

    # 同期度のプロット
    export.plot_synchro(save_path, "all", fhn.t, mean_r_values, const)
