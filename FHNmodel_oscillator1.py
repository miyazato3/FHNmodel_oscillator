"""

"""
import numpy as np
import matplotlib.pyplot as plt

# パラメータの設定
N = 90      # ノード数
a = 0.5
SIGMA = 0.6
EP = 0.05
PHI = np.pi / 2 - 0.1

# 隣接行列
A = np.random.rand(N,N)
for i in range(N): A[i][i] = 0.0

# 初期条件
u0 = np.random.rand(N)
v0 = np.random.rand(N)

# 時間の設定
t0 = 0.0
t_end = 20.0
dt = 0.01

# 微分方程式の定義
def du_dt(u, v):
    uu = np.cos(PHI)
    uv = np.sin(PHI)
    du = np.zeros(N)
    for k in range(N):
        sum = 0.0
        for j in range(N):
            sum += A[k][j] * (uu * (u[j] - u[k]) + uv * (v[j] - v[k]))
        sum *= SIGMA
        du[k] = (u[k] - (u[k]**3) / 3 - v[k] + sum) / EP

    return du

def dv_dt(u, v):
    vv = np.cos(PHI)
    vu = (-1) * np.sin(PHI)
    dv = np.zeros(N)
    for k in range(N):
        sum = 0.0
        for j in range(0,N,1):
            sum += A[k][j] * (vu * (u[j] - u[k]) + vv * (v[j] - v[k]))
        sum *= SIGMA
        dv[k] = u[k] + a + sum

    return dv

# ルンゲ・クッタ法による数値解法
def runge_kutta_step(u, v):
    k1_u = du_dt(u, v) * dt
    k1_v = dv_dt(u, v) * dt
    k2_u = du_dt(u + 0.5 * k1_u, v + 0.5 * k1_v) * dt
    k2_v = dv_dt(u + 0.5 * k1_u, v + 0.5 * k1_v) * dt
    k3_u = du_dt(u + 0.5 * k2_u, v + 0.5 * k2_v) * dt
    k3_v = dv_dt(u + 0.5 * k2_u, v + 0.5 * k2_v) * dt
    k4_u = du_dt(u + k3_u, v + k3_v) * dt
    k4_v = dv_dt(u + k3_u, v + k3_v) * dt
    u_next = u + (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6
    v_next = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

    return u_next, v_next

# 同期度を計算する関数
def get_synchronization(phases):
    return np.abs(np.sum(np.exp(1j * phases))) / N

# 時間の配列を生成
time = np.arange(t0, t_end, dt)

# 解の保存場所を用意
u = np.zeros((len(time), N))
v = np.zeros((len(time), N))
u[0, :] = u0
v[0, :] = v0

# 数値解析の実行
for i in range(1, len(time)):
    u[i, :], v[i, :] = runge_kutta_step(u[i-1, :], v[i-1, :])

# 結果のプロット
for j in range(N):
    plt.plot(time, u[:, j], label=f'v{j+1} (membrane potential of oscillator {j+1})')

plt.xlabel('Time')
plt.ylabel('Membrane potential')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()
