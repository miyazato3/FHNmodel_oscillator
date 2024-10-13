import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 定数の設定
N       = 90
epsilon = 0.05 
sigma   = 0.6
a       = 0.5
phi     = np.pi/2 - 0.1

# カップリング行列B
B = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

# 接続行列A (ここではランダムなネットワークを仮定)
A = np.random.rand(N, N)
# Aの対角成分を0で初期化
for i in range(N): A[i][i] = 0.0

# FHNモデルの定義
def fhn_ode(X, t, N, epsilon, sigma, a, A, B):
    u = X[:N]
    v = X[N:]
    du = np.zeros(N)
    dv = np.zeros(N)
    for k in range(N):
        sum_u = sum(A[k, j] * (B[0, 0] * (u[j] - u[k]) + B[0, 1] * (v[j] - v[k])) for j in range(N))
        sum_v = sum(A[k, j] * (B[1, 0] * (u[j] - u[k]) + B[1, 1] * (v[j] - v[k])) for j in range(N))
        du[k] = (u[k] - u[k]**3 / 3 - v[k]) / epsilon + sigma * sum_u
        dv[k] = u[k] + a + sigma * sum_v
    #if step_cnt % 10 == 0: print(f"{step_cnt}/1000")
    return np.concatenate([du, dv])

# 位相の計算
def calculate_phases(u, v):
    return np.arctan2(v, u)

# 同期度rの計算
def calculate_r(phases):
    return np.abs(np.sum(np.exp(1j * phases)) / N)

# 初期条件の設定
u0 = np.random.rand(N)
v0 = np.random.rand(N)
X0 = np.concatenate([u0, v0])

# 時間の設定
#t = np.linspace(0, 500, 501)
t = np.linspace(0, 11, 10)

# ODEの解を求める
sol = odeint(fhn_ode, X0, t, args=(N, epsilon, sigma, a, A, B))

# 解の取得
u_sol = sol[:, :N]
v_sol = sol[:, N:]

# 時間経過に伴う位相の変遷を計算(uとvの位相という2次元データをuとvを合わせた1次元データに変えている?)
phases = np.array([calculate_phases(u_sol[i, :], v_sol[i, :]) for i in range(len(t))])

# 各時刻での同期度rを計算(ステップiにおける全ノードの位相の同期度を測る)
r_values = np.array([calculate_r(phases[i, :]) for i in range(len(t))])

# 同期度のプロット
plt.figure(figsize=(10, 6))
plt.plot(t, r_values)
plt.title('Time evolution of the global Kuramoto order parameter r(t)')
plt.xlabel('Time')
plt.ylabel('r(t)')
plt.grid(True)
plt.show()

# 解のプロット
plt.figure(figsize=(10, 6))
plt.plot(t, sol[:, :N])  # 各脳領域の膜電位 u のプロット
plt.title('Membrane potentials of the FHN oscillators')
plt.xlabel('Time')
plt.ylabel('u_k')
plt.show()
