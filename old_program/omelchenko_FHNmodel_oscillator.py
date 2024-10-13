import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
N = 100  # 振動子の数
R = 350   # 近傍の範囲
sigma = 0.1  # 結合強度
epsilon = 0.05  # 小さなパラメータ
a = 0.5  # 閾値パラメータ（振動体）
#phi = np.pi / 4  # クロス結合パラメータ
phi = np.pi/2 - 0.1  # クロス結合パラメータ

# 時間パラメータ
T = 100  # シミュレーション時間
dt = 0.01  # 時間ステップ
time_steps = int(T / dt)

# 回転行列 B の計算
cos_phi = np.cos(phi)
sin_phi = np.sin(phi)
B = np.array([[cos_phi, sin_phi],
              [-sin_phi, cos_phi]])

# 初期条件
np.random.seed(42)
u = np.random.rand(N) * 2 - 1  # uの初期値
v = np.random.rand(N) * 2 - 1  # vの初期値

# 時間発展のための保存リスト
u_history = np.zeros((time_steps, N))
v_history = np.zeros((time_steps, N))

# オイラー法で時間発展
for t in range(time_steps):
    u_new = np.copy(u)
    v_new = np.copy(v)
    
    for k in range(N):
        # 非局所的な結合の計算
        sum_u = 0
        sum_v = 0
        for j in range(k - R, k + R + 1):
            j_mod = j % N  # 周期境界条件を適用
            delta_u = u[j_mod] - u[k]
            delta_v = v[j_mod] - u[k]
            
            sum_u += B[0, 0] * delta_u + B[0, 1] * delta_v
            sum_v += B[1, 0] * delta_u + B[1, 1] * delta_v
        
        # 方程式(1a)の更新
        du = u[k] - (u[k]**3) / 3 - v[k] + (sigma / (2 * R)) * sum_u
        u_new[k] = u[k] + (dt / epsilon) * du
        
        # 方程式(1b)の更新
        dv = u[k] + a + (sigma / (2 * R)) * sum_v
        v_new[k] = v[k] + dt * dv

    # 時間発展の保存
    u_history[t, :] = u_new
    v_history[t, :] = v_new
    
    # 更新
    u = u_new
    v = v_new

# 結果をプロット
plt.figure(figsize=(10, 6))
plt.imshow(u_history.T, aspect='auto', cmap='plasma', extent=[0, T, 0, N])
plt.colorbar(label="u")
plt.xlabel("Time")
plt.ylabel("Oscillator index")
plt.title("Evolution of u across oscillators")
plt.show()
