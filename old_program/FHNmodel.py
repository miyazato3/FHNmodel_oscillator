import numpy as np
import matplotlib.pyplot as plt

# パラメータの設定
a = 0.7
b = 0.8
tau = 12.5
I = 0.5

# 初期条件
v0 = 0.0
w0 = 0.0

# 時間の設定
t0 = 0.0
t_end = 100.0
dt = 0.01

# 微分方程式の定義
def dv_dt(v, w, I):
    return v - (v**3) / 3 - w + I

def dw_dt(v, w, a, b, tau):
    return (v + a - b * w) / tau

# 時間の配列を生成
time = np.arange(t0, t_end, dt)

# 解の保存場所を用意
v = np.zeros_like(time)
w = np.zeros_like(time)
v[0] = v0
w[0] = w0

# ルンゲ・クッタ法による数値解法
for i in range(1, len(time)):
    k1_v = dv_dt(v[i-1], w[i-1], I) * dt
    k1_w = dw_dt(v[i-1], w[i-1], a, b, tau) * dt
    
    k2_v = dv_dt(v[i-1] + 0.5 * k1_v, w[i-1] + 0.5 * k1_w, I) * dt
    k2_w = dw_dt(v[i-1] + 0.5 * k1_v, w[i-1] + 0.5 * k1_w, a, b, tau) * dt
    
    k3_v = dv_dt(v[i-1] + 0.5 * k2_v, w[i-1] + 0.5 * k2_w, I) * dt
    k3_w = dw_dt(v[i-1] + 0.5 * k2_v, w[i-1] + 0.5 * k2_w, a, b, tau) * dt
    
    k4_v = dv_dt(v[i-1] + k3_v, w[i-1] + k3_w, I) * dt
    k4_w = dw_dt(v[i-1] + k3_v, w[i-1] + k3_w, a, b, tau) * dt
    
    v[i] = v[i-1] + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    w[i] = w[i-1] + (k1_w + 2*k2_w + 2*k3_w + k4_w) / 6

# 結果のプロット
plt.plot(time, v, label='v (membrane potential)')
plt.plot(time, w, label='w (recovery variable)')
plt.xlabel('Time')
plt.ylabel('Variables')
plt.legend()
plt.show()
