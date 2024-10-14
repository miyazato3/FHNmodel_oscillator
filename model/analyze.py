import numpy as np
from model.const import Const

# 位相の計算
def calculate_phases(u, v):
    return np.arctan2(v, u)

# 同期度rの計算
def calculate_r(phases, N):
    return np.abs(np.sum(np.exp(1j * phases)) / N)

