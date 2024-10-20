import numpy as np

class Const:
    def __init__(self, time_setting, network_name, k, p, num_iterations):
        # シミュレーションパラメータ
        self.N = 90
        self.epsilon = 0.05
        self.sigma = 0.0506
        self.a = 0.5
        self.phi = np.pi/2 - 0.1
        self.num_iterations = num_iterations
        self.B = np.array([[np.cos(self.phi), np.sin(self.phi)], [-np.sin(self.phi), np.cos(self.phi)]])

        # ネットワークパラメータ
        self.network_name = network_name
        self.k = k
        self.p = p
        
        # 時間設定
        self.start_time = time_setting[0]
        self.finish_time = time_setting[1]
        self.step_width = time_setting[2]
        
        # 重み無しフラクタルネットワークの特別な設定
        if network_name == "unweighted-fractal" or network_name == "weighted-fractal":
            self.N = 82
            self.sigma = 0.01
        