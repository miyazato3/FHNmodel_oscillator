from model import FHNmodel_oscillator

def main():
    """ 1. 実験パラメータの設定 """
    time_setting = [0, 11764.7, 10001]   # 時間の設定(3時間) [start, finish, witdh]
    time_setting = [0, 10, 10]   # 時間の設定(3時間) [start, finish, witdh]
    num_iterations = 2                 # シミュレーションの反復回数
    
    """ 2. シミュレーション実行 """
    """ 2-1. WS(Watts & Strogatz)ネットワーク """
    # k:平均次数, p:再配線確率
    network_name = "ws-network"
    k = 6
    p = 0.0
    FHNmodel_oscillator.experiment(time_setting, network_name, k, p, num_iterations)
    p = 0.006
    FHNmodel_oscillator.experiment(time_setting, network_name, k, p, num_iterations)
    p = 0.232
    FHNmodel_oscillator.experiment(time_setting, network_name, k, p, num_iterations)
    p = 1.0
    FHNmodel_oscillator.experiment(time_setting, network_name, k, p, num_iterations)

    """ 2-2. 重み無しフラクタルネットワーク """
    network_name = "unweighted-fractal"
    FHNmodel_oscillator.experiment(time_setting, network_name, k, p, num_iterations)
    
main()