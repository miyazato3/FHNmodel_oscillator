from model import FHNmodel_oscillator

def main():
    """ 1. 実験パラメータの設定 """
    #time_setting = [0, 10800, 108001]   # 時間の設定(3時間) [start, finish, witdh]
    time_setting = [0, 11764.7, 10001]   # 時間の設定(3時間) [start, finish, witdh]
    num_iterations = 10                 # シミュレーションの反復回数
    
    """ 2. ネットワーク構造の定義 """
    network_name = "ws-network"             # WSネットワーク
    #network_name = "unweighted-fractal"     # 重み無しフラクタル
    k = 6           # 平均次数(WSネットワークで使用)
    p = 0.232       # 再配線確率(p=1 で完全なランダムネットワーク)(WSネットワークで使用)

    FHNmodel_oscillator.experiment(time_setting, network_name, k, p, num_iterations)
    
main()