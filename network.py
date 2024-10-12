import networkx as nx
import numpy as np

""" ネットワーク構造の定義 """
def make_network(network_name, N, arg_k, arg_p):
    if network_name == "ws-network":
        A = make_ws_network(N, arg_k, arg_p)
    elif network_name == "unweighted-fractal":
        A, num_links = make_unweighted_fractal(N)
    elif network_name == "weighted-fractal":
        A = make_weighted_fractal(N)
    return A
    
# ワッツ-ストロガッツ・ネットワークを生成する関数
def make_ws_network(N, arg_k, arg_p):
    # パラメータの定義
    k = arg_k           # 平均次数 (各ノードが持つ隣接ノードの数)
    p = arg_p           # 再配線確率 (p = 1 で完全なランダムネットワーク)
    link_weight = 1     # リンク強度
    #num_links = 270  # 全リンク数
    #desired_clustering_coeff = 0.05  # 目標クラスタリング係数
    #desired_shortest_path_length = 2.7  # 目標平均最短経路長
    
    # ワッツ-ストロガッツ・ネットワークを生成
    G = nx.watts_strogatz_graph(N, k, p)
    
    # 隣接行列 A を生成し、リンクの重みを設定
    A = nx.to_numpy_array(G) * link_weight
    
    # Aの対角成分を0で初期化
    for i in range(N): A[i][i] = 0.0

    # 現在のネットワークのクラスタリング係数と平均最短経路長を計算
    clustering_coeff = nx.average_clustering(G, weight=None)
    shortest_path_length = nx.average_shortest_path_length(G)
    print(f'Clustering Coefficient: {clustering_coeff}')
    print(f'Shortest Path Length: {shortest_path_length}')

    return A

# フラクタル構造の生成関数
def generate_fractal_pattern(base, n):
    pattern = base
    for _ in range(n - 1):
        pattern = [
            sub if bit == 1 else [0] * len(base)
            for bit in pattern for sub in [base]
        ]
        pattern = [bit for sublist in pattern for bit in sublist]  # フラット化
    return [0] + pattern  # 自己結合を除くために先頭に0を追加

# 重み無しフラクタルネットワークを生成する関数
def make_unweighted_fractal(N):
    # 基本パターンとフラクタルの階層数
    b_init = [1, 0, 1]  # 基本パターン
    n = 4  # 階層数
    b = len(b_init)  # 基本パターンの長さ

    pattern = generate_fractal_pattern(b_init, n)
    A = np.zeros((N, N))
    
    # 循環行列として隣接行列を構築
    for i in range(N):
        A[i] = np.roll(pattern, i)
        
    # ネットワークの特徴の計算
    num_links = np.sum(A)
    average_node_strength = np.sum(A) / N
    print(f'Number of non-zero links: {num_links}')
    print(f'Average node strength: {average_node_strength}')

    return A, num_links

# 重み付きフラクタルネットワークを生成する関数
def make_weighted_fractal(N):
    A, num_links = make_unweighted_fractal(N)

    # 重み付きネットワークを構築するためのリンクの重みを設定
    np.random.seed(128)  # 再現性のためのシード
    empirical_weights = np.random.uniform(0.001, 1.00, size=int(num_links))  # 仮の重み
    
    # 重み付き隣接行列の生成
    A[A == 1] = empirical_weights
    
    # 重み付きネットワークの特徴の計算
    weighted_strength = np.sum(A) / N
    print(f'Average node strength (weighted): {weighted_strength}')
    
    return A