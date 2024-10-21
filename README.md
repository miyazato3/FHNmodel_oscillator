# 研究用リポジトリ

【実行方法】
1. リポジトリをクローンする
2. `python main.py`

【出力ファイルの説明】
* `[n]_adjacency-matrix.png`：ネットワーク構造（隣接行列）を可視化した図
* `[n]_synchronization.png`：同期度をプロットした図
* `simulation_parameter.txt`：シミュレーション実行時のパラメータの記録
* `network_parameter.txt`：使用したネットワークとそのパラメータ、ネットワーク特徴の記録
* `evaluation.txt`：シミュレーション結果を評価するための評価値の記録
* `network_structure.txt`：隣接行列の生データ
* `[n]_u_sol.txt`：解(u)の生データ
* `[n]_v_sol.txt`：解(v)の生データ
> ※
> nはイテレーション、seed値を表す。n="all"の場合は全イテレーション分の平均を表します。
