from typing import List, Tuple
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx

"""
インストール
pip install matplotlib
pip install networkx
pip install geopandas
pip install osmnx
"""

"""
初期情報
"""
# 出力させる経路数
# PATH_NUMBER = 10

"""
地図情報を取得
"""
def get_map_data(city: str='Shibuya', state: str='Tokyo', country: str='Japan', network_type: str='drive'):
    # 地図データをダウンロード
    return ox.graph_from_place({'city': city, 'state': state, 'country': country}, network_type=network_type) # 場所を指定

def get_nearest_nodes(G, start: Tuple[float, float], goal: Tuple[float, float]):
    #近くの位置座標を取得する
    start_node = ox.nearest_nodes(G, start[0], start[1])
    goal_node = ox.nearest_nodes(G, goal[0], goal[1])
    return start_node, goal_node

"""
経路探索
"""
def get_shortest_path(G, start_node, goal_node, path_number):
    routes = []
    costs = [0 for i in range(path_number)]
    for i in range(path_number):
        # ダイクストラ法で経路を探索
        route = nx.dijkstra_path(G, start_node, goal_node, weight='length') # 一つのみ取得。
        routes.append(route)
        # print("edge count", len(route)) # edgeの本数を出力

        # 最適化した経路が再び出ないようにコストを追加する。
        for j in range(len(route)-1):
            costs[i] += G[route[j]][route[j+1]][0]['length']
            G[route[j]][route[j+1]][0]['length'] += 20

        # print("cost", costs[i]) # 経路のコストを出力

    # コストを追加した部分を元に戻す。
    for route in routes:
        for j in range(len(route)-1):
            G[route[j]][route[j+1]][0]['length'] -= 20

    return routes, costs

"""
経路の可視化
"""
def show_map_data(G, routes, route_colors='r', ax=None) -> None:
    # グラフのみを可視化する
    # ox.plot_graph(G, bgcolor='#ffffff', node_color='#87cefa')

    # 経路を可視化する
    if len(routes) == 1:
        return ox.plot_graph_route(G, routes[0], bgcolor='#ffffff', node_color='#87cefa', route_color=route_colors, route_linewidth=2, route_alpha=0.8, orig_dest_size=100, ax=ax)
    else:
        return ox.plot_graph_routes(G, routes, bgcolor='#ffffff', node_color='#87cefa', route_colors=route_colors, route_linewidth=2, route_alpha=0.8, orig_dest_size=100, ax=ax)