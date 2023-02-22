import numpy as np
from dimod import BinaryQuadraticModel
from neal import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite
from pyqubo import Binary

from congestion import get_noise_data, get_train_data, train, predict, show_congestion
from traffic_map import get_map_data, get_nearest_nodes, get_shortest_path, show_map_data

########################
####   入力データ    #####
########################
# サンプリングデータ
DATA_NUMBER = 1000

# 地域を選択
CITY = 'Shibuya'
STATE = 'Tokyo'
COUNTRY = 'Japan'

# 固定（ startとgoalが決められている場合 ）
START_POINT = (139.7034496066092, 35.68385325806718)
GOAL_POINT = (139.7101434341499, 35.64834729700043)

# 曜日と日時と緩和する基準値を選択（基準値は0以上50未満）
YOUBI = '月'
TIME = '10:00'
CONGESTION = 30

# 車の台数と提案する経路数（車の台数は7台まで。）
CAR_NUMBER = 3
PATH_NUMBER = 5

# アニーリング設定
weight = 1000
SHOTS = 1000
DWAVE_TOKEN=""
DWAVE_DEVICE_NAME="DW_2000Q_6"

########################
######   古典AI    ######
########################
# 混雑度予測
noise_data = get_noise_data(DATA_NUMBER)
x_train, x_test, y_train, y_test = get_train_data(noise_data)
model = train(x_train, x_test, y_train, y_test)
result = predict(model, YOUBI, TIME)
print("混雑度", result)

# 経路を取得
G = get_map_data(CITY, STATE, COUNTRY)
start_node, goal_node = get_nearest_nodes(G, START_POINT, GOAL_POINT)
routes, costs = get_shortest_path(G, start_node, goal_node, PATH_NUMBER)

# 全経路と緩和後の経路を格納する変数
all_routes = [routes.copy() for i in range(CAR_NUMBER)]
all_routes_flat = sum(all_routes, [])
answer_routes = []

# 混雑度判定
if result > CONGESTION:
    ########################
    ###  量子アニーリング   ###
    ########################
    variables = [[Binary(f"routes_{j}-{i}") for i in range(len(all_routes[j]))] for j in range(CAR_NUMBER)]

    # 目的関数を計算
    PATH_list = []
    BASE_QUBO_Obj_list = []
    for i, routes in enumerate(all_routes):
        for j, route in enumerate(routes):
            for k in range(len(route)-1):
                p1 = route[k]
                p2 = route[k+1]
                if (p1, p2) in PATH_list:
                    BASE_QUBO_Obj_list[PATH_list.index((p1, p2))] += variables[i][j]
                elif (p2, p1) in PATH_list:
                    BASE_QUBO_Obj_list[PATH_list.index((p2, p1))] += variables[i][j]
                else:
                    PATH_list.append((p1, p2))
                    BASE_QUBO_Obj_list.append(variables[i][j])

    QUBO_Obj_list = [qubo ** 2 for qubo in BASE_QUBO_Obj_list]

    # QUBOの計算
    QUBO_Obj = sum(QUBO_Obj_list)
    QUBO_Penalty = sum([(sum(variables[i]) - 1) ** 2 for i in range(CAR_NUMBER)])
    QUBO = QUBO_Obj + weight * QUBO_Penalty

    # 量子アニーリング
    qubo_model = QUBO.compile()
    qubo_compiled, offset = qubo_model.to_qubo()

    bqm = BinaryQuadraticModel.from_qubo(qubo_compiled)
    print(f"Degree: {len(bqm)}")

    if not DWAVE_TOKEN:
        sampler = SimulatedAnnealingSampler()
        device = "SimulatedAnnealingSampler"
    else:
        REGION = "eu-central-1" if DWAVE_DEVICE_NAME == "Advantage_system5.3" else "na-west-1"
        sampler = EmbeddingComposite(DWaveSampler(region=REGION, token=DWAVE_TOKEN, solver={"name": DWAVE_DEVICE_NAME}))
        device = sampler.properties['child_properties']['chip_id'] + " (DWaveSampler)"

    sampleset = sampler.sample_qubo(qubo_compiled, num_reads=SHOTS, label="Quamtum Annealing Part2")
    answer = sampleset.first.sample
    energy = sampleset.first.energy
    print("デバイス:", device)
    print(f"結果\n{answer}")
    print(f"Energy: {energy}")

    # 経路マッピング
    for i, (k, v) in enumerate(answer.items()):
        if v == 1:
            answer_routes.append(all_routes_flat[i])

else:
    # ランダムに経路を選択
    for i in range(CAR_NUMBER):
        answer_routes.append(routes[np.random.randint(0, len(routes))])

color_list = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
route_colors = [color_list[i] for i in range(CAR_NUMBER) for j in range(PATH_NUMBER)]

show_map_data(G, all_routes_flat, route_colors)
show_map_data(G, answer_routes, color_list[0:CAR_NUMBER])