from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import os

congestion_data = pd.read_csv(os.path.dirname(__file__) + "/congestion.csv", index_col='Time')

# print(congestion_data.index)
# print(congestion_data.columns)

# 訓練データ作成
def get_noise_data(data_num: int=1000) -> np.ndarray:
    np.random.seed(0) #乱数シード固定
    data_train = []

    for i in range(data_num):
        #ランダムに曜日と時間を選択
        youbi = np.random.choice(congestion_data.columns)
        time = np.random.choice(congestion_data.index)

        #ランダムノイズ
        noise1 = (np.random.randn() / 10) + 1
        noise2 = np.random.rand() * 4 - 2

        #作成
        cong = congestion_data[youbi][time] * noise1 + noise2
        cong = max(0, cong)
        data_train.append([congestion_data.columns.get_loc(youbi), congestion_data.index.get_loc(time), cong])
    data_train = np.array(data_train)
    # print(data_train.shape)
    return data_train

def get_train_data(data_train: np.ndarray) -> Tuple[List, List, List, List]:
    #訓練データを分割
    x_train, x_test, y_train, y_test = train_test_split(data_train[:, :2], data_train[:, 2], test_size=0.3)
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, x_test, y_train, y_test

def train(x_train, x_test, y_train, y_test):
    model = xgb.XGBRegressor()
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=10)
    return model

def predict(model, youbi: str, time: str) -> float:
    _youbi = congestion_data.columns.get_loc(youbi)
    _time = congestion_data.index.get_loc(time)
    return model.predict(np.array([[_youbi, _time]]))[0]

def show_congestion(model):
    #曜日別推定線
    for i, day in enumerate(['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']):
        tmp = np.zeros((24, 2))
        tmp[:, 0] = i
        tmp[:, 1] = np.arange(24)
        plt.plot(tmp[:, 1], model.predict(tmp), label=day)
    plt.title('predicted congestion')
    plt.xlabel('time')
    plt.ylabel('congestion')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    noise_data = get_noise_data(1000)
    x_train, x_test, y_train, y_test = get_train_data(noise_data)
    model = train(x_train, x_test, y_train, y_test)

    #テストデータで推定
    y_pred = model.predict(x_test)
    print('MSE')
    print(mean_squared_error(y_test, y_pred))
    print('相関係数')
    print(np.corrcoef(y_test, y_pred))

    print(model.predict(np.array([[0, 10]]))[0])

    show_congestion(model)