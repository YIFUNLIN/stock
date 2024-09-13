import pandas as pd
import numpy as np
from datetime import timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 建立LSTM模型
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # 預測未來的價格
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 數據預處理，準備LSTM所需的數據
def preprocess_data_for_lstm(data, time_steps=60):
    X_train, y_train = [], []
    for i in range(time_steps, len(data)):
        X_train.append(data[i-time_steps:i])
        y_train.append(data[i])
    return np.array(X_train), np.array(y_train)

# 計算馬可夫鏈的狀態轉移矩陣
def calculate_markov_chain(data):
    states = []
    for i in range(1, len(data)):
        if data[i] > data[i-1]:
            states.append('Up')
        elif data[i] < data[i-1]:
            states.append('Down')
        else:
            states.append('Stable')
    
    transition_matrix = {
        'Up': {'Up': 0, 'Down': 0, 'Stable': 0},
        'Down': {'Up': 0, 'Down': 0, 'Stable': 0},
        'Stable': {'Up': 0, 'Down': 0, 'Stable': 0},
    }
    
    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i + 1]
        transition_matrix[current_state][next_state] += 1
    
    for state in transition_matrix:
        total = sum(transition_matrix[state].values())
        for next_state in transition_matrix[state]:
            if total > 0:
                transition_matrix[state][next_state] /= total

    return transition_matrix

# 買賣策略整合LSTM和馬可夫鏈
def trade_with_lstm(real_movement, lstm_model, time_steps=60, delay=3, initial_state=1, initial_money=10000, max_buy=1, max_sell=1, print_log=True):
    starting_money = initial_money
    state = initial_state
    current_inventory = 0
    states_buy, states_sell, states_entry, states_exit = [], [], [], []
    current_decision = 0
    real_movement = real_movement.close

    # 計算馬可夫鏈的狀態轉移矩陣
    transition_matrix = calculate_markov_chain(real_movement)
    current_state = 'Stable'  # 假設初始狀態為穩定

    # 預處理LSTM所需的數據
    X_test, _ = preprocess_data_for_lstm(real_movement.values, time_steps)
    
    # 開始交易模擬
    for i in range(time_steps, real_movement.shape[0]):
        current_time = real_movement.index[i]
        current_price = real_movement.iloc[i]

        # 使用LSTM進行預測
        predicted_price = lstm_model.predict(X_test[i-time_steps].reshape(1, time_steps, 1))[0, 0]

        # 根據馬可夫鏈判斷未來趨勢
        prob_up = transition_matrix[current_state]['Up']
        prob_down = transition_matrix[current_state]['Down']

        if state == 1 and current_price < real_movement.iloc[i - 1]:  # 考慮買入
            if current_decision >= delay and prob_up > prob_down * 0.9 and predicted_price > current_price:  # 加入LSTM預測的條件
                shares = min(initial_money // current_price, max_buy)
                if shares > 0:
                    initial_money -= shares * current_price
                    current_inventory += shares
                    states_buy.append(i)
                    if print_log:
                        print(f'{current_time}: buy {shares} units at price {current_price}, total balance {initial_money}')
                current_decision = 0
            else:
                current_decision += 1

        elif state == 0 and current_price > real_movement.iloc[i - 1]:  # 考慮賣出
            if current_decision >= delay and prob_down > prob_up * 0.9 and predicted_price < current_price:  # 加入LSTM預測的條件
                sell_units = min(current_inventory, max_sell)
                if sell_units > 0:
                    initial_money += sell_units * current_price
                    current_inventory -= sell_units
                    states_sell.append(i)
                    if print_log:
                        print(f'{current_time}: sell {sell_units} units at price {current_price}, total balance {initial_money}')
                current_decision = 0
            else:
                current_decision += 1

        state = 1 - state  # 切換狀態
        states_entry.append(state == 1)
        states_exit.append(state == 0)

        # 更新當前狀態
        current_state = 'Up' if current_price > real_movement.iloc[i-1] else 'Down'

    total_gains = initial_money - starting_money
    invest = (total_gains / starting_money) * 100
    return states_buy, states_sell, states_entry, states_exit, total_gains, invest

# 模型訓練示例
if __name__ == "__main__":
    # 載入市場數據 (這裡假設有 real_movement DataFrame)
    # real_movement = pd.read_csv('your_market_data.csv', parse_dates=True, index_col=0)
    
    # 假設有 real_movement 並且我們正在使用 'close' 欄位進行預測
    real_movement = pd.DataFrame({
        'close': np.random.rand(1000)  # 隨機生成一些數據作為示例
    })

    # 訓練LSTM模型
    X_train, y_train = preprocess_data_for_lstm(real_movement['close'].values)
    lstm_model = build_lstm_model(input_shape=(X_train.shape[1], 1))
    lstm_model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=32)

    # 執行交易策略
    states_buy, states_sell, states_entry, states_exit, total_gains, invest = trade_with_lstm(real_movement, lstm_model)

    print(f'Total gains: {total_gains}, Investment return: {invest}%')
