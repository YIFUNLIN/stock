import pandas as pd
import numpy as np
from datetime import timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import talib as ta
import kerastuner as kt

# 計算技術指標
def add_technical_indicators(data):
    data['MA5'] = data['close'].rolling(window=5).mean()
    data['MA20'] = data['close'].rolling(window=20).mean()
    data['RSI'] = ta.RSI(data['close'].values, timeperiod=14)
    data['MACD'], data['MACD_signal'], _ = ta.MACD(data['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    data = data.dropna()
    return data

# 數據預處理，準備LSTM所需的數據
def preprocess_data_for_lstm(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i, 0])  # 預測收盤價
    return np.array(X), np.array(y)

# 建立LSTM模型
def build_lstm_model(hp):
    model = Sequential()
    # 調整LSTM層數和單元數
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(LSTM(units=hp.Int('units_' + str(i), min_value=32, max_value=256, step=32), return_sequences=True if i < hp.Int('num_layers', 1, 3) -1 else False))
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), 0.0, 0.5, step=0.1)))
    model.add(Dense(units=1))
    # 調整學習率和優化器
    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

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
    
    # 將計數轉換為概率
    for state in transition_matrix:
        total = sum(transition_matrix[state].values())
        for next_state in transition_matrix[state]:
            if total > 0:
                transition_matrix[state][next_state] /= total
            else:
                transition_matrix[state][next_state] = 0.0
    
    return transition_matrix

# 買賣策略整合LSTM和馬可夫鏈，並考慮交易成本
def trade_with_lstm(real_movement, lstm_model, scaler, time_steps=60, initial_money=10000, max_buy=1, max_sell=1, 
                    transaction_fee_percent=0.001, slippage_percent=0.0005, print_log=True):
    money = initial_money
    starting_money = initial_money
    current_inventory = 0
    states_buy, states_sell = [], []
    real_movement_values = real_movement['close'].values
    dates = real_movement.index

    # 預處理LSTM所需的數據
    features = ['close', 'MA5', 'MA20', 'RSI', 'MACD', 'MACD_signal']
    data = real_movement[features].values
    inputs = scaler.transform(data)
    X_test = []
    for i in range(time_steps, len(inputs)):
        X_test.append(inputs[i-time_steps:i])
    X_test = np.array(X_test)

    # 使用LSTM進行預測
    predicted_prices = lstm_model.predict(X_test)
    # 由於我們的模型預測的是縮放後的價格，需要反轉縮放
    predicted_prices = scaler.inverse_transform(np.hstack((predicted_prices, np.zeros((predicted_prices.shape[0], data.shape[1]-1)))))[:,0]

    # 開始交易模擬
    for i in range(len(predicted_prices)):
        idx = i + time_steps
        if idx >= len(real_movement_values):
            break
        current_price = real_movement_values[idx]
        current_date = dates[idx]

        # LSTM預測的價格
        predicted_price = predicted_prices[i]

        # 馬可夫鏈狀態轉移概率
        prob_up = transition_matrix[current_state]['Up']
        prob_down = transition_matrix[current_state]['Down']

        # 考慮滑點和手續費
        buy_price = current_price * (1 + slippage_percent)
        sell_price = current_price * (1 - slippage_percent)

        # 判斷買入信號
        if predicted_price > buy_price and prob_up > prob_down:
            shares = min(money // buy_price, max_buy)
            if shares > 0:
                total_buy_cost = shares * buy_price * (1 + transaction_fee_percent)
                if money >= total_buy_cost:
                    money -= total_buy_cost
                    current_inventory += shares
                    states_buy.append(idx)
                    if print_log:
                        print(f'{current_date}: 買入 {shares} 股，價格 {buy_price:.2f}，餘額 {money:.2f}')
        # 判斷賣出信號
        elif predicted_price < sell_price and prob_down > prob_up:
            if current_inventory > 0:
                shares = min(current_inventory, max_sell)
                total_sell_gain = shares * sell_price * (1 - transaction_fee_percent)
                money += total_sell_gain
                current_inventory -= shares
                states_sell.append(idx)
                if print_log:
                    print(f'{current_date}: 賣出 {shares} 股，價格 {sell_price:.2f}，餘額 {money:.2f}')

        # 更新馬可夫鏈狀態
        if idx < len(real_movement_values) - 1:
            if real_movement_values[idx + 1] > current_price:
                current_state = 'Up'
            elif real_movement_values[idx + 1] < current_price:
                current_state = 'Down'
            else:
                current_state = 'Stable'

    # 計算最終收益
    portfolio_value = money + current_inventory * real_movement_values[-1]
    total_gains = portfolio_value - starting_money
    invest = (total_gains / starting_money) * 100

    # 計算夏普比率和最大回撤
    returns = []
    portfolio_values = []
    money_temp = starting_money
    inventory_temp = 0

    for i in range(len(real_movement_values)):
        if i in states_buy:
            inventory_temp += max_buy
            money_temp -= max_buy * real_movement_values[i] * (1 + transaction_fee_percent + slippage_percent)
        elif i in states_sell:
            money_temp += max_sell * real_movement_values[i] * (1 - transaction_fee_percent - slippage_percent)
            inventory_temp -= max_sell
        portfolio_values.append(money_temp + inventory_temp * real_movement_values[i])
        if i > 0:
            returns.append((portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2])

    # 夏普比率
    sharpe_ratio = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) != 0 else 0

    # 最大回撤
    drawdowns = []
    peak = portfolio_values[0]
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        drawdowns.append(drawdown)
    max_drawdown = max(drawdowns)

    if print_log:
        print(f'最終資產：{portfolio_value:.2f}')
        print(f'總收益：{total_gains:.2f}')
        print(f'投資回報率：{invest:.2f}%')
        print(f'夏普比率：{sharpe_ratio:.2f}')
        print(f'最大回撤：{max_drawdown:.2f}')

    return states_buy, states_sell, total_gains, invest, sharpe_ratio, max_drawdown

# 模型訓練與回測
if __name__ == "__main__":
    # 載入市場數據（請替換為實際數據）
    # 假設有 real_movement DataFrame，包含 'date' 和 'close' 欄位
    # real_movement = pd.read_csv('your_market_data.csv', parse_dates=['date'])
    # real_movement.set_index('date', inplace=True)

    # 生成示例數據
    np.random.seed(42)
    date_range = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    prices = np.cumsum(np.random.normal(loc=0, scale=1, size=(1000,))) + 100  # 累積隨機漫步
    real_movement = pd.DataFrame({'date': date_range, 'close': prices})
    real_movement.set_index('date', inplace=True)

    # 添加技術指標
    real_movement = add_technical_indicators(real_movement)

    # 處理缺失值
    real_movement.dropna(inplace=True)

    # 數據標準化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(real_movement)

    # 準備訓練和測試數據
    time_steps = 60
    X, y = preprocess_data_for_lstm(scaled_data, time_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 建立和訓練LSTM模型，使用Keras Tuner進行超參數調優
    tuner = kt.RandomSearch(build_lstm_model,
                            objective='val_loss',
                            max_trials=10,
                            executions_per_trial=1,
                            directory='kt_tuner_dir',
                            project_name='lstm_tuning')

    tuner.search(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # 獲取最佳模型
    best_model = tuner.get_best_models(num_models=1)[0]

    # 評估模型
    train_loss = best_model.evaluate(X_train, y_train)
    test_loss = best_model.evaluate(X_test, y_test)
    print(f'訓練集損失：{train_loss:.4f}')
    print(f'測試集損失：{test_loss:.4f}')

    # 執行交易策略
    states_buy, states_sell, total_gains, invest, sharpe_ratio, max_drawdown = trade_with_lstm(
        real_movement, best_model, scaler, time_steps=time_steps)

    print(f'總收益：{total_gains:.2f}')
    print(f'投資回報率：{invest:.2f}%')
    print(f'夏普比率：{sharpe_ratio:.2f}')
    print(f'最大回撤：{max_drawdown:.2f}')
