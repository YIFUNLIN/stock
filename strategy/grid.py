import pandas as pd
import numpy as np

def trade(real_movement, initial_money, rsi_period, low_rsi, high_rsi, cool_down_period, print_log):
    money = initial_money
    states_buy = []
    states_sell = []
    states_entry = [False] * len(real_movement)  # 初始化所有條目為False，標記是否進入或退出市場
    states_exit = [False] * len(real_movement)   # 初始化所有條目為False
    current_inventory = 0                        # 目前持有的股票數量
    last_action_day = -cool_down_period  # 用來追蹤上一次操作發生的日期，避免在同一天內頻繁操作

    # 計算RSI
    delta = real_movement['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    real_movement['rsi'] = 100 - (100 / (1 + rs))

    # 計算布林帶
    real_movement['ma20'] = real_movement['close'].rolling(window=20).mean()
    real_movement['stddev'] = real_movement['close'].rolling(window=20).std()
    real_movement['upper_band'] = real_movement['ma20'] + (real_movement['stddev'] * 2)
    real_movement['lower_band'] = real_movement['ma20'] - (real_movement['stddev'] * 2)
    
    # 計算EMA均線
    real_movement['ema_short'] = real_movement['close'].ewm(span=12, adjust=False).mean()
    real_movement['ema_long'] = real_movement['close'].ewm(span=26, adjust=False).mean()
    real_movement['ema_diff'] = real_movement['ema_short'] - real_movement['ema_long']
    # 生成EMA交叉信號
    real_movement['buy_signal'] = (real_movement['ema_diff'].shift(1) < 0) & (real_movement['ema_diff'] > 0)
    real_movement['sell_signal'] = (real_movement['ema_diff'].shift(1) > 0) & (real_movement['ema_diff'] < 0)

    def buy(i, price):
        nonlocal money, current_inventory, last_action_day
        shares = money // price  # 購買盡可能多的股票
        if shares >= 1:
            current_inventory += shares
            money -= price * shares
            states_buy.append(i)
            states_entry[i] = True
            last_action_day = i
            if print_log:
                print(f'{real_movement.index[i]}: 買入 {shares} 股，價格 {price}，餘額 {money}')
        else:
            if print_log:
                print(f'{real_movement.index[i]}: 資金不足，無法買入。')

    def sell(i, price):
        nonlocal money, current_inventory, last_action_day
        if current_inventory > 0:
            shares = current_inventory  # 賣出所有持倉
            current_inventory -= shares
            money += price * shares
            states_sell.append(i)
            states_exit[i] = True
            last_action_day = i
            if print_log:
                print(f'{real_movement.index[i]}: 賣出 {shares} 股，價格 {price}，餘額 {money}')
        else:
            if print_log:
                print(f'{real_movement.index[i]}: 無持倉，無法賣出。')

    # 交易邏輯
    for i in range(1, len(real_movement)):
        if i - last_action_day > cool_down_period:
            # 買入條件：RSI低於低位，價格低於下軌，EMA金叉
            if (real_movement['buy_signal'].iloc[i] and
                real_movement['rsi'].iloc[i] < low_rsi and
                real_movement['close'].iloc[i] < real_movement['lower_band'].iloc[i]):
                buy(i, real_movement['close'].iloc[i])
            # 賣出條件：RSI高於高位，價格高於上軌，EMA死叉
            elif (real_movement['sell_signal'].iloc[i] and
                  real_movement['rsi'].iloc[i] > high_rsi and
                  real_movement['close'].iloc[i] > real_movement['upper_band'].iloc[i]):
                sell(i, real_movement['close'].iloc[i])

    # 投資報酬率計算
    total_gains = money + current_inventory * real_movement['close'].iloc[-1] - initial_money
    invest = (total_gains / initial_money) * 100

    return states_buy, states_sell, states_entry, states_exit, total_gains, invest
