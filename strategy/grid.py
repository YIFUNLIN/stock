import pandas as pd
import numpy as np

def trade(real_movement, initial_money=100000, rsi_period=14, low_rsi=45, high_rsi=55, ema_period=26, print_log=False):
    money = initial_money
    states_buy = []
    states_sell = []
    states_entry = [False] * len(real_movement)  # 初始化所有條目為False，標記是否進入或退出市場
    states_exit = [False] * len(real_movement)   # 初始化所有條目為False
    current_inventory = 0                        # 目前持有的股票數量

    # 計算RSI
    delta = real_movement['close'].diff() # 計算連續兩天收盤價的差異
    gain = (delta.where(delta > 0, 0)).fillna(0) # 當日價格上漲的情況(即delta > 0):找出所有上檔的日子，其他不符合的下跌日(即NULL值)，則填0
    loss = (-delta.where(delta < 0, 0)).fillna(0) # 當日價格下跌的情況
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean() # 平均盈利:利用rolling去計算指定期間（如14天）內的平均盈利
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean() # 平均損失
    rs = avg_gain / avg_loss # 計算相對強弱(RS):平均盈利除以平均損失。此比率顯示了市場上漲和下跌力度的相對大小
    real_movement['rsi'] = 100 - (100 / (1 + rs)) 

    # 計算布林帶
    real_movement['ma20'] = real_movement['close'].rolling(window=20).mean() # 20日的SMA，取過去20天收盤價的平均值
    real_movement['stddev'] = real_movement['close'].rolling(window=20).std() # 過去20天收盤價的標準差，用於衡量價格波動的大小
    real_movement['upper_band'] = real_movement['ma20'] + (real_movement['stddev'] * 2)  # 上帶通常設置在中間帶以上兩倍標準差的位置
    real_movement['lower_band'] = real_movement['ma20'] - (real_movement['stddev'] * 2)  # 下帶則設置在中間帶以下兩倍標準差的位置
                            # 這樣的設置幫助識別價格是否達到異常高或異常低的水平，因為統計上，價格應該有95%的概率在這兩條帶之間波動
    
    # 計算EMA
    real_movement['ema'] = real_movement['close'].ewm(span=ema_period, adjust=False).mean()

    def buy(i, price):
        nonlocal money, current_inventory
        if money >= price: # 檢查是否有足夠的資金購買至少一股
            shares = 1  
            current_inventory += shares # 更新庫存數量
            money -= price * shares # 從資金中扣除購買股票的費用
            states_buy.append(i)  # 在 states_buy 列表中記錄這次購買的索引
            states_entry[i] = True # 標記這一天發生了買入操作
            if print_log:
                print(f'{real_movement.index[i]}: buy {shares} units at price {price}, total balance {money}')
        else:
            if print_log:
                print(f'{real_movement.index[i]}: attempted to buy, insufficient funds.')

    def sell(i, price):
        nonlocal money, current_inventory
        if current_inventory > 0: # 當存量大於0時，可以進行賣出
            shares = min(current_inventory, 1)  # 決定賣出的股數，這裡取當前庫存和1的最小值，通常意味著每次賣出1股
            current_inventory -= shares # 更新庫存數量，減去賣出的股數
            money += price * shares # 向資金總額添加賣出股票得到的金額
            states_sell.append(i) # 在 states_sell 列表中記錄這次賣出的索引
            states_exit[i] = True # 標記這一天發生了賣出操作
            if print_log:
                print(f'{real_movement.index[i]}: sell {shares} units at price {price}, total balance {money}')
        else:
            if print_log:
                print(f'{real_movement.index[i]}: attempted to sell, no inventory.')

    # 買賣判斷依據
    for i in range(len(real_movement)): 
        if real_movement['rsi'].iloc[i] < low_rsi: #判斷當天的RSI值是否小於設定的低RSI閾值(low_rsi)
            buy(i, real_movement['close'].iloc[i]) # 如果條件成立，則認為股票處於超賣狀態，去進行買入
        elif real_movement['rsi'].iloc[i] > high_rsi: # 判斷當天的RSI值是否高於設定的高RSI閾值(high_rsi)
            sell(i, real_movement['close'].iloc[i])  # 如果條件成立，則認為股票處於超買狀態，進行賣出   

    for i in range(len(real_movement)):
        if (real_movement['rsi'].iloc[i] < low_rsi) or (real_movement['close'].iloc[i] < real_movement['lower_band'].iloc[i]) or (real_movement['close'].iloc[i] < real_movement['ema'].iloc[i]):
            buy(i, real_movement['close'].iloc[i])
        elif (real_movement['rsi'].iloc[i] > high_rsi) and (real_movement['close'].iloc[i] > real_movement['upper_band'].iloc[i]) and (real_movement['close'].iloc[i] > real_movement['ema'].iloc[i]):
            sell(i, real_movement['close'].iloc[i])



    # 投資報酬率計算
    invest = ((money - initial_money) / initial_money) * 100    # 計算投資報酬率 :（當前資金 - 初始資金）/ 初始資金 * 100
    total_gains = money + current_inventory * real_movement['close'].iloc[-1] - initial_money # 計算總盈利:現金+現有股票市值 - 初始投資金額

    return states_buy, states_sell, states_entry, states_exit, total_gains, invest # 返回包括買入、賣出操作的索引、每天是否有進行交易的標記、總盈利、投資回報
