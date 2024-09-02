import pandas as pd
import numpy as np
from datetime import timedelta

def calculate_markov_chain(data): # 在每次進行買入或賣出決策之前，會利用馬可夫鏈的轉移概率來決定是否進行操作
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

def trade(real_movement, delay=5, initial_state=1, initial_money=10000, max_buy=1, max_sell=1, print_log=True):
    """
    根據市場價格變動進行股票買賣模擬
    :param real_movement: 市場價格的真實變動序列 (假定此為DataFrame且包含時間戳索引)
    :param delay: 從買轉賣或賣轉買之間的延遲決策次數
    :param initial_state: 初始狀態，1 表示買，0 表示賣
    :param initial_money: 初始金額
    :param max_buy: 最大買入數量
    :param max_sell: 最大賣出數量
    :param print_log: 是否打印交易日誌
    """
    starting_money = initial_money
    state = initial_state
    current_inventory = 0
    states_buy, states_sell, states_entry, states_exit = [], [], [], []
    current_decision = 0
    real_movement = real_movement.close

    # 計算馬可夫鏈的狀態轉移矩陣
    transition_matrix = calculate_markov_chain(real_movement)
    current_state = 'Stable'  # 假設初始狀態為穩定

    for i in range(1, real_movement.shape[0]):
        current_time = real_movement.index[i]
        current_price = real_movement.iloc[i]

        # 根據馬可夫鏈判斷未來趨勢
        prob_up = transition_matrix[current_state]['Up']
        prob_down = transition_matrix[current_state]['Down']

        if state == 1 and current_price < real_movement.iloc[i - 1]:  # 考慮買入
            if current_decision >= delay and prob_up > prob_down:  # 如果馬可夫鏈預測上升的概率高
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
            if current_decision >= delay and prob_down > prob_up:  # 如果馬可夫鏈預測下降的概率高
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

    # 確保 states_entry 和 states_exit 的長度與 real_movement 相同
    if len(states_entry) < real_movement.shape[0]:
        states_entry.extend([False] * (real_movement.shape[0] - len(states_entry)))
    if len(states_exit) < real_movement.shape[0]:
        states_exit.extend([False] * (real_movement.shape[0] - len(states_exit)))

    total_gains = initial_money - starting_money
    invest = (total_gains / starting_money) * 100
    return states_buy, states_sell, states_entry, states_exit, total_gains, invest
