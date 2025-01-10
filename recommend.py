import nlp2
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 計算夏普比率: 衡量投資組合風險調整後收益的指標。   夏普比率越高，代表單位風險下的回報越高
def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    mean_return = np.mean(returns)  # 計算平均收益
    std_dev = np.std(returns)       # 標準差
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else 0
    return sharpe_ratio

# 計算最大回撤: 衡量投資組合從最高值回落的最大幅度，反映資金在最差情況下的損失
def calculate_max_drawdown(portfolio_values):
    drawdowns = []
    peak = portfolio_values[0]
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
    return max(drawdowns)

def calculate_rsi(prices, timeperiod=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fastperiod=12, slowperiod=26, signalperiod=9):
    ema_fast = prices.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = prices.ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    return macd, signal

# 修改後的 add_technical_indicators，支持批量處理並節省時間
def add_technical_indicators(df):
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'] = calculate_macd(df['close'])
    df.dropna(inplace=True)  # 丟棄缺失值
    return df

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
    
    return transition_matrix, states[-1]  # 返回轉移矩陣和最後的狀態


def build_and_train_xgboost_model(data, features, time_steps):
    # 生成目標變量（預測次日漲跌：漲為1，跌為0）
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    data.dropna(inplace=True)

    # 提取特徵與目標
    X = data[features]
    y = data['target']

    # 分割數據集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost 模型
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'  # 避免警告
    )
    model.fit(X_train, y_train)

    # 評估模型準確率
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Model Accuracy: {accuracy:.2f}")

    return model



# 股票推薦函數
def recommend_stock(url, parameters):
    # 讀取股票資料
    df = pd.read_csv(url, index_col='Datetime', parse_dates=True)
    df.columns = map(str.lower, df.columns)
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    # 添加技術指標
    df = add_technical_indicators(df)

    # 確保有足夠的數據進行預測
    if len(df) < 100:
        return None

    # 特徵選擇
    features = ['MA5', 'MA20', 'RSI', 'MACD', 'MACD_signal']

    # 構建並訓練 XGBoost 模型
    xgboost_model = build_and_train_xgboost_model(df, features, time_steps=parameters['time_steps'])

    # 預測未來價格
    df['predicted'] = xgboost_model.predict(df[features])

    # 計算馬可夫鏈
    transition_matrix, current_state = calculate_markov_chain(df['close'].values)

    # 初始化交易變量
    money = parameters['initial_money']
    holdings = 0
    portfolio = [money]
    states_buy = []
    states_sell = []
    sharpe_threshold = 1.0  # Sharpe Ratio 的最低要求

    for i in range(len(df) - parameters['time_steps']):
        current_price = df['close'].iloc[i + parameters['time_steps']]
        predicted_class = df['predicted'].iloc[i + parameters['time_steps']]

        # 計算暫時的 Sharpe Ratio
        portfolio_value = money + holdings * current_price
        temp_portfolio = portfolio + [portfolio_value]
        temp_returns = np.diff(temp_portfolio) / temp_portfolio[:-1]
        temp_sharpe = calculate_sharpe_ratio(temp_returns)

        # 獲取馬可夫鏈的狀態轉移概率
        prob_up = transition_matrix[current_state]['Up']
        prob_down = transition_matrix[current_state]['Down']

        # 買入策略
        if predicted_class == 1 and temp_sharpe > sharpe_threshold and prob_up > prob_down:
            if money >= current_price * parameters['max_buy']:
                buy_amount = parameters['max_buy']
                money -= current_price * buy_amount * (1 + parameters['transaction_fee_percent'] + parameters['slippage_percent'])
                holdings += buy_amount
                states_buy.append(i + parameters['time_steps'])

        # 賣出策略
        elif predicted_class == 0 and temp_sharpe > sharpe_threshold and prob_down > prob_up:
            if holdings >= parameters['max_sell']:
                sell_amount = parameters['max_sell']
                money += current_price * sell_amount * (1 - parameters['transaction_fee_percent'] - parameters['slippage_percent'])
                holdings -= sell_amount
                states_sell.append(i + parameters['time_steps'])

        # 更新馬可夫鏈的當前狀態
        if i + parameters['time_steps'] < len(df['close']) - 1:
            next_price = df['close'].iloc[i + parameters['time_steps'] + 1]
            if next_price > current_price:
                current_state = 'Up'
            elif next_price < current_price:
                current_state = 'Down'
            else:
                current_state = 'Stable'

        # 更新資產
        portfolio_value = money + holdings * current_price
        portfolio.append(portfolio_value)

    # 計算最終績效指標
    total_gains = portfolio[-1] - parameters['initial_money']
    invest = total_gains / parameters['initial_money'] * 100
    sharpe_ratio = calculate_sharpe_ratio(np.diff(portfolio) / portfolio[:-1])
    max_drawdown = calculate_max_drawdown(portfolio)

    # 確定是否應該買入或賣出
    today_close_price = df['close'].iloc[-1]
    should_buy = len(states_buy) > 0 and (df.index[-1] - df.index[states_buy[-1]]).days < 7
    should_sell = len(states_sell) > 0 and (df.index[-1] - df.index[states_sell[-1]]).days < 7

    return should_buy, should_sell, today_close_price, total_gains, invest, sharpe_ratio, max_drawdown




# 生成報告函數
def generate_report(urls, parameters, limit=30):
    results = []
    
    for url in urls:
        try:
            # 調用 recommend_stock 函數
            recommendation = recommend_stock(url, parameters)
            if recommendation is not None:
                # 解構推薦結果
                should_buy, should_sell, today_close_price, total_gains, invest, sharpe_ratio, max_drawdown = recommendation
                
                # 僅保留有買入或賣出信號的股票
                if should_sell or should_buy:
                    results.append({
                        "Stock": url.split('/')[-1].split('.')[0],  # 從文件名提取股票名稱
                        "Should_Buy": should_buy,
                        "Should_Sell": should_sell,
                        "Close_Price": today_close_price,
                        "Total_Gains": total_gains,
                        "Investment_Return": invest,
                        "Sharpe_Ratio": sharpe_ratio,
                        "Max_Drawdown": max_drawdown
                    })
        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue  # 確保單支股票出錯不影響整個報告生成

    # 根據總收益進行排序，僅保留前 limit 項
    sorted_results = sorted(results, key=lambda x: x['Total_Gains'], reverse=True)[:limit]
    
    if not sorted_results:
        print("No results to generate report. Please check your input data or model.")
        return

    # 渲染模板，生成 HTML 報告
    try:
        env = Environment(loader=FileSystemLoader('templates'))
        template = env.get_template('stock_report_template.html')
        html_output = template.render(stocks=sorted_results)

        # 寫入文件
        with open('stock_report.html', 'w', encoding='utf-8') as f:
            f.write(html_output)
        print("Report generated successfully: stock_report.html")
    except Exception as e:
        print(f"Error generating report: {e}")


# 制定交易策略的參數
parameters = {
    "time_steps": 60,
    "initial_money": 100000,
    "max_buy": 10,
    "max_sell": 10,
    "transaction_fee_percent": 0.001,
    "slippage_percent": 0.0005,
    "print_log": False
}

# 生成報告
generate_report(list(nlp2.get_files_from_dir("data")), parameters)
