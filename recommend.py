import nlp2
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader
from strategy.grid import trade_with_lstm  # 假設您已經在 strategy/grid.py 中實現了優化後的 trade_with_lstm 函數
from sklearn.preprocessing import MinMaxScaler
import talib as ta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 計算夏普比率
def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else 0
    return sharpe_ratio

# 計算最大回撤
def calculate_max_drawdown(portfolio_values):
    drawdowns = []
    peak = portfolio_values[0]
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
    return max(drawdowns)

# 添加技術指標
def add_technical_indicators(df):
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['RSI'] = ta.RSI(df['close'].values, timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = ta.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df.dropna(inplace=True)
    return df

# 構建和訓練LSTM模型
def build_and_train_lstm_model(data, time_steps=60):
    # 準備數據
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
        y.append(scaled_data[i, 0])  # 預測收盤價

    X, y = np.array(X), np.array(y)

    # 構建模型
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 訓練模型
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)  # 減少epochs，加快訓練速度

    return model, scaler

# 股票推薦函數
def recommend_stock(url, parameters):
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

    # 構建和訓練模型
    features = ['close', 'MA5', 'MA20', 'RSI', 'MACD', 'MACD_signal']
    data = df[features].values
    lstm_model, scaler = build_and_train_lstm_model(data, time_steps=parameters['time_steps'])

    # 執行交易策略
    results = trade_with_lstm(df, lstm_model, scaler, **parameters)
    if results is None:
        return None
    states_buy, states_sell, total_gains, invest, sharpe_ratio, max_drawdown = results

    today_close_price = df['close'].iloc[-1]

    # 確定是否應該買入或賣出
    should_buy = len(states_buy) > 0 and (df.index[-1] - df.index[states_buy[-1]]).days < 7
    should_sell = len(states_sell) > 0 and (df.index[-1] - df.index[states_sell[-1]]).days < 7

    return should_buy, should_sell, today_close_price, total_gains, invest, sharpe_ratio, max_drawdown

# 生成報告函數
def generate_report(urls, parameters, limit=30):
    results = []
    for url in urls:
        try:
            recommendation = recommend_stock(url, parameters)
            if recommendation is not None:
                should_buy, should_sell, today_close_price, total_gains, invest, sharpe_ratio, max_drawdown = recommendation
                if should_sell or should_buy:
                    results.append({
                        "Stock": url.split('/')[-1].split('.')[0],
                        "Should_Buy": should_buy,
                        "Should_Sell": should_sell,
                        "Recommended_Price": today_close_price,
                        "Total_Gains": total_gains,
                        "Investment_Return": invest,
                        "Sharpe_Ratio": sharpe_ratio,
                        "Max_Drawdown": max_drawdown
                    })
        except Exception as e:
            print(f"Error processing {url}: {e}")
            pass

    sorted_results = sorted(results, key=lambda x: x['Total_Gains'], reverse=True)[:limit]

    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('stock_report_template.html')
    html_output = template.render(stocks=sorted_results)

    with open('stock_report.html', 'w', encoding='utf-8') as f:
        f.write(html_output)

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
