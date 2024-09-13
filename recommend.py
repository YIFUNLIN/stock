import nlp2
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader
from strategy.grid import trade_with_lstm  # 假設您已經在 strategy/grid.py 中實現了優化後的 trade_with_lstm 函數
from sklearn.preprocessing import MinMaxScaler
import talib as ta
from tensorflow.keras.models import load_model

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
        drawdowns.append(drawdown)
    max_drawdown = max(drawdowns)
    return max_drawdown

# 添加技術指標
def add_technical_indicators(df):
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['RSI'] = ta.RSI(df['close'].values, timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = ta.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df.dropna(inplace=True)
    return df

# 股票推薦函數
def recommend_stock(url, parameters, lstm_model, scaler):
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
def generate_report(urls, parameters, lstm_model, scaler, limit=30):
    results = []
    for url in urls:
        try:
            recommendation = recommend_stock(url, parameters, lstm_model, scaler)
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

# 加載預訓練的LSTM模型和標準化器
# 請確保您已經訓練並保存了LSTM模型和標準化器
lstm_model = load_model('lstm_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 生成報告
generate_report(list(nlp2.get_files_from_dir("data")), parameters, lstm_model, scaler)
