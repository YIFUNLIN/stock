import nlp2
import pandas as pd
import  numpy as np
from jinja2 import Environment, FileSystemLoader
from strategy.grid import trade # 匯入自定義的交易策略

# 從多個股票的數據文件中生成買賣推薦，並將推薦結果生成報告

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev
    return sharpe_ratio

def recommend_stock(url, parameters):
    df = pd.read_csv(url, index_col='Datetime')
    df.columns = map(str.lower, df.columns)
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    states_buy, states_sell, states_entry, states_exit, total_gains, invest = trade(df, **parameters) # 基於每支股票的數據計算買入和賣出的信號（通過 RSI、EMA、RSI)

    # 計算 Sharpe Ratio
    returns = df['close'].pct_change().dropna()
    sharpe_ratio = calculate_sharpe_ratio(returns)

    today = len(df)
    today_close_price = df.close.iloc[-1]

    # 確保推薦的股票買賣信號是近期且有效的，而不是基於很久以前發出的信號
    # 如果某個股票的買入信號距離今天已經超過 27 天，這意味著市場情況可能已經變化，該信號可能不再有效或適合，因此不再推薦它買入。
    # 同理，如果賣出信號距離今天超過 27 天，該信號可能也已經過期，賣出的建議就不應再被強烈推薦

    should_buy = abs(today - states_buy[-1]) < 27     #  計算的是最近一次買入操作距離今天的天數。如果這個值小於 27，則說明這個買入信號是最近 27 天內產生的，判斷它仍然有效，因此將其設為 should_buy = True
    should_sell = abs(today - states_sell[-1]) < 27   # 如果最近一次賣出操作距離今天小於 27 天，則表示這個賣出信號仍然有效，因此設為 should_sell = True

    return should_buy, should_sell, today_close_price, total_gains, sharpe_ratio # 前兩個回傳True or False


# 網頁前端顯示
def generate_report(urls, parameters, limit=30):
    results = []
    for url in urls:
        try:
            should_buy, should_sell, today_close_price, total_gains, sharpe_ratio = recommend_stock(url, parameters)
            if should_sell or should_buy:
                results.append({
                    "Stock": url.split('/')[-1].split('.')[0],
                    "Should_Buy": should_buy,
                    "Should_Sell": should_sell,
                    "Close_Price": today_close_price,
                    "Total_Gains": total_gains,
                    "Sharpe_Ratio": sharpe_ratio,
                })
        except Exception as e:
            pass

    sorted_results = sorted(results, key=lambda x: x['Total_Gains'], reverse=True)[:limit]
    
    df = pd.DataFrame(sorted_results)
    env = Environment(loader=FileSystemLoader('templates')) # 指定了模板文件所在的目錄
    template = env.get_template('stock_report_template.html') # 載入模板文件，包含了生成報告所需的HTML結構和Jinja2佔位符
    html_output = template.render(stocks=df.to_dict(orient='records')) # 將 Jinja2 模板中的內容渲染成完整的 HTML 文件，並將數據插入到相應的位置，並將結果寫入stock_report.html文件中
    """
    - 接收數據：這裡的 template.render() 方法用來渲染模板，其中的 stocks 參數是傳遞給模板的數據。
    df.to_dict(orient='records') 將 pandas DataFrame 轉換成一個列表作為 stocks 參數傳遞給模板，列表中的每一個元素是一個字典，代表一支股票及其相關的數據。這些數據會填充到模板文件中的對應位置

    - 填充佔位符：將模板中 {% for stock in stocks %} 迴圈內的佔位符 {{ stock.Stock }}, {{ stock.Should_Buy }}, {{ stock.Should_Sell }}, {{ stock.Recommended_Price }}，以及 {{ stock.Stock }} 用實際的數據來替換。
    - 生成完整的 HTML：Jinja2 會生成一個包含所有股票數據的完整 HTML 文件，這個文件就是渲染後的 html_output
    """


    with open('stock_report.html', 'w') as f:  # 最後將程式碼渲染後的 HTML 內容寫入一個新的文件 stock_report.html 中
        f.write(html_output)

# 制定交易策略的參數
parameters = {
    "initial_money":100000,
    "rsi_period": 14,
    "low_rsi": 20,
    "high_rsi": 80,
    "ema_period": 26,
    "cool_down_period":1,
    "print_log":False
}

for i in nlp2.get_files_from_dir("data"): # 獲取所有股票數據文件的URL，對每個文件執行recommend_stock函數來生成買賣信號並打印結果
    try:
        url = i
        should_buy, should_sell, today_close_price = recommend_stock(url, parameters)
        if should_sell or should_buy:
            print(
                f"{i.split('/')[-1].split('.')[0]} Should buy today: {should_buy}, Should sell today: {should_sell}, Recommended price: {today_close_price}")
    except Exception as e:
        pass
generate_report(list(nlp2.get_files_from_dir("data")), parameters) # 最後調用generate_report函數生成一個包含所有推薦結果的報告，報告結果寫入stock_report.html文件中
