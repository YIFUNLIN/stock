import nlp2
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from strategy.grid import trade


def recommend_stock(url, parameters):
    df = pd.read_csv(url, index_col='Datetime')
    df.columns = map(str.lower, df.columns) # 將所有欄位名轉換為小寫，確保後續處理中列名的一致性
    df['open'] = pd.to_numeric(df['open'], errors='coerce') # 將股票資料的欄位轉換為數值類型。如果轉換失敗，則用 NaN 填充
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    states_buy, states_sell, states_entry, states_exit, total_gains, invest = trade(df, **parameters) # 執行交易策略

    today = len(df) # 總交易天數
    today_close_price = df.close.iloc[-1] # 抓最後一筆的收盤價

    should_buy = abs(today - states_buy[-1]) < 27
    should_sell = abs(today - states_sell[-1]) < 27   # 若最後一次賣出到目前的天數小於27天，就認為應該進行賣出操作

    return should_buy, should_sell, today_close_price, total_gains


def generate_report(urls, parameters, limit=10): # 此函數負責產生股票推薦的 HTML 報告
    results = []
    for url in urls:
        try:
            should_buy, should_sell, today_close_price, total_gains = recommend_stock(url, parameters) # 對於給定的 URL 列表，函數嘗試為每個 URL 執行 recommend_stock 並收集結果
            if should_sell or should_buy:
                results.append({
                    "Stock": url.split('/')[-1].split('.')[0],
                    "Should_Buy": should_buy,
                    "Should_Sell": should_sell,
                    "Recommended_Price": today_close_price,
                    "Total_Gains": total_gains,
                })
        except Exception as e:
            pass

    # 排序並選擇前10檔股票，假設是根據推薦價格排序
    sorted_results = sorted(results, key=lambda x: x['Total_Gains'], reverse=True)[:limit]

    df = pd.DataFrame(sorted_results)

    # 使用 Jinja2 範本引擎和預先定義的 HTML 範本產生報告
    env = Environment(loader=FileSystemLoader('templates')) 
    # Environment 是 Jinja2 中的一個核心概念，它封裝了模板的配置和全域物件。這裡建立一個 Environment 實例是為了管理範本檔案。
    # 使用 FileSystemLoader，指定 Jinja2 從哪個目錄下載入模板檔案。在這個範例中，模板檔案被放在名為 templates 的目錄中。
    
    template = env.get_template('stock_report_template.html') 
    # 使用 get_template 方法從環境中載入一個名為 stock_report_template.html 的模板檔案。
    # 這意味著你應該有一個這樣命名的 HTML 模板檔案存放在 templates 目錄中。

    html_output = template.render(stocks=df.to_dict(orient='records')) # 模板渲染的過程，這裡將資料傳遞給模板進行渲染。
    # 將 pandas DataFrame 轉換為字典列表，每個列表項目代表 DataFrame 中的一行，這使得資料可以在範本中按照預定的 HTML 結構進行展示


    with open('stock_report.html', 'w') as f: 
        f.write(html_output)
    # 將渲染後的 HTML 輸出儲存到一個名為 stock_report.html 的檔案中。使用 with open(...) 語句確保檔案正確開啟且最終會被關閉。 
    # 'w' 參數表示以寫入模式開啟文件，如果文件已存在，則覆蓋原有內容。

# 交易參數設定和執行
parameters = { 
    "rsi_period": 14,
    "low_rsi": 30,
    "high_rsi": 70,
    "ema_period": 26,
}
for i in nlp2.get_files_from_dir("data"):
    try:
        url = i
        should_buy, should_sell, today_close_price = recommend_stock(url, parameters)
        if should_sell or should_buy:
            print(
                f"{i.split('/')[-1].split('.')[0]} Should buy today: {should_buy}, Should sell today: {should_sell}, Recommended price: {today_close_price}")
    except Exception as e:
        pass
generate_report(list(nlp2.get_files_from_dir("data")), parameters)
