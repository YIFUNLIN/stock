from datetime import datetime, timedelta
import os
import pandas as pd
import pytz
import vectorbt as vbt
import twstock

'''
這段程式碼是設計用於從 Yahoo Finance 自動更新台灣股市的股票和 ETF 資料，
並將更新的數據保存為 CSV 文件。會將此工作被放在 GitHub Actions 中執行，以實現定期的資料更新
'''

codes = twstock.codes #  獲取所有台灣股市的股票代碼及相關信息
'''
裡面長這樣
700001': StockCodeInfo(type='上櫃認購(售)權證', code='700001', name='新普台新85購01', ISIN='TW18Z7000011', start='2018/10/01', market='上櫃', group='', CFI='RWSCCE'),
 '700002': StockCodeInfo(type='上櫃認購(售)權證', code='700002', name='環球晶台新84購01', ISIN='TW18Z7000029', start='2018/10/01', market='上櫃', group='', CFI='RWSCCE')
 '''

def get_data_since_last_record(stock_num, base_path='./data/'): #  函式負責獲取指定股票號碼的最新數據
    csv_path = f'{base_path}{stock_num}.csv'
    tz_taipei = pytz.timezone('Asia/Taipei')
    today = datetime.now(tz_taipei).replace(hour=0, minute=0, second=0, microsecond=0)  # Reset to start of day

    if os.path.exists(csv_path):  # 若csv存在，則讀取
        data = pd.read_csv(csv_path, header=0)
        if not data.empty: # 若資料內不為空
            try:
                last_record_date = pd.to_datetime(data['Datetime'].iloc[-1]).tz_convert('Asia/Taipei') # 讀取最後一筆資料(以確認上次更新到哪)
                start_date = last_record_date + timedelta(minutes=5) # 若成功讀取，將該時間(最後一筆資料日期)再加5分鐘，以獲取新數據
            except Exception as e: # 若無法成功讀取，程式會從今天往回推59天開始抓取數據，一直抓取到當前的日期
                print(f"Error parsing last record date: {e}")
                start_date = today - timedelta(days=59)
        else:
            start_date = today - timedelta(days=59) 
    else:
        start_date = today - timedelta(days=59)

    end_date = today - timedelta(minutes=5)  # 每天會更新到13:55才停止 (資料的結束時間設定為當前時間減去五分鐘，以確保資料的可用性和完整性。)

    yf_data = vbt.YFData.download(
        f"{stock_num}.TW",
        start=start_date.strftime('%Y-%m-%d %H:%M:%S'),
        end=end_date.strftime('%Y-%m-%d %H:%M:%S'),
        interval='5m',
        missing_index='drop'
    )

    new_data = yf_data.get()

    if os.path.exists(csv_path):
        new_data.to_csv(csv_path, mode='a', header=False)
    else:
        new_data.to_csv(csv_path)

    return new_data

for k, v in codes.items(): # 迭代更新股票數據
    if v.market == '上市' and (v.type == '股票' or v.type == 'ETF'): # 只抓「上市」且類型為「股票」或「ETF」的股票
        new_data = get_data_since_last_record(k)
        print(f"Updated data for {k}")
