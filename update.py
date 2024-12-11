from datetime import datetime, timedelta
import os
import pandas as pd
import pytz
import vectorbt as vbt
import twstock

codes = twstock.codes

def clean_csv(csv_path):
    """ Clean the CSV file by removing rows with incorrect number of columns. """
    with open(csv_path, 'r') as file:
        lines = file.readlines()

    # Count the number of columns in the header
    num_columns = len(lines[0].split(','))

    with open(csv_path, 'w') as file:
        for line in lines:
            if len(line.strip()) > 0 and len(line.split(',')) == num_columns:
                file.write(line)

def get_data_since_last_record(stock_num, base_path='./data/'):
    csv_path = f'{base_path}{stock_num}.csv'
    tz_taipei = pytz.timezone('Asia/Taipei')

    # 計算開始時間
    if os.path.exists(csv_path):
        clean_csv(csv_path)
        data = pd.read_csv(csv_path)
        if not data.empty:
            # 修正警告：加上 utc=True
            data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce', utc=True)
            data.dropna(subset=['Datetime'], inplace=True)  # 移除無效日期行
            data.set_index('Datetime', inplace=True)
            data.index = data.index.tz_convert(tz_taipei)  # 將 UTC 轉換為台北時間
            last_record_date = data.index[-1]
            start_date = last_record_date + timedelta(days=1)
        else:
            start_date = datetime.now(tz_taipei) - timedelta(days=90)  # 預設抓取最近90天
    else:
        start_date = datetime.now(tz_taipei) - timedelta(days=90)

    # 計算結束日期
    end_date = datetime.now(tz_taipei) - timedelta(hours=2)  # 減去2小時的緩衝時間

    # 從 Yahoo Finance 下載數據
    try:
        yf_data = vbt.YFData.download(
            f"{stock_num}.TW",
            start=start_date,
            end=end_date,
            interval='1d',
            missing_index='drop'
        )
    except Exception as e:
        print(f"Error fetching data for {stock_num}: {e}")
        return

    new_data = yf_data.get()
    if new_data.empty:
        print(f"No new data for {stock_num}")
        return

    # 數據處理
    new_data.reset_index(inplace=True)
    new_data['Datetime'] = new_data['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')  # 處理時區
    new_data.drop_duplicates(subset=['Datetime'], inplace=True)  # 移除重複數據

    # 寫入或更新 CSV
    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path)
        combined_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=['Datetime']).reset_index(drop=True)
        combined_data.to_csv(csv_path, index=False)
    else:
        new_data.to_csv(csv_path, index=False)

    print(f"Updated data for {stock_num}. Start: {start_date}, End: {end_date}")

# 主程式
if __name__ == "__main__":
    for k, v in codes.items():
        if v.market == '上市' and (v.type == '股票' or v.type == 'ETF'):
            print(f"Fetching data for {k} ({v.name})...")
            get_data_since_last_record(k)
