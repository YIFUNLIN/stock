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

    if os.path.exists(csv_path):
        clean_csv(csv_path)
        data = pd.read_csv(csv_path)
        if not data.empty:
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data.set_index('Datetime', inplace=True)
            data.index = data.index.tz_localize('UTC').tz_convert(tz_taipei)
            last_record_date = data.index[-1]
            start_date = last_record_date + timedelta(days=1)
        else:
            start_date = datetime.now(tz_taipei) - timedelta(days=60)
    else:
        start_date = datetime.now(tz_taipei) - timedelta(days=60)

    end_date = datetime.now(tz_taipei) - timedelta(minutes=30)  # 確保資料已更新，減去30分鐘

    yf_data = vbt.YFData.download(
        f"{stock_num}.TW",
        start=start_date,
        end=end_date,
        interval='1d',
        missing_index='drop',
        timezone='Asia/Taipei'
    )

    new_data = yf_data.get()
    if new_data.empty:
        print(f"No new data for {stock_num}")
        return

    new_data.reset_index(inplace=True)
    new_data['Datetime'] = new_data['Datetime'].dt.tz_convert('Asia/Taipei')

    if os.path.exists(csv_path):
        new_data.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        new_data.to_csv(csv_path, index=False)

    print(f"Updated data for {stock_num}")

for k, v in codes.items():
    if v.market == '上市' and (v.type == '股票' or v.type == 'ETF'):
        get_data_since_last_record(k)
