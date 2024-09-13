from datetime import datetime, timedelta
import os
import pandas as pd
import vectorbt as vbt
import twstock

codes = twstock.codes

def get_past_x_days(stock_num, days=60):
    csv_path = f'./data/{stock_num}.csv'

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and len(df) > 1:
                print(f"Data for {stock_num} already exists and is not empty. Skipping download.")
                return
        except pd.errors.EmptyDataError:
            pass

    tz_taipei = 'Asia/Taipei'
    today = datetime.now()

    start_date = today - timedelta(days=days)
    end_date = today

    yf_data = vbt.YFData.download(
        f"{stock_num}.TW",
        start=start_date,
        end=end_date,
        interval='1d',  # 使用日線資料
        missing_index='drop',
        timezone=tz_taipei
    )

    data = yf_data.get()
    data.reset_index(inplace=True)
    data['Datetime'] = data['Datetime'].dt.tz_convert(tz_taipei)
    data.to_csv(csv_path, index=False)
    print(f"Downloaded data for {stock_num}")

for k, v in codes.items():
    if v.market == '上市' and (v.type == '股票' or v.type == 'ETF'):
        get_past_x_days(k)
