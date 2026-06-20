import requests
import pandas as pd
import os

def get_binance_data(symbol="BTCUSDT", interval="1h", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    response = requests.get(url, params=params)
    response.raise_for_status() 
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    df[["open","high","low","close","volume"]] = \
        df[["open","high","low","close","volume"]].astype(float)

    return df


if __name__ == "__main__":
    df = get_binance_data()

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/btc_data.csv", index=False)

    print("Data saved successfully!")