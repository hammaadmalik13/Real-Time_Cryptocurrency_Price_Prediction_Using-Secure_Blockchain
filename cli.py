from __future__ import annotations

import argparse
import json

from crypto_lstm.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_FORECAST_STEPS,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PLOT_PATH,
    DEFAULT_SCALER_PATH,
    DEFAULT_SIGNAL_THRESHOLD,
    WINDOW_SIZE,
)
from crypto_lstm.data import fetch_binance_klines, load_price_data, save_price_data
from crypto_lstm.features import prepare_feature_frame
from crypto_lstm.forecasting import (
    forecast_prices_and_signals,
    format_forecast_table,
    simulate_portfolio,
)
from crypto_lstm.model import load_metadata, result_as_dict, train_lstm_model


def collect(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect crypto OHLCV data from Binance.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", default=str(DEFAULT_DATA_PATH))
    args = parser.parse_args(argv)

    df = fetch_binance_klines(args.symbol, args.interval, args.limit)
    output_path = save_price_data(df, args.output)
    print(f"Saved {len(df)} rows to {output_path}")


def train(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the LSTM price prediction model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--scaler-path", default=str(DEFAULT_SCALER_PATH))
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--plot-path", default=str(DEFAULT_PLOT_PATH))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--include-indicators", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    df = load_price_data(args.data)
    result = train_lstm_model(
        df,
        include_indicators=args.include_indicators,
        window_size=args.window_size,
        train_ratio=args.train_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        metadata_path=args.metadata_path,
        plot_path=None if args.no_plot else args.plot_path,
    )
    print(json.dumps(result_as_dict(result), indent=2))


def forecast(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Forecast future prices from saved artifacts.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--scaler-path", default=str(DEFAULT_SCALER_PATH))
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--steps", type=int, default=DEFAULT_FORECAST_STEPS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIGNAL_THRESHOLD)
    parser.add_argument("--initial-capital", type=float, default=10_000)
    args = parser.parse_args(argv)

    from tensorflow.keras.models import load_model
    import joblib

    metadata = load_metadata(args.metadata_path)
    include_indicators = bool(metadata.get("include_indicators", False))
    window_size = int(metadata.get("window_size", WINDOW_SIZE))

    df = load_price_data(args.data)
    feature_frame, feature_names = prepare_feature_frame(df, include_indicators)
    scaler = joblib.load(args.scaler_path)
    scaled_data = scaler.transform(feature_frame[feature_names])
    model = load_model(args.model_path)

    points = forecast_prices_and_signals(
        model,
        scaled_data,
        scaler,
        feature_frame,
        feature_names,
        window_size=window_size,
        steps=args.steps,
        threshold=args.threshold,
    )
    print(format_forecast_table(points))
    print(f"\nSimulated final value: {simulate_portfolio(points, args.initial_capital):.2f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Crypto LSTM project commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("collect", help="Collect OHLCV data from Binance.")
    subparsers.add_parser("train", help="Train the LSTM model.")
    subparsers.add_parser("forecast", help="Forecast prices from saved artifacts.")
    args, remaining = parser.parse_known_args(argv)

    commands = {
        "collect": collect,
        "train": train,
        "forecast": forecast,
    }
    commands[args.command](remaining)


if __name__ == "__main__":
    main()
