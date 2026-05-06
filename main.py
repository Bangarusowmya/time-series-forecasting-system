"""
main.py — Entry point for training the forecasting system.

Usage:
  # Train all states (slow — includes LSTM)
  python main.py

  # Train specific states only
  python main.py --states "Texas" "California" "Florida"

  # Skip LSTM (faster for testing)
  python main.py --skip-lstm

  # Quick test: 3 states, no LSTM
  python main.py --states "Texas" "California" "Florida" --skip-lstm
"""

import os
import argparse
import sys

# ensure project root is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.train_pipeline import run_full_pipeline
from src.logger import get_logger

logger = get_logger("main")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sales_data.xlsx")


def parse_args():
    parser = argparse.ArgumentParser(description="Train sales forecasting models")
    parser.add_argument(
        "--states",
        nargs="+",
        default=None,
        help="List of states to train on (default: all states)",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        default=False,
        help="Skip LSTM training (faster for testing)",
    )
    parser.add_argument(
        "--data-path",
        default=DATA_PATH,
        help="Path to the Excel data file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)

    logger.info(f"Data path: {args.data_path}")
    if args.states:
        logger.info(f"Training states: {args.states}")
    else:
        logger.info("Training all available states")

    if args.skip_lstm:
        logger.info("LSTM training will be skipped")

    run_full_pipeline(
        data_path=args.data_path,
        states_to_train=args.states,
        skip_lstm=args.skip_lstm,
    )

    print("\nDone! To start the API:")
    print("  uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload\n")


if __name__ == "__main__":
    main()
