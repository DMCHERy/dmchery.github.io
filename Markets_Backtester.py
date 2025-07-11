

import pandas as pd
import requests
from datetime import datetime

def fetch_all_markets():
    url = "https://api.manifold.markets/v0/markets"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def filter_resolved_ids():
    markets = fetch_all_markets()
    resolved = [m for m in markets if m['outcomeType'] == 'BINARY' and m.get('isResolved')]
    return {m['id']: m['resolution'] for m in resolved}

def load_predictions(file_path="manifold_predictions.csv"):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print("Prediction file not found.")
        return None

def evaluate_predictions(predictions_df, actual_outcomes):
    matched = predictions_df[predictions_df['id'].isin(actual_outcomes.keys())].copy()
    matched['actual'] = matched['id'].map(actual_outcomes)
    matched['actual_label'] = matched['actual'].apply(lambda x: 1 if x == 'YES' else 0)
    matched['predicted_label'] = matched['predicted_prob_yes'].apply(lambda x: 1 if x >= 0.5 else 0)
    matched['correct'] = matched['predicted_label'] == matched['actual_label']

    win_rate = matched['correct'].mean()
    print(f"\nBacktest Results:")
    print(f"Total matched: {len(matched)}")
    print(f"Accuracy: {win_rate * 100:.2f}%")

    print("\nSample Incorrect Predictions:")
    print(matched[matched['correct'] == False][['question', 'predicted_prob_yes', 'actual']].head(5))

    matched.to_csv("backtest_results.csv", index=False)
    print("\nSaved detailed results to backtest_results.csv")

if __name__ == "__main__":
    print("Loading predictions...")
    predictions = load_predictions()

    if predictions is not None:
        print("Fetching actual market outcomes...")
        outcomes = filter_resolved_ids()
        evaluate_predictions(predictions, outcomes)
