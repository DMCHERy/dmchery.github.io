import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import openai
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def fetch_markets():
    url = "https://api.manifold.markets/v0/markets"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def filter_binary_resolved(markets):
    return [
        m for m in markets
        if m['outcomeType'] == 'BINARY'
        and m.get('isResolved') is True
        and m.get('resolution') in ['YES', 'NO']
    ]

def get_gpt_opinion_summary(question):
    try:
        messages = [
            {"role": "system", "content": "You are a probability estimation expert."},
            {"role": "user", "content": f"Estimate the probability (between 0 and 1) that the answer to the question will be YES: {question}\nRespond ONLY with the number."}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3
        )
        reply = response.choices[0].message.content.strip()
        score = float(reply) if 0 <= float(reply) <= 1 else 0.5
        return {'yes_confidence': score}
    except:
        return {'yes_confidence': 0.5}

def build_feature_set(markets):
    data = []
    for m in markets:
        try:
            gpt = get_gpt_opinion_summary(m['question'])
            row = {
                'id': m['id'],
                'question': m['question'],
                'questionLength': len(m['question']),
                'volume': m.get('volume', 0),
                'numTraders': m.get('uniqueBettorCount', 0),
                'timeOpen': (m['closeTime'] - m['createdTime']) / (1000 * 60 * 60 * 24),
                'gpt_pos_confidence': gpt['yes_confidence'],
                'label': 1 if m['resolution'] == 'YES' else 0
            }
            data.append(row)
        except:
            continue
    return pd.DataFrame(data)

def train_and_evaluate(df):
    X = df.drop(columns=["id", "question", "label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")

    importance = sorted(zip(X.columns, clf.feature_importances_), key=lambda x: -x[1])
    print("\nFeature Importance:")
    for feat, score in importance:
        print(f"{feat}: {score:.4f}")

    return clf

def fetch_unresolved_binary():
    url = "https://api.manifold.markets/v0/markets"
    response = requests.get(url)
    response.raise_for_status()
    return [
        m for m in response.json()
        if m['outcomeType'] == 'BINARY'
        and not m.get('isResolved', False)
    ]

def build_unresolved_features(markets):
    data = []
    ids = []
    questions = []
    links = []
    dates = []
    for m in markets:
        try:
            gpt = get_gpt_opinion_summary(m['question'])
            row = {
                'questionLength': len(m['question']),
                'volume': m.get('volume', 0),
                'numTraders': m.get('uniqueBettorCount', 0),
                'timeOpen': (m['closeTime'] - m['createdTime']) / (1000 * 60 * 60 * 24),
                'gpt_pos_confidence': gpt['yes_confidence']
            }
            data.append(row)
            ids.append(m['id'])
            questions.append(m['question'])
            links.append(f"https://manifold.markets/{m['creatorUsername']}/{m['slug']}")
            dates.append(datetime.utcnow().strftime('%Y-%m-%d'))
        except:
            continue

    df = pd.DataFrame(data)
    df['id'] = ids
    df['question'] = questions
    df['link'] = links
    df['date'] = dates
    return df

def predict_unresolved(model):
    print("\nFetching unresolved binary markets...")
    unresolved = fetch_unresolved_binary()
    if not unresolved:
        print("No unresolved binary markets found.")
        return

    df_features = build_unresolved_features(unresolved)
    X = df_features.drop(columns=["id", "question", "link", "date"])

    print("Predicting...")
    df_features['predicted_prob_yes'] = model.predict_proba(X)[:, 1]

    top = df_features.sort_values(by='predicted_prob_yes', ascending=False).head(10)
    print("\nTop Predictions:")
    print(top[['question', 'predicted_prob_yes']])

    df_features.to_csv("manifold_predictions.csv", index=False)
    print("Predictions saved to 'manifold_predictions.csv'")

    # Save subset for frontend as JSON
    predictions_json = df_features[['question', 'predicted_prob_yes', 'link', 'date']].copy()
    predictions_json.rename(columns={"predicted_prob_yes": "probability"}, inplace=True)
    predictions_json.to_json("predictions.json", orient="records", indent=2)
    print("Predictions also saved to 'predictions.json'")

if __name__ == "__main__":
    try:
        print("Fetching markets...")
        markets = fetch_markets()
        resolved = filter_binary_resolved(markets)

        print("\nBuilding features...")
        df = build_feature_set(resolved)

        if df.empty:
            print("No resolved binary markets available.")
        else:
            print(f"Training on {len(df)} resolved markets...")
            clf = train_and_evaluate(df)
            predict_unresolved(clf)

    except Exception as e:
        print(f"Error: {e}")
