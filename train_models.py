import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle

df = pd.read_csv("strava_data.csv")

# Keep required columns
df = df[["date",
    "sport_type",
    "distance",
    "moving_time",
    "average_speed",
    "average_heartrate",
    "pr_count"]]

# Keep only HYROX related sport types
df = df[df['sport_type'].isin(['VirtualRide', 'Ride', 'Run'])]

# Basic Preprocessing
df = df.drop_duplicates(subset=['date'])
df["average_heartrate"] = df["average_heartrate"].fillna(df["average_heartrate"].median())

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Training load feature calculation
df["training_load"] = (df["distance"].fillna(0) * df["average_speed"].fillna(0))

# PB label (ground truth from dataset)
df["is_personal_best"] = (df["pr_count"] > 0).astype(int)

df["prev_date"] = df["date"].shift(1)
df["day_gap"] = (df["date"] - df["prev_date"]).dt.days

# New streak starts if gap > 1 day
df["streak_break"] = (df["day_gap"] > 1).astype(int)

# Streak length
streak = 0
streaks = []

for b in df["streak_break"]:
    if b == 1:
        streak = 1
    else:
        streak += 1
    streaks.append(streak)

df["streak_length"] = streaks

df["next_is_pb"] = df["is_personal_best"].shift(-1)
df["next_streak_break"] = df["streak_break"].shift(-1)

df = df.dropna().reset_index(drop=True)


def run_logistic_regression(X, y):
    # Test train split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False  # keep time order
    )

    # Baseline Classifier: Logistic Regression
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(X_train_scaled, y_train)
    y_pred_lr = logistic_regression_model.predict(X_test_scaled)

    acc_lr = accuracy_score(y_test, y_pred_lr)
    return logistic_regression_model, acc_lr

def run_random_forest(X, y):
    # Test train split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False  # keep time order
    )

    # Medium Classifier: Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    return rf, acc_rf


def run_xgboost(X, y):
    # Test train split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False  # keep time order
    )

    # Advanced Classifier: XGBoost Classifier
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    return xgb, acc_xgb

FEATURES = [
    "training_load",
    "average_heartrate",
    "moving_time",
    "streak_length"
]

X = df[FEATURES].fillna(0)

# Firstly, training the models on personal best data
trained_lr_model_pb, accuracy_lr_pb = run_logistic_regression(X, df["next_is_pb"])
trained_rf_model_pb, accuracy_rf_pb = run_random_forest(X, df["next_is_pb"])
trained_xgb_model_pb, accuracy_xgb_pb = run_xgboost(X, df["next_is_pb"])

# Secondly, training the models on streak data
trained_lr_model_streak, accuracy_lr_streak = run_logistic_regression(X, df["next_streak_break"])
trained_rf_model_streak, accuracy_rf_streak = run_random_forest(X, df["next_streak_break"])
trained_xgb_model_streak, accuracy_xgb_streak = run_xgboost(X, df["next_streak_break"])

print(f"Accuracy of LogisticRegression on personal best data: {accuracy_lr_pb}")
print(f"Accuracy of LogisticRegression on streak data: {accuracy_lr_streak}")
print(f"Accuracy of RandomForest on personal best data: {accuracy_rf_pb}")
print(f"Accuracy of RandomForest on streak data: {accuracy_rf_streak}")
print(f"Accuracy of XGB on personal best data: {accuracy_xgb_pb}")
print(f"Accuracy of XGB on streak data: {accuracy_xgb_streak}")

# Logistic Regression is selected as the production model
# because it slightly outperforms the other models

# dumping the lr models as .pkl files for use later
with open("model_pkl_files/lr_model_pb.pkl", "wb") as f:
    pickle.dump(trained_lr_model_pb, f)

with open("model_pkl_files/lr_model_streak.pkl", "wb") as f:
    pickle.dump(trained_lr_model_streak, f)
