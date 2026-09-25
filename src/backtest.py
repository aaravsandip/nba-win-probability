"""Honest walk-forward backtest.

For every game day we rate teams using ONLY earlier games, predict that day's
games, then move to the next day. Nothing from the future leaks in.

Usage:
    python src/backtest.py --data data/TeamStatistics.csv --tune 2024 --test 2025 2026
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from data import load_games
from model import team_ratings, rating_gap, win_probability

DECAYS = [0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 1.00]  # 1.00 = no recency weighting at all


def add_gaps(games, seasons, decay, sos):
    """Return the games from `seasons`, each with the rating gap known BEFORE tip-off."""
    target = games[games["season"].isin(seasons)]
    rows = []
    for day, todays in target.groupby("date"):
        history = games[(games["date"] < day) & (games["date"] >= day - pd.Timedelta(days=400))]
        ratings = team_ratings(history, day, decay=decay, sos=sos)
        for _, g in todays.iterrows():
            rows.append({**g.to_dict(), "gap": rating_gap(ratings, g["home"], g["away"])})
    return pd.DataFrame(rows)


def fit_curve(df):
    """Learn a and b in P(home win) = 1/(1+e^-(a*gap+b)) with logistic regression."""
    lr = LogisticRegression(C=1e6)  # large C = essentially no penalty
    lr.fit(df[["gap"]].values, df["home_win"].values)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def score(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(np.mean((p > 0.5) == (y == 1))),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def record_baseline(games, test):
    """Baseline: pick whichever team has the better win % so far this season."""
    correct = []
    for _, g in test.iterrows():
        before = games[(games["season"] == g["season"]) & (games["date"] < g["date"])]

        def win_pct(team):
            h = before[before["home"] == team]["home_win"]
            a = 1 - before[before["away"] == team]["home_win"]
            n = len(h) + len(a)
            return (h.sum() + a.sum()) / n if n else 0.5

        hp, ap = win_pct(g["home"]), win_pct(g["away"])
        pick_home = hp >= ap  # ties go to the home team
        correct.append(pick_home == (g["home_win"] == 1))
    return float(np.mean(correct))


def calibration_plot(y, p, path):
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        mask = idx == b
        if mask.sum() >= 10:  # skip nearly empty buckets
            xs.append(p[mask].mean())
            ys.append(y[mask].mean())
            ns.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(xs, ys, marker="o", label="Model")
    for x, yv, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, yv), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set_xlabel("Predicted home win probability")
    ax.set_ylabel("Actual home win rate")
    ax.set_title("Calibration (test seasons)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return pd.DataFrame({"predicted": xs, "actual": ys, "games": ns})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join("data", "TeamStatistics.csv"))
    parser.add_argument("--tune", type=int, nargs="+", default=[2024])  # choose settings
    parser.add_argument("--test", type=int, nargs="+", default=[2025, 2026])  # never seen
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)

    games = load_games(args.data, first_season=min(args.tune) - 1)
    first, last = games["season"].min(), games["season"].max()
    print(f"Loaded {len(games)} games from seasons {first}-{last}")

    # ---- Step A: choose decay and SoS using ONLY the tuning season(s) ----
    print("\nTuning on season(s)", args.tune)
    tuning_rows = []
    for sos in [False, True]:
        for decay in DECAYS:
            tune_df = add_gaps(games, args.tune, decay, sos)
            a, b = fit_curve(tune_df)
            s = score(tune_df["home_win"].values, win_probability(tune_df["gap"], a, b))
            tuning_rows.append({"decay": decay, "sos": sos, "a": a, "b": b, **s})
            print(f"  decay={decay:.2f} sos={sos!s:5} "
                  f"log_loss={s['log_loss']:.4f} acc={s['accuracy']:.3f}")
    tuning = pd.DataFrame(tuning_rows)
    tuning.to_csv("results/tuning.csv", index=False)
    best = tuning.sort_values("log_loss").iloc[0]
    decay, sos = float(best["decay"]), bool(best["sos"])
    a, b = float(best["a"]), float(best["b"])
    print(f"\nBest: decay={decay}, sos={sos}, a={a:.4f}, b={b:.4f}")
    print(f"Home-court edge = {b / a:.2f} points")

    # ---- Step B: test on seasons the model has never seen ----
    test_df = add_gaps(games, args.test, decay, sos)
    y = test_df["home_win"].values
    p = win_probability(test_df["gap"], a, b)
    test_df["p_home"] = p
    test_df.to_csv("results/backtest_predictions.csv", index=False)

    home_rate = games[games["season"].isin(args.tune)]["home_win"].mean()
    rows = [
        {"method": "Model", **score(y, p)},
        {"method": "Always pick home team", **score(y, np.full(len(y), home_rate))},
        {"method": "Coin flip (50%)", **score(y, np.full(len(y), 0.5)), "accuracy": 0.5},
        {"method": "Better record so far", "accuracy": record_baseline(games, test_df),
         "log_loss": np.nan, "brier": np.nan},
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv("results/metrics.csv", index=False)
    cal = calibration_plot(y, p, "results/calibration.png")
    cal.to_csv("results/calibration.csv", index=False)

    print(f"\nTest season(s) {args.test}: {len(test_df)} games")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nCalibration buckets:")
    print(cal.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nSaved everything in the results/ folder.")


if __name__ == "__main__":
    main()