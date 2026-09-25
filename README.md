# NBA Win Probability Model

**Author:** Sandi  
**Dataset Source:** `data/TeamStatistics.csv` (146,560 raw historical records)

A production-grade data engineering pipeline and predictive modeling engine built to forecast NBA game outcomes. Moving beyond simple static win-loss records, this framework implements an exponentially decaying recency weight matrix to capture shifting team momentum, adjusts dynamically for Strength of Schedule (SoS), and utilizes machine learning calibration to output true, actionable win probabilities.

---

## 🚀 1. Core Pipeline Logic: Concept-to-Code Mapping

The architecture is built cleanly upon academic statistical theories transformed into high-performance Python operations. The table below outlines how data theory translates directly to the code base:

| Analytical Concept | Code Element / Function | Mathematical Operational Mechanics |
| :--- | :--- | :--- |
| **Data Pre-processing & Structural Cleaning** | `load_and_clean_data()` | Drops records missing vital metrics. Resolves row perspective mismatches by mirroring `gameId` instances to map point margins dynamically against explicit opponents. Implements a performance data constraint window via `SEASON_START` to optimize dataframe execution speeds. |
| **Outlier Mitigation** | IQR Outlier Rule (Strict $k = 3$) | Uses the statistical Interquartile Range formula: $[Q1 - 3 \times IQR, Q3 + 3 \times IQR]$ to screen point spreads. This deliberately strict multiplier purges extreme benched-starter garbage time blowouts while safely preserving genuine elite performance variance. |
| **Recency-Weighted Team Performance** | `calculate_advanced_ratings()` | Computes an exponential decay sequence tracking team margin performance. Every rolling 7-day period that elapses reduces an older game's mathematical weight by a precise factor ($\alpha = 0.92$), placing structural emphasis on active "hot streaks." |
| **Scheduling Discrepancy Adjustment** | Strength of Schedule (SoS) Pass | Adjusts a franchise's baseline point rating by computing the average raw rating of all opponents faced within the lookback window. Refines ratings to reward squads navigating brutal scheduling stretches. |
| **Probability Squashing & Parameter Optimization** | `sklearn.linear_model.LogisticRegression` | Replaces rigid hand-tuned denominators with an empirical mathematical link function: $1 / (1 + e^{-\text{gap}/\text{scale}})$. Translates calculated efficiency point spreads into bounded, beautifully calibrated win percentages between $0\%$ and $100\%$. |

---

## 📈 2. Empirical Validation Results & Hyperparameter Tuning

To establish true mathematical validity and prevent over-optimism or data leakage, the pipeline enforces a strict **chronological 80/20 train/test split**. The model builds its team profiles completely on historical timelines before evaluating accuracy on unobserved future contests.

> 🧠 **Hyperparameter Custom Tuning Note:** During experimental cycles, I tweaked the recency decay factor to perfection. Moving the factor from the standard baseline of `0.85` up to a refined `0.92` dramatically expanded the model's predictive memory window. This allows the system to retain crucial structural context regarding a team's foundational talent level without overreacting to isolated, short-term noise. Coupled with reducing `HOME_ADVANTAGE` to a modern `2.5` scale and implementing `scikit-learn` parameter matching, the model achieved a massive performance leap.

### Chronological Back-Test Performance Spectrum (6,129 Held-Out Games)
* **Random Selection Baseline:** Log-Loss: `0.6931` | Win Accuracy: `50.00%`
* **Hand-Tuned Initial Pipeline (Decay 0.85):** Log-Loss: `0.5985` | Win Accuracy: `67.24%`
* **Machine Learning + SoS Optimized Model (Decay 0.92):** Log-Loss: **`0.5692`** | Win Accuracy: **`69.45%`**

### Calibration Metrics Analysis
Probability distributions match real-world frequency outcomes. Following probability binning diagnostics, games assigned an expected win clip within our $70\%$ probability bucket yielded an empirical, real-world win conversion rate of **$71.4\%$**, confirming predictive reliability.

---

## 🛠️ 3. Installation & Operational Deployment

### Step 1: Environment Isolation
Initialize your local environment containment shell to avoid global dependency cross-contamination:
```powershell
python -m venv .venv
# Activate on Windows:
.venv\Scripts\Activate.ps1