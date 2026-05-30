"""
Student Performance Predictor
- Generates synthetic student data
- Trains Linear Regression and Random Forest models
- Displays 6 graphs as pop-up windows
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

sns.set_theme(style="whitegrid", palette="muted")

# ── 1. Generate data ───────────────────────────────────────────────────────────
np.random.seed(42)
N = 300

study_hours   = np.random.uniform(1, 10, N)
attendance    = np.random.uniform(50, 100, N)
prev_gpa      = np.random.uniform(1.5, 4.0, N)
sleep_hours   = np.random.uniform(4, 9, N)
tutoring      = np.random.randint(0, 2, N)          # 0 = No, 1 = Yes
noise         = np.random.normal(0, 4, N)

final_grade = (
    study_hours  * 3.5 +
    attendance   * 0.3 +
    prev_gpa     * 9.0 +
    sleep_hours  * 1.0 +
    tutoring     * 5.0 +
    noise
)
final_grade = np.clip(final_grade, 0, 100)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance":  attendance,
    "prev_gpa":    prev_gpa,
    "sleep_hours": sleep_hours,
    "tutoring":    tutoring,
    "final_grade": final_grade,
})

def grade_letter(g):
    if g >= 90: return "A"
    if g >= 80: return "B"
    if g >= 70: return "C"
    if g >= 60: return "D"
    return "F"

df["grade"] = df["final_grade"].apply(grade_letter)

print(f"Dataset created: {N} students")
print(df.describe().round(2))

# ── 2. Train / test split ──────────────────────────────────────────────────────
FEATURES = ["study_hours", "attendance", "prev_gpa", "sleep_hours", "tutoring"]
X = df[FEATURES].values
y = df["final_grade"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 3. Train models ────────────────────────────────────────────────────────────
lr  = LinearRegression()
rf  = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

for name, pred in [("Linear Regression", lr_pred), ("Random Forest", rf_pred)]:
    r2   = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name:20s} | R² = {r2:.3f} | RMSE = {rmse:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
#  GRAPHS
# ══════════════════════════════════════════════════════════════════════════════

# ── Graph 1: Grade distribution ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["final_grade"], bins=25, color="steelblue", edgecolor="white")
axes[0].set_title("Final Grade Distribution")
axes[0].set_xlabel("Grade")
axes[0].set_ylabel("Count")

grade_order = ["A", "B", "C", "D", "F"]
counts = df["grade"].value_counts().reindex(grade_order, fill_value=0)
axes[1].bar(counts.index, counts.values, color=["#2ecc71","#3498db","#f1c40f","#e67e22","#e74c3c"])
axes[1].set_title("Students per Letter Grade")
axes[1].set_xlabel("Grade")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

# ── Graph 2: Correlation heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
corr = df[FEATURES + ["final_grade"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ── Graph 3: Feature vs final grade (scatter grid) ─────────────────────────────
fig, axes = plt.subplots(1, len(FEATURES), figsize=(16, 4))
for ax, feat in zip(axes, FEATURES):
    ax.scatter(df[feat], df["final_grade"], alpha=0.4, s=15, color="steelblue")
    m, b = np.polyfit(df[feat], df["final_grade"], 1)
    xline = np.linspace(df[feat].min(), df[feat].max(), 100)
    ax.plot(xline, m * xline + b, color="red", lw=1.5)
    ax.set_xlabel(feat.replace("_", " ").title())
    ax.set_ylabel("Final Grade")
    ax.set_title(feat.replace("_", " ").title())

plt.suptitle("Each Feature vs Final Grade", y=1.02)
plt.tight_layout()
plt.show()

# ── Graph 4: Actual vs Predicted ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, pred) in zip(axes, [("Linear Regression", lr_pred), ("Random Forest", rf_pred)]):
    ax.scatter(y_test, pred, alpha=0.6, s=25, color="darkorange")
    mn, mx = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Grade")
    ax.set_ylabel("Predicted Grade")
    ax.set_title(f"Actual vs Predicted\n{name}")
    r2 = r2_score(y_test, pred)
    ax.annotate(f"R² = {r2:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=10)
    ax.legend()

plt.tight_layout()
plt.show()

# ── Graph 5: Residuals ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (name, pred) in zip(axes, [("Linear Regression", lr_pred), ("Random Forest", rf_pred)]):
    residuals = y_test - pred
    ax.scatter(pred, residuals, alpha=0.6, s=25, color="mediumpurple")
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Predicted Grade")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title(f"Residual Plot — {name}")

plt.tight_layout()
plt.show()

# ── Graph 6: Feature importance (Random Forest) ────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
colors = sns.color_palette("viridis", len(importances))
importances.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Feature Importance — Random Forest")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.show()
