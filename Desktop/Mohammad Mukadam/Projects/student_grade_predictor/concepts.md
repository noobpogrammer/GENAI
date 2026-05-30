# Student Performance Predictor — Concepts Explained

This document explains every concept used in the project and how it directly affects the results.

---

## 1. Dataset & Features

### What is a Feature?
A **feature** (also called an input variable) is a measurable property of a student that the model uses to make a prediction. In this project the features are:

| Feature | What it measures |
|---|---|
| `study_hours` | How many hours per day the student studies |
| `attendance` | Percentage of classes attended |
| `prev_gpa` | GPA from the previous term (scale 0–4) |
| `sleep_hours` | Average sleep per night |
| `tutoring` | Whether the student has a private tutor (0 = No, 1 = Yes) |

### What is a Target Variable?
The **target** is what we want to predict — here it is `final_grade` (0–100). The model learns to map features → target.

### Why Synthetic Data?
Real student data is private. We generate realistic data mathematically so every concept can be demonstrated without privacy concerns. The grade formula is:

```
final_grade = study_hours×3.5 + attendance×0.3 + prev_gpa×9 + sleep_hours×1 + tutoring×5 + noise
```

The multipliers reflect how strongly each feature influences the grade. **Noise** (random variation) simulates the unpredictable real-world factors.

---

## 2. Train / Test Split

### What is it?
We split the dataset into two parts:
- **Training set (80%)** — the model learns patterns from this data.
- **Test set (20%)** — used only for evaluation; the model never sees it during training.

### Why does it matter?
Without a test set you cannot know if the model has actually learned general patterns or simply memorised the training data. Evaluating on unseen data gives an honest measure of real-world performance.

### Effect on this project
With 300 students, 240 go to training and 60 to testing. If the training score is much higher than the test score, the model is **overfitting** (memorising instead of learning).

---

## 3. Linear Regression

### What is it?
Linear Regression fits a straight line (or hyperplane in multiple dimensions) through the data by minimising the sum of squared errors between predictions and actual values.

The model equation:

```
predicted_grade = w1×study_hours + w2×attendance + w3×prev_gpa + w4×sleep_hours + w5×tutoring + bias
```

The algorithm finds the best values for the weights `w1…w5` and `bias`.

### Strengths
- Fast to train.
- Easy to interpret (each weight tells you the exact contribution of that feature).
- Works well when the relationship between features and target is truly linear.

### Weaknesses
- Cannot capture non-linear relationships.
- Sensitive to outliers.

### Effect on this project
Since the data was generated with a linear formula, Linear Regression performs very well here — R² ≈ 0.86. In real datasets with more complex patterns, it would likely perform worse than Random Forest.

---

## 4. Random Forest

### What is it?
A Random Forest builds many **Decision Trees** (typically 100+), each trained on a random subset of the data and a random subset of features. The final prediction is the **average** of all trees.

### Decision Tree (quick explanation)
A decision tree splits data by asking questions:
```
Is study_hours > 6?
  Yes → Is attendance > 80? → ...
  No  → Is prev_gpa > 2.5? → ...
```
Each leaf node outputs a predicted grade.

### Why "Random"?
- **Random samples** (bagging): each tree is trained on a different random sample of students — this reduces variance.
- **Random features**: each split considers only a random subset of features — this reduces correlation between trees.

### Strengths
- Handles non-linear relationships.
- Robust to outliers and noise.
- Provides feature importance scores.

### Weaknesses
- Harder to interpret than Linear Regression.
- Slower to train.
- Can overfit if trees are too deep (controlled by `max_depth`).

### Effect on this project
Since our data is linear, Random Forest (R² ≈ 0.78) slightly underperforms Linear Regression. With real, messy data the relationship would be reversed.

---

## 5. Model Evaluation Metrics

### R² (R-squared / Coefficient of Determination)
Measures how much of the variance in grades the model explains.

```
R² = 1 − (sum of squared residuals) / (total variance)
```

- **R² = 1.0** → perfect predictions
- **R² = 0.0** → model is no better than predicting the mean every time
- **R² < 0** → model is worse than guessing the mean

**In this project:** R² ≈ 0.86 means the model explains 86% of the variation in student grades.

### RMSE (Root Mean Squared Error)
The average prediction error in the same unit as the target (grade points).

```
RMSE = sqrt( mean( (actual − predicted)² ) )
```

An RMSE of 4.24 means predictions are off by about 4.24 grade points on average. Squaring penalises large errors more, so the model is pushed to avoid big mistakes.

### Which metric to trust?
- Use **R²** to compare models against each other.
- Use **RMSE** to understand the practical size of the errors (e.g. "off by 4 points on a 100-point scale").

---

## 6. Correlation

### What is it?
The **Pearson correlation coefficient** measures the linear relationship between two variables. It ranges from −1 to +1:

| Value | Meaning |
|---|---|
| +1 | Perfect positive relationship |
| 0 | No linear relationship |
| −1 | Perfect negative relationship |

### Effect on this project (Correlation Heatmap — Graph 2)
The heatmap reveals which features most strongly predict the final grade. In our data `prev_gpa` and `study_hours` have the strongest positive correlations with `final_grade`, which matches the weights in the data-generation formula. Features with low correlation (like `sleep_hours`) contribute less, but are still informative when combined with others.

---

## 7. Feature Importance

### What is it?
Random Forest measures how much each feature reduces prediction error across all trees. Features used at the top of many trees (where splits affect the most data) receive higher importance scores.

### Effect on this project (Graph 6)
`prev_gpa` and `study_hours` appear as the most important features because they have the largest coefficients in the data formula. `tutoring` scores moderately — it helps, but since it is binary (0 or 1) it has less range than continuous features.

Feature importance helps you understand **which inputs to focus on** if you were designing an intervention programme (e.g. prioritise tutoring for students with low GPA).

---

## 8. Actual vs Predicted Plot

### What is it?
A scatter plot where each dot is a student in the test set. The x-axis is their real grade, the y-axis is what the model predicted. The red dashed line (diagonal) represents perfect predictions.

### How to read it (Graph 4)
- Dots close to the diagonal → accurate predictions.
- Dots far above → model over-predicted (was too optimistic).
- Dots far below → model under-predicted (was too pessimistic).
- A tight cloud around the diagonal = low RMSE, high R².

---

## 9. Residual Plot

### What is it?
A residual is the difference between the actual grade and the predicted grade:

```
residual = actual − predicted
```

A residual plot shows residuals (y-axis) against predicted values (x-axis).

### How to read it (Graph 5)
A **good** model shows:
- Residuals scattered randomly around the y = 0 line (red dashed).
- No visible pattern or fan shape.

A **bad** model shows:
- A curved pattern → the model missed a non-linear relationship.
- A funnel shape → errors grow with the predicted value (heteroscedasticity).

In this project both models show roughly random scatter, confirming a good fit.

---

## 10. Putting It All Together

```
Raw student data
      │
      ▼
  Feature selection  ──────────────────────────────────────────┐
  (study_hours, attendance, prev_gpa, sleep_hours, tutoring)   │
      │                                                         │
      ▼                                                         ▼
  Train / Test Split                                   Correlation analysis
  80% train │ 20% test                                 (which features matter?)
      │
      ├──► Linear Regression  ──► predictions ──► R², RMSE
      │
      └──► Random Forest      ──► predictions ──► R², RMSE
                │
                └──► Feature Importance
```

| Graph | What it answers |
|---|---|
| Grade Distribution | What does the grade spread look like? |
| Correlation Heatmap | Which features predict grade most strongly? |
| Feature vs Grade Scatter | How does each feature individually relate to grade? |
| Actual vs Predicted | How accurate are each model's predictions? |
| Residual Plot | Are errors random, or is there a pattern the model missed? |
| Feature Importance | Which inputs does Random Forest rely on most? |
