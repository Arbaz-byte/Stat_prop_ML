# Day 6 – June 15, 2026

## Topic
Normal Equation for Linear Regression – Medical Insurance Cost Prediction

## Dataset
Insurance charges (age, sex, bmi, children, smoker, region) from `insurance.csv`.

## Work Accomplished
1. Data ingestion and EDA (shape, nulls, data types).
2. Feature engineering:
   - Binary encoding (sex, smoker)
   - One‑hot encoding for region (drop_first to avoid dummy trap)
   - Target scaling (charges divided by 100 for numerical stability)
3. Z‑score standardization of features and target using `StandardScaler`.
4. Added intercept column to feature matrix.
5. Implemented **Normal Equation** using Moore‑Penrose pseudo‑inverse:
   \[
   \theta = (X^T X)^{\dagger} X^T y
   \]
6. Train/test split (80/20, random_state=42).
7. Model evaluation: MSE = 0.22926, R² = 0.78359.
8. Comparison with `sklearn.linear_model.LinearRegression` – identical predictions (max diff = 0.0).
9. Cross‑validation (5‑fold) gave mean R² = 0.74686 ± 0.02487.
10. Residual analysis: homoscedasticity mostly holds, slight pattern at high charges.
11. Inference pipeline: function `predict_insurance_charges()` that scales new inputs using the original scalers and returns dollar amounts.

## Key Learnings
- The Normal Equation gives an exact solution without learning rate tuning.
- Feature scaling is not required for the Normal Equation mathematically, but it improves numerical stability.
- The dummy variable trap is avoided by dropping one category – otherwise \(X^T X\) becomes singular.
- Cross‑validation provides a more realistic generalisation estimate than a single train/test split.

## Challenges Faced During Implementation

## 1. **File Path Error**
- **Problem**: `FileNotFoundError` when reading CSV
- **Cause**: Working directory was `prob_notebooks`, but data was in parent directory
- **Solution**: Used relative path `../Data_sets/insurance.csv`

## 2. **Normal Equation Formula Error**
- **Problem**: `ValueError: matmul dimension mismatch`
- **Cause**: Used `X @ X.T` instead of `X.T @ X`
- **Solution**: Corrected formula to: `θ = (X^T X)^(-1) X^T 

## 3. **Array Comparison Error**
- **Problem**: `ValueError: The truth value of an array with more than one element is ambiguous`
- **Cause**: Comparing arrays directly with `<` operator
- **Solution**: Used `.all()` method or `np.allclose()`

## 4. **StandardScaler Dimension Mismatch**
- **Problem**: `ValueError: X has 8 features, but StandardScaler is expecting 1 features`
- **Cause**: Used same `scaler` for both X (8 features) and y (1 feature), second fit overwrote it
- **Solution**: Created separate scalers: `x_scaler` and `y_scaler` Challenges Faced During Implementation

## Next Steps
- Complete documentation and push this notebook.
- Implement the same logic using SGD and Mini‑Batch GD (already done).
- July plan: integrate deployment (FastAPI, Docker, cloud).

## Time Spent
~13 hours (spread over several days due to personal break, but completed on June 15)


