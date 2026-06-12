# Day 6 – Friday, June 12, 2026

## Topic
Linear Regression on Diabetes Dataset – Solving via the Normal Equation (Closed‑Form Solution)

## Key Formula
The normal equation solves for the optimal parameters $\theta$ without iterative gradient descent:

$$\theta = (X^T X)^{-1} X^T y$$

Where:
- $X$ is the design matrix (samples × features, with a column of ones for the intercept)
- $y$ is the target vector
- $\theta$ contains the coefficients (including bias)

For the diabetes dataset (442 samples, 10 features), this yields the best linear fit in one step.

## Implementation
* **Dataset Loading:** Used `sklearn.datasets.load_diabetes()` to obtain features and target. The dataset contains 10 baseline variables (age, sex, BMI, blood pressure, and six blood serum measurements) for 442 diabetes patients. Target is a quantitative measure of disease progression one year after baseline.
* **Design Matrix Construction:** Added an intercept column of ones to $X$ so that the normal equation simultaneously solves for bias and coefficients.
* **Closed‑Form Solution:** Implemented `theta = np.linalg.inv(X.T @ X) @ X.T @ y` using NumPy’s linear algebra module. Added a small regularization term (`1e-8 * np.eye`) to the $X^T X$ matrix to handle near‑singularity.
* **Performance Comparison:** Compared the normal equation results with scikit‑learn’s `LinearRegression` to validate correctness.
* **Prediction & Evaluation:** Computed predictions on the training set and evaluated using Mean Squared Error (MSE) and $R^2$ score.

## Results
* **Optimal Parameters ($\theta$):** Learned coefficients for 10 features + intercept. BMI, serum measurements (s5 – `ltg`), and blood pressure showed the strongest positive influence on disease progression.
* **Model Performance (training set):**
  - MSE: $\approx 2900$ (baseline variance of target is large)
  - $R^2$: $\approx 0.517$ – the model explains about 52% of the variance in disease progression.
* **Comparison with scikit‑learn:** Coefficients matched up to $10^{-15}$, confirming the implementation.
* **Timing:** Normal equation solved in $\approx 2$ milliseconds, much faster than gradient descent (which would require hundreds of iterations for this dataset).

## Challenges & Limitations Faced (My Perspective)
* **Matrix Inversion Stability:** The $X^T X$ matrix for the diabetes dataset is not perfectly conditioned. Computing `np.linalg.inv` directly caused numerical warnings. I added a small Tikhonov regularization (`λ = 1e-8 * I`) to the matrix before inversion, which eliminated instability without altering results meaningfully.
* **Feature Scaling Awareness:** The normal equation is **not** sensitive to feature scaling (unlike gradient descent) because scaling cancels out in the closed form. However, I realized that the coefficients’ magnitudes reflect the original feature scales, making interpretation less intuitive. I standardized features afterward for better interpretability.
* **Lack of Built‑In Cross‑Validation:** The normal equation computes the global optimum for the training set but does not provide a way to tune hyperparameters. Overfitting is possible if the dataset has many features relative to samples (not a problem here: 10 features vs 442 samples). For higher‑dimensional data, ridge regression (normal equation with L2 penalty) would be safer.
* **Computational Cost for Large Datasets:** For datasets with thousands of features, the $O(n^3)$ inversion becomes prohibitively slow. I learned that the normal equation is only feasible when the number of features is modest (e.g., < 10,000).

## Next Steps
Day 7: Implement Ridge Regression (L2 regularization) using both the normal equation extension and gradient descent. Apply it to the diabetes dataset to see if regularization improves test generalization.
