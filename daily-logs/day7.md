# Day 16 – June 16, 2026

## Videos Watched
- CS229 Lecture 3 (first half – Locally Weighted Regression)
- Decode AI/ML: 10.28 (PMF/CDF discrete), 10.29 (PDF/CDF continuous)

## Key Learnings
- LWR is non‑parametric: no explicit training; prediction uses all training data each time.
- The Gaussian weight function gives more influence to nearby points.
- Bandwidth \(\tau\) controls the smoothness: small \(\tau\) overfits, large \(\tau\) underfits.
- LWR can fit non‑linear functions without polynomial feature engineering.

## Code Implemented
- `lwr_predict()` from scratch using weighted least squares.
- Comparison of \(\tau\) values (0.01, 0.05, 0.1, 0.5, 1.0) on a sine wave dataset.
- Comparison with global OLS (LinearRegression).

## Struggles / Notes
- (Write any difficulties: e.g., matrix inversion stability, choosing \(\tau\))
- Using `np.linalg.pinv` avoided singular matrix issues.

## Next Steps
- Logistic regression (binary classification) – CS229 Lecture 3 (second half) & Lecture 4.

## Time Spent
~3 hours
