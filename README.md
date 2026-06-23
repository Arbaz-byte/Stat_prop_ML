# Probability & Statistics – ML Foundation

Daily implementations from scratch, following a structured 12‑month plan to become a Robotics AI Engineer.

## Structure
- `stat_notebooks/` – Statistics & probability theory implementations
- `prob_notebooks/` – Calculus & gradient descent notebooks
- `mini_projects/` – Sunday projects integrating concepts
- `src/` – Reusable Python modules

## Progress

| Day | Date | Topic | Notebook |
|-----|------|-------|----------|
| 1 | June 2 | Factorial, Perm/Comb, MSE gradient | [`Permutation_Combination.ipynb`](stat_notebooks/Permutation_Combination.ipynb), [`01_MSE_gradient_decent.ipynb`](prob_notebooks/01_MSE_gradient_decent.ipynb) |
| 2 | June 8 | Full‑batch GD, learning rate experiments | [`02_gradient_decent_linear.ipynb`](prob_notebooks/02_gradient_decent_linear.ipynb) |
| 3 | June 9 | 2D GD animation, contour plot, probability axioms | [`03_GD_2d_animation.ipynb`](prob_notebooks/03_GD_2D_Animation.ipynb) |
| 4 | June 10 | SGD from scratch, numerical nudging (Symmetric Difference), LR analysis | [`04_Stochastic_Gradient_Decent.ipynb`](prob_notebooks/04_Stochastic_Gradient_Decent.ipynb) |
| 5 | June 11 | Mini‑Batch GD, vectorized implementation, batch size & LR sweeps | [`05_Mini_Batch_Gradient_descent.ipynb`](prob_notebooks/05_Mini_Batch_Gradient_descent.ipynb) |
| 6 | June 15 | Normal Equation on Insurance Dataset (Medical Cost Prediction) | [`06_Noraml_equation.ipynb`](prob_notebooks/06_Noraml_equation.ipynb) |
| 9 | June 16 | Locally Weighted Regression (LWR) from scratch | [`07_locally_weighted_regression.ipynb`](prob_notebooks/07_locally_weighted_regression.ipynb) |
| 10 | June 17 | Logistic Regression from scratch (binary classification) | [`08_logistic_regression.ipynb`](prob_notebooks/08_binomial_logistic_regression.ipynb) |
| 11 | June 23 | Logistic Regression from scratch (Multinomial classification) | ['09_Mutinomial_logistic_regression.ipynb'](prob_notebooks/09_Mutinomial_logistic_regression.ipynb) |

## Daily Logs
- [Day 1](day1.md)
- [Day 2](day2.md)
- [Day 3](day3.md)
- [Day 4](day4.md)
- [Day 5](day5.md)
- [Day 6](day6.md)
- [Day 7](day7.md)
- [Day 8](day8.md)
- [Day 9](day9.md)
## How to Run
1. Create virtual environment: `python -m venv venv`
2. Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
3. Install requirements: `pip install -r requirements.txt`
4. Run Jupyter: `jupyter notebook`

## Notes
- **Stochastic Updates:** Successfully implemented row-by-row updates to handle high-frequency parameter mutation.
- **Gradient Verification:** Implemented the "Manual Nudge" (Symmetric Difference Quotient) to verify analytical gradients to $10^{-14}$ precision.
- **Learning Rate Research:** Documented the impact of $\alpha$ on convergence stability, including "Numerical Overflow" and "Overshooting" scenarios.
- **Vectorized Mini‑Batch:** Replaced per‑row loops with matrix operations; added design matrix, shuffling, remainder batch handling, and loss logging.
- **Hyperparameter Insights:** Found that $\alpha = 0.01$ with batch size $10$ gives smooth convergence; $\alpha = 0.25$ diverges; smaller $\alpha$ underfits within 400 epochs.


## Project: Medical Insurance Cost Prediction (Normal Equation)

- **Notebook:** [`06_Noraml_equation.ipynb`](prob_notebooks/06_Noraml_equation.ipynb)
- **Goal:** Predict medical insurance charges using closed‑form linear regression.
- **Highlights:**
  - Cleaned and encoded categorical features.
  - Applied Z‑score standardisation.
  - Implemented Normal Equation with pseudo‑inverse.
  - Achieved R² = 0.78359 on test set, matching scikit‑learn.
  - Built inference pipeline for new customer data.
