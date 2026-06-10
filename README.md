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
| 1 | June 2 | Factorial, Perm/Comb, MSE gradient | [`Permutation_Combination.ipynb`](stat_notebooks/Permutation_Combination.ipynb), ['01_MSE_gradient_decent.ipynb'](prob_notebooks/01_MSE_gradient_decent.ipynb) |
| 2 | June 8 | Full‑batch GD, learning rate experiments | [`02_gradient_decent_linear.ipynb `](prob_notebooks/02_gradient_decent_linear.ipynb) |
| 3 | June 9 | 2D GD animation, contour plot, probability axioms | [`03_GD_2d_animation.ipynb`](prob_notebooks/ 03_GD_2D_Animation.py) |
| 4 | June 10 | SGD from scratch, numerical nudging (Symmetric Difference), LR analysis | [` 04_Stochastic_Gradient_Decent.ipynb`](prob_notebooks/04_Stochastic_Gradient_Decent.ipynb) |

## Daily Logs
- [Day 2](day2.md)
- [Day 3](day3.md)
- [Day 4](day4.md)

## How to Run
1. Create virtual environment: `python -m venv venv`
2. Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
3. Install requirements: `pip install -r requirements.txt`
4. Run Jupyter: `jupyter notebook`

## Notes
- **Stochastic Updates:** Successfully implemented row-by-row updates to handle high-frequency parameter mutation.
- **Gradient Verification:** Implemented the "Manual Nudge" (Symmetric Difference Quotient) to verify analytical gradients to $10^{-14}$ precision.
- **Learning Rate Research:** Documented the impact of $\alpha$ on convergence stability, including "Numerical Overflow" and "Overshooting" scenarios.
- Daily pushes ensure a public, accountable learning trail.
