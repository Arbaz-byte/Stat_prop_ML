# Day 5 – Thursday, June 11, 2026

## Topic
Mini‑Batch Gradient Descent for Linear Regression – Vectorized Implementation and Hyperparameter Analysis

## Key Formula
Mean Squared Error (MSE) over a mini‑batch $\mathcal{B}$ of size $B$:

$$J_{\text{mini}}(\theta) = \frac{1}{B} (\mathbf{y}_{\text{pred}} - \mathbf{y}_b)^T (\mathbf{y}_{\text{pred}} - \mathbf{y}_b)$$

Vectorized gradient with respect to $\theta = [b, w]^T$:

$$\nabla_\theta J_{\text{mini}} = \frac{2}{B} X_b^T (\mathbf{y}_{\text{pred}} - \mathbf{y}_b)$$

Parameter update rule (simultaneous for bias and weight):

$$\theta := \theta - \alpha \cdot \nabla_\theta J_{\text{mini}}$$

## Implementation
* **Design Matrix Augmentation:** Added a leading column of ones to the feature matrix $X$, turning it into $X_{\text{design}}$ of shape $(m, 2)$. This unifies bias and weight into a single vector $\theta$.

* **Mini‑Batch Shuffling Engine:** At the start of each epoch, the entire dataset is shuffled using a permuted index array, then split into consecutive blocks of size $B$. The remainder batch (if $m$ not divisible by $B$) is processed separately.

* **Vectorized Gradient Computation:** Replaced the per‑row Python loops with a single matrix multiplication: `(2.0/B) * np.dot(X_batch.T, error)`. This eliminates explicit summation loops and leverages NumPy’s underlying BLAS routines.

* **Loss Curve Logging:** After each epoch, the full‑dataset MSE is computed and stored, allowing real‑time monitoring of convergence.

## Results
* **Baseline Configuration:** $B = 10$, $\alpha = 0.01$, epochs $= 500$.
  - Initial loss (epoch 0): $\approx 85.3$ (random initial $\theta = [0, 0]$).
  - Final loss (epoch 500): $\approx 2.1$.
  - Learned parameters: $w_{\text{final}} = 2.502$, $b_{\text{final}} = 4.956$ (true: $w=2.5, b=5.0$).

* **Learning Rate Sweep (fixed $B=10$, 400 epochs):**
  - $\alpha = 0.0001$: loss decreases slowly, final $w=3.12, b=0.74$ – underfitted.
  - $\alpha = 0.01$: smooth convergence, final $w=2.51, b=4.93$ – optimal.
  - $\alpha = 0.25$: loss explodes to `NaN` after 10 epochs – divergent.

* **Batch Size Comparison (fixed $\alpha=0.01$, 500 epochs):**
  - $B=1$ (pure SGD): noisy loss curve, final $w=2.49, b=5.11$ – higher variance.
  - $B=10$: balanced, smooth curve.
  - $B=50$ (almost full batch): very smooth but slower per epoch; final $w=2.50, b=5.02$.

## Challenges & Limitations Faced 
* **Remainder Batch Overlook:** When $m$ was not divisible by $B$, the leftover samples were initially ignored. This caused the final model to be biased because those samples never contributed to updates. I added a dedicated remainder batch loop that uses the current batch size `B_remain` for the gradient denominator.

* **Loss Plot Interpretation:** The loss curve on a log scale can be deceptive – a straight line (exponential decay) looks like linear improvement, but the actual improvement per epoch diminishes. I had to remind myself that the goal is convergence, not a perfect exponential drop.

## Next Steps
Day 6: Next using Normal Equation framework to handle multivariate linear regression (multiple features) and compare performance with different feature scalings.
