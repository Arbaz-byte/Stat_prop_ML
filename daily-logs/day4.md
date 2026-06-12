# Day 4 – Wednesday, June 10, 2026

## Topic
Stochastic Gradient Descent (SGD) for Linear Regression – Mathematical Derivations and Parameter Updates

## Key Formula
Instantaneous localized cost function for a single randomly selected training row $i$:
$$J_{\text{single}}(w, b) = \left( (w X^{(i)} + b) - y^{(i)} \right)^2$$

Simultaneous coordinate update equations applied immediately per row:
$$w := w - \alpha \cdot \left[ 2 \cdot X^{(i)} \left( (wX^{(i)} + b) - y^{(i)} \right) \right]$$
$$b := b - \alpha \cdot \left[ 2 \cdot \left( (wX^{(i)} + b) - y^{(i)} \right) \right]$$

## Implementation
* **Isolated Single-Row Processing:** Completely eliminated matrix operations and summation arrays ($\sum$) for gradients to compute instant adjustments using pure Python loops and standard arrays.
* **In-Place Mutation Loop:** Configured the parameters ($w$ and $b$) to overwrite their stored values immediately inside the step loop, ensuring subsequent row predictions instantly benefit from the previous row's adjustment.
* **Deterministic Trace Override:** Hardcoded the initial shuffle sequence (`[2, 0, 1]`) during the first epoch pass to cross-verify code calculations with hand-derived mathematical trace outputs down to 3 decimal places.

## Results
* **Starting State:** $w = 1.000$, $b = 0.000$ (Predictions: $y = 1.0X$)
* **End of Epoch 1 State:** $w = 0.964$, $b = 0.072$ (Final Model: $y = 0.964X + 0.072$)
* **Unseen Inference Prediction:** Size $2.5 \implies$ Predicted Value: $2.482$ (Decoded Market Valuation: $\$248,200.00$)
* **Mathematical Parallels:** Manual numerical nudge calculation using the Symmetric Difference Quotient perfectly matched analytical calculus derivatives within a margin of $10^{-14}$.

## Challenges & Limitations Faced (My Perspective)
* **The Parameters Reset Bug:** I realized that if the initial weights $w$ and $b$ are not explicitly reset to $0.0$ *inside* the macro cross-validation loop, subsequent learning rate benchmarks inherit unoptimized weights from previous runs, corrupting the comparison data.
* **Numerical Overflow and Crashes:** When testing aggressive learning rates ($\alpha = 0.25$), the parameter steps overshot the minimum valley so violently that values expanded exponentially. This caused standard `math.isnan()` evaluations to crash due to floating-point infinity values (`inf`) before resolving to `NaN`. I solved this by transitioning to `np.isnan()` and implementing an active anti-crash gradient shield.
* **Volatile Optimization Path:** Because the parameters react aggressively to individual data rows, the path down the cost bowl bounces around heavily. It becomes highly sensitive to target noise, meaning a constant learning rate will bounce around the bottom center forever instead of locking onto the absolute minimum point.

## Next Steps
Day 5: Upgrading to Matrix Vectorization and Mini-Batch Shuffling Engine to group row arrays together for balanced convergence stability.
