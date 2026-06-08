# Day 2 – Monday, June 8

## Code Implemented
- `gradient_descent()` full-batch for linear regression  
- Loss history tracking  
- Learning rate comparison (0.001, 0.01, 0.1)

## Verification
- Final parameters matched `np.polyfit` within <1e-2  
- Loss curves decreasing (except divergent rate)

## What I Struggled With

### Learning Rate Instability
I experimented with three different learning rates. When I used `α = 0.1`, the model diverged badly — weights and bias exploded, loss went to infinity, and I started getting `nan` values. This clearly showed that a large learning rate can overshoot the minimum and make gradient descent unstable.

### Runtime Warnings
I received several `RuntimeWarning: overflow encountered` and `invalid value encountered` warnings during training with high learning rates. These warnings appeared because the predictions became extremely large, causing the squared error in MSE to overflow.

### Better Loss Visualization
I initially plotted loss using scatter, but later realized a line plot (`plt.plot(loss_hist)`) is much more informative. Adding `plt.yscale('log')` helped me clearly see the convergence behavior, especially when loss drops rapidly at the beginning.

### Verifying with `np.polyfit`
Comparing my Gradient Descent result with `np.polyfit(X, y, 1)` was very useful. It helped me validate that my implementation was correct when using a proper learning rate (`0.01`). With the right learning rate, both methods gave almost identical `w` and `b`.

### Final Takeaway
The biggest challenge was choosing the right learning rate. Too high (`0.1`) → divergence and `nan`. Too low (`0.001`) → very slow convergence. Moderate (`0.01`) gave stable and fast convergence. This experiment taught me how sensitive Gradient Descent is to the learning rate hyperparameter.
