

# Day 9 – June 23, 2026

**Topic** 
Multinomial Logistic Regression (Softmax Regression) from Scratch – Visualizing Multiclass Decision Space Topology

**Dataset** 
Synthetic 3‑class ($K=3$) continuous spatial dataset generated via localized normal distributions ($1,000$ total samples across $2$ physical coordinates).

**Work Accomplished** 
*   **Data ingestion and EDA**: Generated and analyzed class structural configurations (shapes, centers, and standard deviations).

*   **Feature engineering**:
    *   Prepended unit vector column ($1.0$) to raw features to implicitly handle the baseline intercept vector.
    *   One‑hot target encoding using coordinate-basis array bitmasks to translate discrete scalar class indices ($y \in \{0, 1, 2\}$) into matching orthogonal array spaces ($Y \in \mathbb{R}^{m \times K}$).

*   **Implemented Vectorized Training Pipeline from Scratch**:
    *   Numerical stability engineering: Integrated row-wise shift‑invariance property ($z - \max(z)$) inside the Softmax activation function to eliminate explosive floating-point overflow (`NaN`).
    *   Cost function formulation: Built global loss calculations using the matrix Trace operator ($\text{Tr}$) over element-wise log-probability densities:

    $$J(\Theta) = -\frac{1}{m} \text{Tr}\left( Y^T \ln(H) \right)$$
 
    *   Vector calculus optimization: Derived and implemented a clean, parallelized gradient descent engine that bypasses explicit loop paths:
    $$\Theta := \Theta - \alpha \cdot \left[ \frac{1}{m} X^T (H - Y) \right]$$
 
*   **Train/test split**: Partitioned raw matrices using an 80/20 division ratio (`random_state=56`).
*   **Model evaluation**: Evaluated structural metrics via macro‑averaging heuristics: Accuracy, Macro Precision, Macro Recall, and Macro F1‑Score.
*   **Deconstructed Class Confusion Matrix**: Analyzed the main diagonal entries for correct class allocations and evaluated off-diagonal cells to pinpoint specific misclassification zones.
*   **Comparison with Benchmark**: Cross‑validated against `sklearn`’s `OneVsRestClassifier` wrapped with an `L-BFGS` Newton-curvature solver over a 5‑fold cross-validation layout.
*   **Spatial Decision Topology Mapping**: Built a fine-resolution $300 \times 300$ continuous meshgrid, mapped boundary shifts using soft opacity background fills (`contourf`), and projected out‑of‑sample test coordinates as large high-contrast stars.
*   **Inference pipeline**: Function `predict_multiclass()` built with adaptive input validation to handle both 1D coordinates and multi-row batches while dynamically auto-injecting missing intercept vectors.

**Key Learnings** 
*   The Softmax function acts as a vector-valued activation layer that safely maps raw logits into a valid probability simplex where row outputs strictly sum to 1.0.
*   Subtracting the maximum logit per row from the input vector preserves exact probability outputs while completely removing floating-point calculation risks.
*   During hard categorical inference, evaluating the raw logit scores via index maximization (`np.argmax`) yields the exact same classification label as evaluating the fully transformed Softmax matrix, allowing for major compute speedups.
*   Macro-averaging calculates diagnostic scores independently per class before computing an unweighted mean. This ensures any optimization failures in smaller classes are highlighted immediately rather than hidden by a high global accuracy score.

**Challenges Faced During Implementation** 
1. **Exponent Floating-Point Overflow Error** 
   *   *Problem*: Cost function returning `NaN` values after a few hundred training iterations. 
   *   *Cause*: Raw logits grew larger than 709, causing the standard exponential function `np.exp(z)` to break double-precision limits and return infinite boundaries. 
   *   *Solution*: Redesigned the activation code to extract the row maximum and implement the mathematical property of shift‑invariance: `np.exp(z - np.max(z, axis=1, keepdims=True))`. 
2. **Logarithm of Zero Domain Crash** 
   *   *Problem*: Runtime warning triggered: `divide by zero encountered in log`. 
   *   *Cause*: When the model made a 100% confident but incorrect prediction, the alternate true class probability dropped to exactly 0.0, making $\ln(0)$ mathematically undefined ($-\infty$). 
   *   *Solution*: Added a tiny epsilon threshold constant inside the core cross-entropy logging array: `np.log(H + 1e-15)`. 
3. **Inference Input Matrix Dimension Mismatch** 
   *   *Problem*: `ValueError: matmul dimension mismatch` when running a single test point coordinate vector through the pipeline. 
   *   *Cause*: Passing a simple 1D array dropped the necessary matrix dimensions, and the input lacked the required leading intercept bias unit (1.0). 
   *   *Solution*: Added input check guards to catch and fix shapes using `ndim == 1` vector expansion and auto-injected a column of 1s if feature lengths missed the param shape requirement. 
4. **Target Multi-Class Evaluation Error** 
   *   *Problem*: Precision and recall functions threw classification format errors. 
   *   *Cause*: Passing a one-hot encoded matrix to evaluation metrics that expect single-column target indices. 
   *   *Solution*: Flattened targets using `.ravel()` and extracted index labels from probabilities using `np.argmax(probs, axis=1)`. 
 

**Time Spent** 
*   ~10 hours (focused block development across synthetic mapping tests, structural vector derivations, and troubleshooting underflow errors).
