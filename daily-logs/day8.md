# Day 8: 19 June 2026

### 🎬 Videos Watched
*   **CS229 Lecture 3** (Second half – Logistic Regression)
*   **CS229 Lecture 4** (First 20 min – GLM & Logistic derivation)

### 💡 Key Learnings
*   **Probabilistic Core**: Logistic regression models the probability of a binary outcome using the squishing mechanics of the sigmoid function.
*   **Geometric Space**: The resulting decision boundary forms a rigid linear hyper-plane within the feature space.
*   **Objective Origin**: The binary cross-entropy loss function is derived directly from Maximum Likelihood Estimation (MLE).
*   **Mathematical Elegance**: The final gradient descent update rule matches the exact form of linear regression, differing only by the non-linear wrapping of the hypothesis function.

### 💻 Code Implemented
*   **Functions**: `sigmoid()`, `logistic_loss()`, `logistic_gradient()`, `gradient_descent_logistic()`, and `predict()`.
*   **Metrics Matrix**: Computed Accuracy, Precision, Recall, F1 Score, and compiled the Confusion Matrix layout.
*   **Visualizations**: Created spatial plots of the decision boundary.
*   **Benchmarking**: Compared final matrix coefficients and convergence metrics against `sklearn.linear_model.LogisticRegression`.

### ⏱️ Time Invested
*   **Duration**:  1 Day
