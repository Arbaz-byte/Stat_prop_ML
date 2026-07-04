# 📘 Day 10 – Saturday, July 4

---

## 📚 Key Learnings

- **Data Generation & Preprocessing:** Created synthetic 2D binary clusters (Class –1 centered at (2.5, 5.5), Class +1 at (4.0, 4.0)) and applied Z‑score normalization to avoid scale dominance.

- **Vector Homogenization:** Augmented the feature matrix with a column of ones (`X_hom`) to absorb the bias term into the weight vector, enabling a single dot product for net input.

- **Perceptron Training Loop:** Implemented the online error‑correction rule `w += lr * y_true * X_i` with a fixed learning rate (`lr = 0.1`) and a maximum of 100 epochs.
- **Theoretical Bounds:** Understood the Perceptron Convergence Theorem and attempted to compute the radius `R` and margin `γ` from the final weight vector, but discovered that the trained model did not achieve perfect separation (some margins are negative).
- **Visualization:** Plotted the final decision boundary and the weight vector (normal to the boundary) using `plt.contourf` and an arrow.


---

## ⚠️ Struggles & Notes

- **Non‑Convergence:** The training loop ran for 100 epochs, but the margins calculated from the final weights were not all positive (`γ = –1.1442`). This indicates the data was **not linearly separable** with the given random seed (46). The algorithm therefore never converged perfectly, and the theoretical bound `K` becomes infinite (`inf`). In practice, the perceptron stops after a fixed number of epochs, yielding a finite number of mistakes (3308).

- **Why the Naive γ Search Fails:** Initially, I tried to find `γ` by testing hyperplanes through the origin (bias = 0), which gave zero margin. The correct approach is to normalise the *learned* weight vector (including bias) and compute the minimum signed distance. Even then, if separation is not perfect, `γ` will be negative, making the bound invalid.

- **Gradient Descent vs. Error Correction:** The Heaviside step function has zero derivative almost everywhere, so gradient‑based optimisation fails. Rosenblatt’s geometric rule rotates the weight vector toward misclassified positive points and away from negative ones, using the label sign to handle both error types automatically.

- **Bias Translation:** Without the bias term, the decision line is locked to the origin; homogenisation elegantly shifts the line anywhere in space.

---

## Insight: Why the Bound is Infinite

- The theorem guarantees finite `k` only if `γ > 0`. Since our trained model did not separate all points (some margins are negative), `γ` is negative, so `(R/γ)²` is not a meaningful bound. 

- To obtain a finite bound, one would need a perfectly separating hyperplane—either by choosing a different random seed or by increasing the number of epochs (but even then, linearly inseparable data will never converge).



## ⏱️ Time Spent

~5.5 hours (including coding, debugging, and visualisation)
